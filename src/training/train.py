"""Fine-tune an Ultralytics YOLO person detector from pretrained weights.

Usage::

    python cli.py train --config configs/training.yaml --yes [--set epochs=5 batch=8]

Guarantees: seeds are fixed (Python/NumPy/PyTorch + Ultralytics ``seed``/``deterministic``), hardware is
detected first, nothing is downloaded or trained without explicit confirmation, checkpoints (``last.pt``,
``best.pt``) are written every epoch and training can be resumed, and a ``run_metadata.json`` records the
config, hardware, dataset provenance and final metrics.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import yaml

from src.training.checkpoint_manager import CheckpointManager
from src.utils.config import PROJECT_ROOT, ConfigError, deep_merge
from src.utils.device import hardware_report, resolve_device

log = logging.getLogger(__name__)

# Approximate sizes of official checkpoints (MB) used for the download guard.
KNOWN_SIZES_MB = {"n": 6, "s": 20, "m": 45, "l": 55, "x": 120}


class TrainingError(RuntimeError):
    pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_training_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    path = Path(path or PROJECT_ROOT / "configs" / "training.yaml")
    if not path.exists():
        raise ConfigError(f"Training config not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    return deep_merge(cfg, overrides or {})


def _estimate_download_mb(weights: str) -> float | None:
    stem = Path(weights).stem.lower()
    if Path(weights).exists() or (PROJECT_ROOT / "models" / Path(weights).name).exists():
        return 0.0
    for suffix, mb in KNOWN_SIZES_MB.items():
        if stem.endswith(suffix):
            return mb
    return None


def build_train_args(cfg: dict, device: str) -> dict:
    data = Path(cfg["data"])
    if not data.is_absolute():
        data = PROJECT_ROOT / data
    if not data.exists():
        raise TrainingError(
            f"Dataset file {data} not found. Prepare a dataset first (README > Datasets): download/import, "
            "`cli.py datasets convert`, `cli.py datasets validate`, `cli.py datasets split`."
        )
    project = Path(cfg.get("project", "models/runs"))
    args = {
        "data": str(data),
        "epochs": int(cfg["epochs"]),
        "batch": cfg.get("batch", 16),
        "imgsz": int(cfg.get("imgsz", 640)),
        "lr0": float(cfg.get("lr0", 0.0002)),
        "lrf": float(cfg.get("lrf", 0.01)),
        "optimizer": cfg.get("optimizer", "auto"),
        "weight_decay": float(cfg.get("weight_decay", 0.0005)),
        "warmup_epochs": float(cfg.get("warmup_epochs", 1)),
        "warmup_bias_lr": float(cfg.get("warmup_bias_lr", 0.0)),
        "patience": int(cfg.get("patience", 10)),
        "amp": bool(cfg.get("amp", True)) and device.startswith("cuda"),
        "workers": int(cfg.get("workers", 4)),
        "cache": cfg.get("cache", False),
        "single_cls": bool(cfg.get("single_cls", False)),
        "seed": int(cfg.get("seed", 42)),
        "deterministic": True,
        "device": "cpu" if device == "cpu" else device.replace("cuda:", ""),
        "project": str(project if project.is_absolute() else PROJECT_ROOT / project),
        "name": cfg.get("name", "person_finetune"),
        "exist_ok": False,
        "val": bool(cfg.get("val", True)),
        "plots": bool(cfg.get("plots", True)),
        "verbose": True,
    }
    if cfg.get("freeze") is not None:
        args["freeze"] = cfg["freeze"]
    args.update(cfg.get("augment", {}) or {})
    return args


def train(cfg: dict, confirm: bool = False, allow_large: bool = False) -> dict:
    """Run fine-tuning. Returns a dict with the run dir, best/last checkpoints and final metrics."""
    if not confirm:
        raise TrainingError("Training is compute-intensive and may download weights. Re-run with --yes to confirm.")
    hw = hardware_report()
    device = resolve_device(cfg.get("device", "auto"))
    log.info("Hardware: %s | training device: %s", {k: hw.get(k) for k in ("cpu_count", "ram_gb", "cuda_available", "gpus")}, device)
    if device == "cpu":
        log.warning("Training on CPU is very slow; use it only for smoke tests (few epochs, small imgsz).")
    weights = cfg.get("weights", "yolo26n.pt")
    mb = _estimate_download_mb(weights)
    if mb and mb > float(cfg.get("max_auto_download_mb", 60)) and not allow_large:
        raise TrainingError(f"'{weights}' needs a ~{mb:.0f} MB download. Pass --allow-large to permit it.")
    set_seed(int(cfg.get("seed", 42)))

    ckpts = CheckpointManager(PROJECT_ROOT / cfg.get("project", "models/runs"))
    resume = ckpts.resume_path(cfg.get("resume", False), cfg.get("name"))

    from ultralytics import YOLO

    from src.detection.detector import resolve_weights
    from src.utils.privacy import disable_third_party_telemetry

    disable_third_party_telemetry()

    if resume:
        model = YOLO(str(resume))
        results = model.train(resume=True)
    else:
        args = build_train_args(cfg, device)
        env = cfg.get("environmental", {}) or {}
        if env.get("enabled"):
            args["data"] = _with_environmental_train(args["data"], env, int(cfg.get("seed", 42)))
        model = YOLO(resolve_weights(weights))
        if cfg.get("keep_pretrained_head", True):
            args["data"] = _with_pretrained_class_names(args["data"], model.names)
        results = model.train(**args)
    run_dir = Path(model.trainer.save_dir)
    best, last = ckpts.best(run_dir), ckpts.last(run_dir)
    metrics = {}
    rd = getattr(results, "results_dict", None)
    if isinstance(rd, dict):
        metrics = {k: float(v) for k, v in rd.items()}
    from src.data.dataset_registry import DatasetRegistry

    CheckpointManager.write_metadata(run_dir, {
        "config": cfg, "device": device, "hardware": hw, "best": best, "last": last,
        "metrics": metrics, "datasets": DatasetRegistry().all(),
    })
    return {"run_dir": run_dir, "best": best, "last": last, "metrics": metrics}


def _with_pretrained_class_names(data_yaml: str, names) -> str:
    """Keep the pretrained detection head by training with the checkpoint's own class list.

    Our datasets have one class (``0: person``). Training a COCO checkpoint on ``nc=1`` makes Ultralytics
    build a fresh, randomly initialised classification head, which throws away the pretrained person
    classifier and needs many epochs to recover. COCO's class 0 is also ``person``, so we write a data yaml
    with the checkpoint's full name list: labels stay valid, the head is reused, and the other classes
    simply receive no positives. At inference the detector only keeps the person class anyway.
    """
    names = dict(names) if not isinstance(names, dict) else names
    d = yaml.safe_load(Path(data_yaml).read_text())
    ds_names = d.get("names", {})
    ds_names = dict(enumerate(ds_names)) if isinstance(ds_names, list) else ds_names
    if len(ds_names) != 1 or str(ds_names.get(0, "")).lower() != "person" or str(names.get(0, "")).lower() != "person":
        return data_yaml
    if len(names) == 1:
        return data_yaml
    d["names"] = {int(k): v for k, v in names.items()}
    out = Path(data_yaml).with_name(Path(data_yaml).stem + "_pretrained_names.yaml")
    out.write_text(yaml.safe_dump(d, sort_keys=False))
    log.info("Keeping the pretrained %d-class head (person = class 0): %s", len(names), out)
    return str(out)


def _with_environmental_train(data_yaml: str, env: dict, seed: int) -> str:
    """Create augmented copies of the train split and a data yaml pointing at train + copies."""
    from src.training.augmentations import augment_split

    d = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(d.get("path", Path(data_yaml).parent))
    train_list = Path(d["train"]) if Path(d["train"]).is_absolute() else root / d["train"]
    if not train_list.exists():
        raise TrainingError(f"Train split file not found: {train_list}")
    if train_list.suffix != ".txt":
        raise TrainingError("Environmental augmentation needs a split file (train.txt) produced by `datasets split`.")
    out_dir = root / "env_aug"
    new_list = augment_split(train_list, out_dir, copies=int(env.get("copies", 1)), seed=seed, probs=env.get("probs") or None)
    d2 = dict(d)
    d2["train"] = str(new_list.resolve())
    out = Path(data_yaml).with_name("data_env_aug.yaml")
    out.write_text(yaml.safe_dump(d2, sort_keys=False))
    log.info("Environmental augmentation: train list %s", new_list)
    return str(out)
