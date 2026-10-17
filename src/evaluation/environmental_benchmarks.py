"""Benchmarks: per-condition detection accuracy, enhancement A/B, model comparison, tracker comparison, speed.

Rules this module follows:
* Real per-condition metrics are only reported for conditions that have labelled test images
  (``conditions.csv``) and at least ``min_images`` of them; otherwise the condition is listed as skipped.
* Synthetic-corruption results (fog/rain/low light applied to clean test images) are labelled
  ``synthetic`` everywhere and never mixed with real-condition numbers.
* Speed numbers come from timing on the machine the benchmark runs on; nothing is extrapolated.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import BaseDetector
from src.detection.preprocessing import Preprocessor
from src.evaluation.detection_metrics import evaluate_detections, load_yolo_ground_truth
from src.training.augmentations import CORRUPTIONS

ENV_CATEGORIES = ["daylight", "low_light", "night", "backlit_glare", "rain", "fog", "indoor", "crowded",
                  "heavy_occlusion", "small_distant"]


def _predict_all(detector: BaseDetector, images: dict[str, np.ndarray], transform=None) -> dict:
    preds = {}
    for k, img in images.items():
        x = transform(img) if transform else img
        d = detector.detect(x)
        preds[k] = (d.xyxy, d.conf)
    return preds


def load_images(paths: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for p in paths:
        im = cv2.imread(str(p))
        if im is not None:
            out[str(p)] = im
    return out


def per_condition_detection(detector: BaseDetector, image_paths: list[str], conditions: dict[str, str],
                            min_images: int = 20, conf: float = 0.25) -> dict:
    """``conditions``: {image_stem: condition}. Images without a label are grouped as 'unlabelled' (not reported)."""
    gts = load_yolo_ground_truth(image_paths)
    imgs = load_images(list(gts))
    preds = _predict_all(detector, imgs)
    groups: dict[str, list[str]] = defaultdict(list)
    for k in gts:
        groups[conditions.get(Path(k).stem, "unlabelled")].append(k)
    out = {"overall": evaluate_detections(preds, gts, conf).to_dict(), "conditions": {}, "skipped": {}}
    for cond, keys in sorted(groups.items()):
        if cond == "unlabelled":
            out["skipped"][cond] = f"{len(keys)} images without a condition label"
        elif len(keys) < min_images:
            out["skipped"][cond] = f"only {len(keys)} labelled images (< {min_images})"
        else:
            out["conditions"][cond] = evaluate_detections({k: preds[k] for k in keys}, {k: gts[k] for k in keys}, conf).to_dict()
    return out


def synthetic_corruption_benchmark(detector: BaseDetector, image_paths: list[str], corruptions: list[str] | None = None,
                                   severities: tuple[int, ...] = (1, 2, 3), conf: float = 0.25) -> dict:
    gts = load_yolo_ground_truth(image_paths)
    imgs = load_images(list(gts))
    res = {"type": "synthetic", "clean": evaluate_detections(_predict_all(detector, imgs), gts, conf).to_dict(), "corrupted": {}}
    for name in corruptions or list(CORRUPTIONS):
        fn = CORRUPTIONS[name]
        for s in severities:
            preds = _predict_all(detector, imgs, lambda im, fn=fn, s=s: fn(im, s))
            res["corrupted"][f"{name}@{s}"] = evaluate_detections(preds, gts, conf).to_dict()
    return res


def compare_enhancement(detector: BaseDetector, image_paths: list[str], methods: tuple[str, ...] = ("clahe", "gamma", "clahe_gamma", "dehaze"),
                        darken: str | None = None, severity: int = 2, conf: float = 0.25) -> dict:
    """Detection metrics with vs without each enhancement. ``darken`` optionally applies a synthetic corruption
    first (e.g. ``low_light``) to emulate the conditions enhancement is meant for (labelled as synthetic)."""
    gts = load_yolo_ground_truth(image_paths)
    imgs = load_images(list(gts))
    if darken:
        imgs = {k: CORRUPTIONS[darken](v, severity) for k, v in imgs.items()}
    pre = Preprocessor(enabled=True, method="clahe")
    out = {"input": f"synthetic {darken}@{severity}" if darken else "original", "none": evaluate_detections(_predict_all(detector, imgs), gts, conf).to_dict()}
    for m in methods:
        t0 = time.perf_counter()
        preds = _predict_all(detector, imgs, lambda im, m=m: pre.enhance(im, m))
        r = evaluate_detections(preds, gts, conf).to_dict()
        r["seconds"] = round(time.perf_counter() - t0, 3)
        out[m] = r
    return out


def compare_models(detectors: dict[str, BaseDetector], image_paths: list[str], conf: float = 0.25) -> dict:
    """Baseline vs fine-tuned (or any set of detectors) on the same test images."""
    gts = load_yolo_ground_truth(image_paths)
    imgs = load_images(list(gts))
    return {name: evaluate_detections(_predict_all(d, imgs), gts, conf).to_dict() for name, d in detectors.items()}


def benchmark_speed(detector: BaseDetector, frames: list[np.ndarray], warmup: int = 3, repeats: int = 1) -> dict:
    for f in frames[:warmup]:
        detector.detect(f)
    times = []
    for _ in range(repeats):
        for f in frames:
            t0 = time.perf_counter()
            detector.detect(f)
            times.append(time.perf_counter() - t0)
    ms = np.asarray(times) * 1000
    return {"frames": len(times), "fps": round(1000 / ms.mean(), 2), "latency_ms_mean": round(float(ms.mean()), 2),
            "latency_ms_p50": round(float(np.percentile(ms, 50)), 2), "latency_ms_p95": round(float(np.percentile(ms, 95)), 2)}


# ------------------------------------------------------------------------ reporting
def _fmt(v) -> str:
    if isinstance(v, float):
        return "n/a" if not np.isfinite(v) else f"{v:.3f}"
    return str(v)


def table(rows: dict[str, dict], cols: list[str]) -> str:
    head = "| name | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    return head + "".join(f"| {k} | " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |\n" for k, r in rows.items())


def bar_plot(rows: dict[str, dict], metric: str, path: str | Path, title: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    names = list(rows)
    vals = [rows[n].get(metric, 0) or 0 for n in names]
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(names) + 2), 3.5))
    ax.bar(names, vals, color="#3b6ea5")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, 1 if max(vals, default=0) <= 1 else None)
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path
