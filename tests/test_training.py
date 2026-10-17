from pathlib import Path

import cv2
import numpy as np
import pytest

from src.training.augmentations import CORRUPTIONS, DEFAULT_PROBS, EnvironmentalAugmenter, augment_split
from src.training.checkpoint_manager import CheckpointManager
from src.training.train import TrainingError, build_train_args, train


def test_augmenter_is_seeded_and_shape_preserving():
    img = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    a1, ap1 = EnvironmentalAugmenter({k: 1.0 for k in DEFAULT_PROBS}, seed=5)(img, np.array([[10, 10, 60, 100.0]]))
    a2, ap2 = EnvironmentalAugmenter({k: 1.0 for k in DEFAULT_PROBS}, seed=5)(img, np.array([[10, 10, 60, 100.0]]))
    assert a1.shape == img.shape and a1.dtype == np.uint8
    assert ap1 == ap2 and np.array_equal(a1, a2)
    assert not ({"fog", "low_light"} <= set(ap1))  # mutually exclusive


def test_corruptions_all_severities():
    img = np.full((64, 64, 3), 128, np.uint8)
    for name, fn in CORRUPTIONS.items():
        for s in (1, 2, 3):
            assert fn(img, s).shape == img.shape, name


def test_augment_split_refuses_val_and_copies_labels(tmp_path):
    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    for i in range(3):
        cv2.imwrite(str(imgs / f"a{i}.jpg"), np.full((64, 64, 3), 100, np.uint8))
        (lbls / f"a{i}.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    val = tmp_path / "val.txt"
    val.write_text("\n".join(str(p) for p in imgs.glob("*.jpg")))
    with pytest.raises(ValueError, match="training only"):
        augment_split(val, tmp_path / "aug")
    train_list = tmp_path / "train.txt"
    train_list.write_text("\n".join(str(p) for p in imgs.glob("*.jpg")))
    out = augment_split(train_list, tmp_path / "aug", copies=2, seed=0, probs={k: 1.0 for k in DEFAULT_PROBS})
    items = out.read_text().split()
    assert len(items) == 9
    for p in items[3:]:
        assert (tmp_path / "aug" / "labels" / f"{Path(p).stem}.txt").read_text() == "0 0.5 0.5 0.2 0.4\n"


def test_train_requires_confirmation_and_dataset():
    with pytest.raises(TrainingError, match="--yes"):
        train({"data": "x.yaml", "epochs": 1})
    with pytest.raises(TrainingError, match="not found"):
        build_train_args({"data": "/nonexistent/data.yaml", "epochs": 1}, "cpu")


def test_checkpoint_manager(tmp_path):
    run = tmp_path / "runs" / "exp1" / "weights"
    run.mkdir(parents=True)
    (run / "last.pt").write_bytes(b"x")
    cm = CheckpointManager(tmp_path / "runs")
    assert cm.resume_path(True) == run / "last.pt"
    assert cm.best(run.parent) is None
    with pytest.raises(FileNotFoundError):
        CheckpointManager(tmp_path / "empty").resume_path(True)


def test_pretrained_head_names_kept(tmp_path):
    import yaml

    from src.training.train import _with_pretrained_class_names

    d = tmp_path / "data.yaml"
    d.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "train.txt", "val": "val.txt", "names": {0: "person"}}))
    out = _with_pretrained_class_names(str(d), {0: "person", 1: "bicycle", 2: "car"})
    names = yaml.safe_load(open(out))["names"]
    assert names[0] == "person" and len(names) == 3
    # a checkpoint whose class 0 is not person must not be relabelled
    assert _with_pretrained_class_names(str(d), {0: "cat", 1: "person"}) == str(d)
