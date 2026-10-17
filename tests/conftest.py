"""Shared fixtures. Tests use synthetic frames and tiny generated clips; nothing is downloaded.

Tests that need the real pretrained detector are skipped unless ``models/yolo26n.pt`` exists
(run ``python cli.py track <any video>`` once, or download it manually, to enable them).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.detector import BaseDetector, Detections  # noqa: E402
from src.utils.sample_video import make_synthetic_sequence, write_frames  # noqa: E402

WEIGHTS = ROOT / "models" / "yolo26n.pt"


class ScriptedDetector(BaseDetector):
    """Returns pre-scripted detections frame by frame (ground truth with optional drop-outs)."""

    name = "scripted"

    def __init__(self, per_frame: list[Detections]):
        self.per_frame = per_frame
        self.i = 0

    def detect(self, image):
        d = self.per_frame[self.i] if self.i < len(self.per_frame) else Detections()
        self.i += 1
        return d


class FailingDetector(BaseDetector):
    name = "failing"

    def __init__(self, fail_every: int = 3):
        self.n = 0
        self.fail_every = fail_every

    def detect(self, image):
        self.n += 1
        if self.n % self.fail_every == 0:
            raise RuntimeError("simulated detector crash")
        h, w = image.shape[:2]
        return Detections([[w * 0.4, h * 0.2, w * 0.6, h * 0.9]], [0.9], [0])


def gt_to_detections(gt_rows) -> Detections:
    vis = [r for r in gt_rows if r[5]]
    return Detections([r[1:5] for r in vis], [0.9] * len(vis), [0] * len(vis))


@pytest.fixture
def synthetic():
    frames, gt = make_synthetic_sequence(n_frames=40)
    return frames, gt


@pytest.fixture
def synthetic_video(tmp_path, synthetic):
    frames, gt = synthetic
    return write_frames(frames, tmp_path / "synthetic.mp4", fps=10), gt


@pytest.fixture
def base_cfg():
    from src.utils.config import load_config

    return load_config(overrides={"output": {"h264": False}, "device": "cpu"})


@pytest.fixture(scope="session")
def real_detector():
    if not WEIGHTS.exists():
        pytest.skip("models/yolo26n.pt not present; real-model tests skipped")
    from src.detection.detector import UltralyticsPersonDetector

    return UltralyticsPersonDetector(str(WEIGHTS), "cpu", allow_download=False)


@pytest.fixture
def pan_clip(tmp_path):
    from src.utils.sample_video import make_pan_clip

    return make_pan_clip(tmp_path / "pan.mp4", frames=20, fps=10)
