"""Body-appearance embedding extractors for optional person ReID. No face features are used.

Backends (choose with ``reid.backend``):

* ``histogram`` - HSV colour histograms over head/torso/legs stripes. No weights, always available, fast.
  Captures clothing colour only; weak under illumination change. Useful as a baseline.
* ``yolo_cls``  - global-pooled features from an Ultralytics ImageNet classifier (e.g. ``yolo26n-cls.pt``).
  Generic visual features, NOT trained for person ReID.
* ``torchreid`` - a real person-ReID network (e.g. OSNet trained on Market-1501/MSMT17) via the
  ``torchreid`` package and a user-supplied checkpoint. This is the recommended backend for ReID quality.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


def crop_person(image: np.ndarray, xyxy, pad: float = 0.05) -> np.ndarray | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
    y1, y2 = int(max(0, y1 - pad * bh)), int(min(h, y2 + pad * bh))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return image[y1:y2, x1:x2]


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


class EmbeddingExtractor(ABC):
    name = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return an (N, D) array of L2-normalised embeddings for BGR crops."""

    def embed_boxes(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        crops = [crop_person(image, b) for b in boxes]
        valid = [i for i, c in enumerate(crops) if c is not None]
        out = np.zeros((len(boxes), self.dim), np.float32)
        if valid:
            out[valid] = self.embed([crops[i] for i in valid])
        return out


class HistogramEmbedder(EmbeddingExtractor):
    name = "histogram"

    def __init__(self, bins: tuple[int, int, int] = (16, 4, 4), stripes: int = 3):
        self.bins, self.stripes = bins, stripes
        self.dim = int(np.prod(bins)) * stripes

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        feats = []
        for crop in crops:
            hsv = cv2.cvtColor(cv2.resize(crop, (64, 128)), cv2.COLOR_BGR2HSV)
            parts = []
            for s in np.array_split(hsv, self.stripes, axis=0):
                hist = cv2.calcHist([s], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256]).ravel()
                parts.append(np.sqrt(hist / max(hist.sum(), 1.0)))  # Hellinger-style
            feats.append(np.concatenate(parts))
        return l2_normalize(np.asarray(feats, np.float32))


class YoloClsEmbedder(EmbeddingExtractor):
    name = "yolo_cls"

    def __init__(self, weights: str = "yolo26n-cls.pt", device: str = "cpu"):
        from src.detection.detector import resolve_weights

        try:
            from ultralytics import YOLO

            self.model = YOLO(resolve_weights(weights))
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Could not load classifier '{weights}': {exc}") from exc
        self.device = device
        self.dim = int(self.embed([np.zeros((128, 64, 3), np.uint8)]).shape[1])

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        feats = self.model.embed(list(crops), imgsz=224, device=self.device, verbose=False)
        return l2_normalize(np.stack([f.detach().cpu().numpy().ravel() for f in feats]).astype(np.float32))


class TorchreidEmbedder(EmbeddingExtractor):
    name = "torchreid"

    def __init__(self, model_name: str = "osnet_x0_25", weights: str | None = None, device: str = "cpu"):
        try:
            from torchreid.utils import FeatureExtractor  # type: ignore
        except ImportError as exc:
            raise EmbeddingError(
                "The 'torchreid' backend needs the torchreid package: "
                "pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
            ) from exc
        if not weights:
            raise EmbeddingError("reid.weights must point to a torchreid checkpoint (e.g. osnet_x0_25_msmt17.pth).")
        self.extractor = FeatureExtractor(model_name=model_name, model_path=weights, device=device)
        self.dim = int(self.embed([np.zeros((256, 128, 3), np.uint8)]).shape[1])

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
        return l2_normalize(self.extractor(rgb).cpu().numpy().astype(np.float32))


def build_embedder(cfg: dict, device: str) -> EmbeddingExtractor:
    r = cfg.get("reid", {})
    backend = r.get("backend", "histogram")
    if backend == "histogram":
        return HistogramEmbedder()
    if backend == "yolo_cls":
        return YoloClsEmbedder(r.get("weights") or "yolo26n-cls.pt", device)
    if backend == "torchreid":
        return TorchreidEmbedder(r.get("torchreid_model", "osnet_x0_25"), r.get("weights"), device)
    raise EmbeddingError(f"Unknown reid.backend '{backend}'")
