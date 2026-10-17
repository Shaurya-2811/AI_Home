"""Person detection. The pipeline only depends on ``BaseDetector``; the YOLO backend is one implementation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils.config import PROJECT_ROOT
from src.utils.device import use_half

log = logging.getLogger(__name__)


class DetectorError(RuntimeError):
    pass


@dataclass
class Detections:
    """Person detections for one frame. ``xyxy`` is in pixel coordinates of the frame that was detected on."""

    xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    conf: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    cls: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(self.conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(self.cls, dtype=np.float32).reshape(-1)
        if not (len(self.xyxy) == len(self.conf) == len(self.cls)):
            raise ValueError("xyxy, conf and cls must have the same length")

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "Detections":
        return Detections(self.xyxy[mask], self.conf[mask], self.cls[mask])

    @property
    def xywh(self) -> np.ndarray:
        """Center-x, center-y, width, height (the format Ultralytics trackers consume)."""
        out = np.empty_like(self.xyxy)
        out[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2
        out[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2
        out[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        out[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return out

    @property
    def area(self) -> np.ndarray:
        return np.clip(self.xyxy[:, 2] - self.xyxy[:, 0], 0, None) * np.clip(self.xyxy[:, 3] - self.xyxy[:, 1], 0, None)

    def sanitize(self, width: int, height: int, min_area: float = 0.0) -> "Detections":
        """Clip to the image, drop non-finite / degenerate boxes, clamp confidence to [0, 1]."""
        if len(self) == 0:
            return self
        xyxy = self.xyxy.copy()
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
        conf = np.clip(self.conf, 0.0, 1.0)
        d = Detections(xyxy, conf, self.cls)
        keep = np.isfinite(xyxy).all(1) & np.isfinite(conf) & (d.area > max(min_area, 0.0))
        keep &= (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        return d[keep]

    def scaled(self, factor: float) -> "Detections":
        return Detections(self.xyxy * factor, self.conf, self.cls) if factor != 1.0 else self


class BaseDetector(ABC):
    name = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> Detections: ...

    def detect_batch(self, images: list[np.ndarray]) -> list[Detections]:
        return [self.detect(im) for im in images]

    def info(self) -> dict:
        return {"name": self.name}


class UltralyticsPersonDetector(BaseDetector):
    """Pretrained Ultralytics YOLO restricted to the COCO ``person`` class (or the class named 'person')."""

    name = "ultralytics"

    def __init__(
        self,
        weights: str = "yolo26n.pt",
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        half: str | bool = "auto",
        max_det: int = 300,
        min_box_area: float = 0.0,
        allow_download: bool = True,
    ):
        self.weights = resolve_weights(weights, allow_download)
        self.device = device
        self.conf, self.iou, self.imgsz, self.max_det = conf, iou, imgsz, max_det
        self.half = use_half(device, half)
        self.min_box_area = min_box_area
        try:
            from ultralytics import YOLO

            from src.utils.privacy import disable_third_party_telemetry

            disable_third_party_telemetry()
            self.model = YOLO(self.weights)
        except Exception as exc:  # noqa: BLE001 - surface a clear message for any load failure
            raise DetectorError(f"Could not load detector weights '{self.weights}': {exc}") from exc
        names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        person = [k for k, v in names.items() if str(v).lower() in {"person", "pedestrian", "people"}]
        if not person and len(names) == 1:
            log.warning("Single-class model '%s' (class '%s'): assuming it detects people.", self.weights, names[next(iter(names))])
            person = [next(iter(names))]
        if not person:
            raise DetectorError(f"Model '{self.weights}' has no 'person' class (classes: {list(names.values())[:10]}...)")
        self.person_class = int(person[0])

    def _predict(self, source):
        return self.model.predict(
            source,
            classes=[self.person_class],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            quantize=16 if self.half else None,  # FP16 on CUDA only ('half' is deprecated in Ultralytics 8.4)
            max_det=self.max_det,
            verbose=False,
        )

    def _to_dets(self, result, shape) -> Detections:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return Detections()
        b = boxes.cpu().numpy()
        dets = Detections(b.xyxy, b.conf, np.zeros(len(b.conf), np.float32))
        return dets.sanitize(shape[1], shape[0], self.min_box_area)

    def detect(self, image: np.ndarray) -> Detections:
        return self._to_dets(self._predict(image)[0], image.shape)

    def detect_batch(self, images: list[np.ndarray]) -> list[Detections]:
        if not images:
            return []
        return [self._to_dets(r, im.shape) for r, im in zip(self._predict(list(images)), images)]

    def info(self) -> dict:
        return {
            "name": self.name,
            "weights": str(self.weights),
            "device": self.device,
            "half": self.half,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "person_class": self.person_class,
        }


def resolve_weights(weights: str, allow_download: bool = True) -> str:
    """Find weights locally (as given, or under models/), otherwise let Ultralytics download a known asset name."""
    p = Path(weights)
    candidates = [p, PROJECT_ROOT / p, PROJECT_ROOT / "models" / p.name]
    for c in candidates:
        if c.is_file():
            return str(c)
    if allow_download and p.suffix == ".pt" and len(p.parts) <= 2:
        # Ultralytics downloads official asset names (e.g. yolo26n.pt) on first use into the CWD;
        # direct it into models/ so the file is reused and ignored by git.
        target = PROJECT_ROOT / "models" / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            from ultralytics.utils.downloads import attempt_download_asset

            return str(attempt_download_asset(str(target)))
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(
                f"Weights '{weights}' not found locally and download failed ({exc}). "
                f"Place the file in {PROJECT_ROOT / 'models'} or set detector.weights to its path."
            ) from exc
    raise DetectorError(
        f"Weights '{weights}' not found. Looked in: {', '.join(str(c) for c in candidates)}. "
        "Set detector.allow_download=true to fetch official Ultralytics weights."
    )


def build_detector(cfg: dict, device: str) -> BaseDetector:
    d = cfg["detector"]
    backend = d.get("backend", "ultralytics")
    if backend != "ultralytics":
        raise DetectorError(f"Unknown detector backend '{backend}'")
    return UltralyticsPersonDetector(
        weights=d["weights"],
        device=device,
        conf=d["conf"],
        iou=d["iou"],
        imgsz=d["imgsz"],
        half=d.get("half", "auto"),
        max_det=d.get("max_det", 300),
        min_box_area=d.get("min_box_area", 0),
        allow_download=d.get("allow_download", True),
    )
