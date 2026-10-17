"""Small generated clips for smoke tests and demos (no dataset download required).

* ``make_pan_clip`` - a slow pan/zoom over a real photo containing people (the images bundled with the
  Ultralytics package), so a real detector produces real detections and the tracker must follow motion.
* ``make_synthetic_sequence`` - moving rectangles with exact ground truth (for tracker tests and metrics).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def bundled_image(name: str = "bus.jpg") -> Path:
    import ultralytics

    p = Path(ultralytics.__file__).parent / "assets" / name
    if not p.exists():
        raise FileNotFoundError(f"Bundled image {name} not found in the Ultralytics package")
    return p


def make_pan_clip(out_path: str | Path, image: str | Path | None = None, frames: int = 60, fps: float = 15.0,
                  size: tuple[int, int] = (640, 480), darken: float = 1.0, fourcc: str = "mp4v") -> Path:
    img = cv2.imread(str(image or bundled_image()))
    if img is None:
        raise FileNotFoundError(f"Could not read {image}")
    h, w = img.shape[:2]
    out_w, out_h = size
    crop_w0 = min(w, int(h * out_w / out_h))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, (out_w, out_h))
    for i in range(frames):
        a = i / max(1, frames - 1)
        zoom = 1.0 - 0.15 * a
        cw, ch = int(crop_w0 * zoom), int(crop_w0 * zoom * out_h / out_w)
        ch = min(ch, h)
        x0 = int((w - cw) * (0.2 + 0.6 * a))
        y0 = int((h - ch) * 0.5)
        frame = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (out_w, out_h), interpolation=cv2.INTER_AREA)
        if darken != 1.0:
            frame = np.clip(frame.astype(np.float32) * darken, 0, 255).astype(np.uint8)
        vw.write(frame)
    vw.release()
    return out_path


@dataclass
class SyntheticObject:
    obj_id: int
    start: tuple[float, float]
    velocity: tuple[float, float]
    size: tuple[int, int]
    color: tuple[int, int, int]
    hidden: tuple[int, int] | None = None  # [start, end) frames during which the object is occluded


def default_objects() -> list[SyntheticObject]:
    return [
        SyntheticObject(1, (40, 120), (6, 0.5), (50, 120), (0, 0, 255)),
        SyntheticObject(2, (560, 140), (-6, 0.0), (50, 120), (0, 255, 0), hidden=(18, 22)),
        SyntheticObject(3, (300, 300), (1, -2), (40, 100), (255, 0, 0)),
    ]


def make_synthetic_sequence(n_frames: int = 40, size: tuple[int, int] = (640, 480),
                            objects: list[SyntheticObject] | None = None):
    """Return (frames, ground_truth) where ground_truth[f] = list of (obj_id, x1, y1, x2, y2) for visible objects."""
    objects = objects or default_objects()
    w, h = size
    frames, gt = [], []
    for f in range(n_frames):
        img = np.full((h, w, 3), 40, np.uint8)
        rows = []
        for o in objects:
            x = o.start[0] + o.velocity[0] * f
            y = o.start[1] + o.velocity[1] * f
            x1, y1, x2, y2 = x, y, x + o.size[0], y + o.size[1]
            if x2 < 0 or x1 > w or y2 < 0 or y1 > h:
                continue
            if o.hidden and o.hidden[0] <= f < o.hidden[1]:
                rows.append((o.obj_id, x1, y1, x2, y2, False))
                continue
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), o.color, -1)
            rows.append((o.obj_id, x1, y1, x2, y2, True))
        frames.append(img)
        gt.append(rows)
    return frames, gt


def write_frames(frames: list[np.ndarray], out_path: str | Path, fps: float = 10.0) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return out_path
