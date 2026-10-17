"""Multi-object tracking wrapper around the Ultralytics tracker implementations.

The trackers are used standalone (not via ``model.track``) so detection, tracking and ReID stay independent:
any ``BaseDetector`` can feed any tracker, and appearance embeddings from ``src.recognition`` can be passed
into BoT-SORT's association.

Default: ByteTrack. It associates low-confidence boxes in a second stage, which recovers partially occluded
people, and it is fast on CPU. BoT-SORT adds camera-motion compensation and optional appearance features,
which helps with moving cameras and crossing paths at extra cost.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.detection.detector import Detections

log = logging.getLogger(__name__)

TRACKER_KEYS = (
    "track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh", "fuse_score",
    "gmc_method", "proximity_thresh", "appearance_thresh", "with_reid",
)


@dataclass
class TrackedObject:
    track_id: int
    xyxy: np.ndarray
    score: float
    cls: int
    det_index: int  # index into the frame's Detections (-1 if unknown)


def available_trackers() -> list[str]:
    from ultralytics.trackers.track import TRACKER_MAP

    return sorted(TRACKER_MAP)


class PersonTracker:
    def __init__(self, cfg: dict, frame_stride: int = 1):
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import IterableSimpleNamespace, YAML

        tcfg = dict(cfg)
        self.type = tcfg.get("type", "bytetrack").lower()
        if self.type not in TRACKER_MAP:
            raise ValueError(f"Unknown tracker '{self.type}'. Available: {sorted(TRACKER_MAP)}")
        import ultralytics

        yaml_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{self.type}.yaml"
        args = dict(YAML.load(yaml_path)) if yaml_path.exists() else {}
        for key in TRACKER_KEYS:
            if key in tcfg and (key in args or key in ("with_reid",)):
                args[key] = tcfg[key]
        # track_buffer is configured in source frames; when skipping frames the tracker sees fewer updates,
        # so convert to processed frames to keep the same wall-clock occlusion tolerance.
        self.frame_stride = max(1, int(frame_stride))
        args["track_buffer"] = max(1, math.ceil(int(tcfg.get("track_buffer", args.get("track_buffer", 30))) / self.frame_stride))
        # Appearance features (if any) are supplied by our own embedder via `feats`, so use the pass-through mode.
        args["model"] = "auto"
        args.setdefault("with_reid", False)
        args["tracker_type"] = self.type
        self.args = IterableSimpleNamespace(**args)
        self.uses_appearance = bool(self.args.with_reid) and self.type in ("botsort", "deepocsort", "tracktrack")
        self._impl = TRACKER_MAP[self.type](args=self.args)

    def update(self, dets: Detections, image: np.ndarray | None = None, feats: np.ndarray | None = None) -> list[TrackedObject]:
        out = self._impl.update(_UltralyticsDets(dets), image, feats=feats if self.uses_appearance else None)
        if out is None or len(out) == 0:
            return []
        out = np.asarray(out)
        return [
            TrackedObject(
                track_id=int(row[4]),
                xyxy=row[:4].astype(np.float32),
                score=float(row[5]),
                cls=int(row[6]),
                det_index=int(row[7]) if row.shape[0] > 7 else -1,
            )
            for row in out
        ]

    def reset(self) -> None:
        self._impl.reset()


class _UltralyticsDets:
    """Duck-typed stand-in for ``ultralytics.engine.results.Boxes`` (numpy, boolean-indexable)."""

    def __init__(self, dets: Detections):
        self._d = dets
        self.conf = dets.conf
        self.cls = dets.cls
        self.xyxy = dets.xyxy
        self.xywh = dets.xywh
        self.data = np.concatenate([dets.xyxy, dets.conf[:, None], dets.cls[:, None]], axis=1)

    def __len__(self) -> int:
        return len(self._d)

    def __getitem__(self, mask) -> "_UltralyticsDets":
        return _UltralyticsDets(self._d[mask])
