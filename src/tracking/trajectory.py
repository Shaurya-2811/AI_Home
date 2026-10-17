"""Per-track history: boxes, timestamps, confidences and derived trajectory statistics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackPoint:
    frame: int
    timestamp_s: float
    xyxy: tuple[float, float, float, float]
    score: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def foot(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.xyxy
        return (x1 + x2) / 2, y2


@dataclass
class Trajectory:
    track_id: int
    points: list[TrackPoint] = field(default_factory=list)
    reid_label: str = "unknown"
    reid_similarity: float | None = None

    def add(self, point: TrackPoint) -> None:
        self.points.append(point)

    @property
    def first(self) -> TrackPoint:
        return self.points[0]

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]

    @property
    def num_observations(self) -> int:
        return len(self.points)

    @property
    def duration_s(self) -> float:
        return self.last.timestamp_s - self.first.timestamp_s if self.points else 0.0

    @property
    def mean_score(self) -> float:
        return float(np.mean([p.score for p in self.points])) if self.points else 0.0

    @property
    def path_length_px(self) -> float:
        if len(self.points) < 2:
            return 0.0
        c = np.asarray([p.center for p in self.points])
        return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())

    def gaps(self, frame_stride: int = 1) -> int:
        """Number of interruptions (missing processed frames) inside the track's lifetime."""
        if len(self.points) < 2:
            return 0
        frames = np.asarray([p.frame for p in self.points])
        return int((np.diff(frames) > frame_stride).sum())

    def recent_centers(self, n: int) -> list[tuple[float, float]]:
        return [p.foot for p in self.points[-n:]]

    def summary(self, frame_stride: int = 1) -> dict:
        return {
            "track_id": self.track_id,
            "first_frame": self.first.frame,
            "last_frame": self.last.frame,
            "first_seen_s": round(self.first.timestamp_s, 3),
            "last_seen_s": round(self.last.timestamp_s, 3),
            "duration_s": round(self.duration_s, 3),
            "observations": self.num_observations,
            "mean_confidence": round(self.mean_score, 4),
            "path_length_px": round(self.path_length_px, 1),
            "gaps": self.gaps(frame_stride),
            "reid_label": self.reid_label,
            "reid_similarity": None if self.reid_similarity is None else round(self.reid_similarity, 4),
        }

    def to_dict(self, frame_stride: int = 1) -> dict:
        d = self.summary(frame_stride)
        d["trajectory"] = [
            {"frame": p.frame, "t": round(p.timestamp_s, 3), "bbox": [round(v, 1) for v in p.xyxy], "conf": round(p.score, 4)}
            for p in self.points
        ]
        return d
