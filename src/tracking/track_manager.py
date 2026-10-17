"""Collects tracker output over time: trajectories, enter/exit events, statistics, quality warnings, exports."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.tracking.tracker import TrackedObject
from src.tracking.trajectory import TrackPoint, Trajectory

CSV_FIELDS = ["frame", "timestamp_s", "track_id", "x1", "y1", "x2", "y2", "confidence", "reid_label", "reid_similarity"]

COUNT_DISCLAIMER = (
    "Unique track IDs are not the number of unique real-world people: ID switches, fragmentation and "
    "re-entry can inflate or deflate the count."
)
REID_DISCLAIMER = (
    "ReID labels indicate visual (body appearance) similarity only. They are not verified identities and "
    "must not be used to attribute identity to a named individual."
)


class TrackManager:
    def __init__(self, frame_stride: int = 1, quality_cfg: dict | None = None):
        self.frame_stride = max(1, frame_stride)
        self.quality = {"low_confidence": 0.4, "short_track_frames": 5, "short_track_ratio": 0.5, **(quality_cfg or {})}
        self.trajectories: dict[int, Trajectory] = {}
        self.rows: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.active_counts: list[int] = []
        self.detection_counts: list[int] = []
        self._active_prev: set[int] = set()
        self._last_frame: int | None = None
        self._last_ts: float = 0.0

    def update(self, frame: int, timestamp_s: float, tracks: list[TrackedObject], num_detections: int,
               reid_labels: dict[int, tuple[str, float | None]] | None = None) -> None:
        active = set()
        for tr in tracks:
            tid = tr.track_id
            active.add(tid)
            traj = self.trajectories.get(tid)
            if traj is None:
                traj = self.trajectories[tid] = Trajectory(tid)
                self.events.append({"type": "track_start", "track_id": tid, "frame": frame, "timestamp_s": round(timestamp_s, 3)})
            if reid_labels and tid in reid_labels:
                traj.reid_label, traj.reid_similarity = reid_labels[tid]
            xyxy = tuple(float(v) for v in tr.xyxy)
            traj.add(TrackPoint(frame, timestamp_s, xyxy, tr.score))
            self.rows.append({
                "frame": frame, "timestamp_s": round(timestamp_s, 3), "track_id": tid,
                "x1": round(xyxy[0], 1), "y1": round(xyxy[1], 1), "x2": round(xyxy[2], 1), "y2": round(xyxy[3], 1),
                "confidence": round(tr.score, 4), "reid_label": traj.reid_label,
                "reid_similarity": "" if traj.reid_similarity is None else round(traj.reid_similarity, 4),
            })
        for tid in self._active_prev - active:
            self.events.append({"type": "track_lost", "track_id": tid, "frame": frame, "timestamp_s": round(timestamp_s, 3)})
        for tid in active - self._active_prev:
            if self.trajectories[tid].num_observations > 1:
                self.events.append({"type": "track_recovered", "track_id": tid, "frame": frame, "timestamp_s": round(timestamp_s, 3)})
        self._active_prev = active
        self.active_counts.append(len(active))
        self.detection_counts.append(num_detections)
        self._last_frame, self._last_ts = frame, timestamp_s

    def trails(self, active_ids: list[int], length: int) -> dict[int, list[tuple[float, float]]]:
        return {tid: self.trajectories[tid].recent_centers(length) for tid in active_ids if tid in self.trajectories}

    # ---------------------------------------------------------------- analysis
    def summary(self) -> dict:
        trajs = list(self.trajectories.values())
        n = len(trajs)
        short_thr = self.quality["short_track_frames"]
        lengths = [t.num_observations for t in trajs]
        short = sum(1 for l in lengths if l < short_thr)
        mean_conf = float(np.mean([r["confidence"] for r in self.rows])) if self.rows else 0.0
        reid_labels = Counter(t.reid_label for t in trajs if t.reid_label != "unknown")
        return {
            "processed_frames": len(self.active_counts),
            "unique_track_ids": n,
            "unique_track_ids_note": COUNT_DISCLAIMER,
            "max_concurrent_tracks": max(self.active_counts) if self.active_counts else 0,
            "mean_concurrent_tracks": round(float(np.mean(self.active_counts)), 3) if self.active_counts else 0.0,
            "total_detections": int(sum(self.detection_counts)),
            "frames_without_detections": int(sum(1 for c in self.detection_counts if c == 0)),
            "mean_track_confidence": round(mean_conf, 4),
            "mean_track_length_frames": round(float(np.mean(lengths)), 2) if lengths else 0.0,
            "median_track_duration_s": round(float(np.median([t.duration_s for t in trajs])), 3) if trajs else 0.0,
            "short_tracks": short,
            "short_track_ratio": round(short / n, 3) if n else 0.0,
            "track_gaps_total": int(sum(t.gaps(self.frame_stride) for t in trajs)),
            "recoveries": sum(1 for e in self.events if e["type"] == "track_recovered"),
            "visual_identities": len(reid_labels),
        }

    def warnings(self) -> list[str]:
        s = self.summary()
        q = self.quality
        out = []
        if s["processed_frames"] and s["total_detections"] == 0:
            out.append("No people were detected. Check lighting, resolution, or lower detector.conf.")
        if s["unique_track_ids"] and s["mean_track_confidence"] < q["low_confidence"]:
            out.append(f"Mean track confidence is low ({s['mean_track_confidence']:.2f}); results may be unreliable.")
        if s["unique_track_ids"] >= 4 and s["short_track_ratio"] > q["short_track_ratio"]:
            out.append(
                f"{s['short_track_ratio']:.0%} of tracks are shorter than {q['short_track_frames']} frames: tracking is "
                "fragmenting (occlusion, low frame rate, or detector misses). Consider BoT-SORT, a larger "
                "track_buffer, or a lower frame_stride."
            )
        if s["processed_frames"] > 10 and s["frames_without_detections"] / s["processed_frames"] > 0.5 and s["total_detections"]:
            out.append("More than half of the processed frames have no detections.")
        return out

    # ----------------------------------------------------------------- export
    def export_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        return path

    def to_dict(self, metadata: dict | None = None, config: dict | None = None, timing: dict | None = None) -> dict:
        return {
            "schema_version": 1,
            "video": metadata or {},
            "config": config or {},
            "timing": timing or {},
            "summary": self.summary(),
            "warnings": self.warnings(),
            "disclaimers": [COUNT_DISCLAIMER, REID_DISCLAIMER],
            "events": self.events,
            "tracks": [t.to_dict(self.frame_stride) for t in sorted(self.trajectories.values(), key=lambda t: t.track_id)],
        }

    def export_json(self, path: str | Path, **kwargs) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(**kwargs), indent=2, default=_json_default))
        return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")
