"""Controlled tracker comparison on a synthetic crowd with exact ground truth.

People are rendered as textured, differently coloured figures walking on crossing paths, with occlusion
by a static pillar. A *simulated detector* derives detections from ground truth with configurable misses,
box jitter, confidence noise and false positives. This isolates association quality (ID switches, IDF1,
HOTA) from detector quality. Results are synthetic and are reported as such; use MOT17/MOT20 for real
benchmarks (``cli.py evaluate tracking``).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.detection.detector import Detections
from src.evaluation.tracking_metrics import evaluate_tracking
from src.tracking.tracker import PersonTracker


def make_crowd(n_people: int = 8, n_frames: int = 150, size: tuple[int, int] = (960, 540), seed: int = 0,
               pillar: bool = True):
    rng = np.random.default_rng(seed)
    w, h = size
    people = []
    for i in range(n_people):
        left = i % 2 == 0
        y = rng.uniform(0.25, 0.75) * h
        ph = rng.uniform(110, 170)
        people.append({
            "id": i + 1, "x": -40.0 - rng.uniform(0, 300) if left else w + rng.uniform(0, 300),
            "y": y, "vx": rng.uniform(3, 7) * (1 if left else -1), "vy": rng.uniform(-0.6, 0.6),
            "h": ph, "w": ph * 0.38,
            "top": tuple(int(c) for c in rng.integers(30, 255, 3)), "bottom": tuple(int(c) for c in rng.integers(30, 255, 3)),
        })
    pillar_box = (w * 0.45, 0, w * 0.52, h) if pillar else None
    frames, gt = [], {}
    bg = np.full((h, w, 3), 90, np.uint8)
    cv2.randn(bg, (90, 90, 90), (12, 12, 12))
    for f in range(n_frames):
        img = bg.copy()
        rows = []
        for p in sorted(people, key=lambda p: p["y"]):  # painter's order: lower on screen = closer
            x1 = p["x"] + p["vx"] * f
            y1 = p["y"] + p["vy"] * f - p["h"] / 2
            x2, y2 = x1 + p["w"], y1 + p["h"]
            if x2 < 0 or x1 > w:
                continue
            mid = int(y1 + p["h"] * 0.45)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), mid), p["top"], -1)
            cv2.rectangle(img, (int(x1), mid), (int(x2), int(y2)), p["bottom"], -1)
            cv2.circle(img, (int((x1 + x2) / 2), int(y1 + p["h"] * 0.08)), int(p["w"] * 0.3), (180, 160, 150), -1)
            vis = 1.0
            if pillar_box:
                ov = max(0.0, min(x2, pillar_box[2]) - max(x1, pillar_box[0])) / max(p["w"], 1)
                vis = 1.0 - ov
            rows.append((p["id"], max(0.0, x1), y1, min(float(w), x2), y2, vis))
        if pillar_box:
            cv2.rectangle(img, (int(pillar_box[0]), 0), (int(pillar_box[2]), h), (60, 60, 70), -1)
        frames.append(img)
        gt[f + 1] = rows
    return frames, gt


def simulate_detections(gt_rows, rng: np.random.Generator, miss: float = 0.1, jitter: float = 0.04,
                        fp_rate: float = 0.1, size=(960, 540)) -> Detections:
    boxes, confs = [], []
    for _, x1, y1, x2, y2, vis in gt_rows:
        if vis < 0.4 or rng.random() < miss * (1.5 - vis):
            continue
        bw, bh = x2 - x1, y2 - y1
        n = rng.normal(0, jitter, 4) * [bw, bh, bw, bh]
        boxes.append([x1 + n[0], y1 + n[1], x2 + n[2], y2 + n[3]])
        confs.append(float(np.clip(rng.normal(0.75, 0.15) * (0.5 + 0.5 * vis), 0.05, 0.99)))
    if rng.random() < fp_rate:
        x, y = rng.uniform(0, size[0] - 60), rng.uniform(0, size[1] - 150)
        boxes.append([x, y, x + 50, y + 130])
        confs.append(float(rng.uniform(0.2, 0.5)))
    return Detections(np.asarray(boxes).reshape(-1, 4), confs, np.zeros(len(confs))).sanitize(*size)


def run_tracker_benchmark(configs: dict[str, dict], seeds: tuple[int, ...] = (0, 1, 2), n_people: int = 8,
                          n_frames: int = 150, embedder=None, **det_kwargs) -> dict:
    """``configs``: {label: tracker_cfg}. Returns mean metrics over seeds for each label."""
    results: dict[str, list[dict]] = {k: [] for k in configs}
    for seed in seeds:
        frames, gt = make_crowd(n_people, n_frames, seed=seed)
        gt_eval = {f: [r[:5] for r in rows if r[5] > 0.0] for f, rows in gt.items()}
        dets_per_frame = [simulate_detections(gt[f + 1], np.random.default_rng(1000 + seed * 7919 + f), **det_kwargs)
                          for f in range(n_frames)]
        for label, tcfg in configs.items():
            tracker = PersonTracker(tcfg)
            pred = {}
            t0 = time.perf_counter()
            for f, (img, dets) in enumerate(zip(frames, dets_per_frame)):
                feats = embedder.embed_boxes(img, dets.xyxy) if (tracker.uses_appearance and embedder and len(dets)) else None
                pred[f + 1] = [(t.track_id, *t.xyxy.tolist()) for t in tracker.update(dets, img, feats)]
            elapsed = time.perf_counter() - t0
            m = evaluate_tracking(gt_eval, pred)
            m["tracker_fps"] = n_frames / elapsed
            results[label].append(m)
    keys = ["HOTA", "DetA", "AssA", "idf1", "mota", "num_switches", "num_fragmentations", "track_persistence", "tracker_fps"]
    return {label: {k: float(np.mean([r[k] for r in rs])) for k in keys} for label, rs in results.items()}
