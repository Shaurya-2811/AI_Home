"""Tracking metrics.

* CLEAR-MOT and identity metrics (MOTA, MOTP, IDF1, ID switches, fragmentation, MT/ML) come from the
  established ``motmetrics`` package (py-motmetrics).
* HOTA (DetA, AssA, LocA) is implemented here following the reference algorithm of TrackEval
  (Luiten et al., IJCV 2020), averaged over IoU thresholds 0.05..0.95, so there is no hard dependency on
  TrackEval. ``tests/test_metrics.py`` checks known values (perfect tracking = 1, an identity swap lowers
  AssA but not DetA) and, when the optional ``trackeval`` package is installed, that HOTA/DetA/AssA/LocA
  match TrackEval's own HOTA implementation.
* Track persistence: for each GT identity, the fraction of its frames covered by its single most
  frequently matched predicted ID (1.0 = never lost or switched).

Input format (both GT and predictions): {frame: [(id, x1, y1, x2, y2), ...]}.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.evaluation.detection_metrics import box_iou

Sequence = dict[int, list[tuple]]


def _arrays(rows: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.asarray(rows, np.float64)
    return a[:, 0].astype(int), a[:, 1:5]


def clear_mot(gt: Sequence, pred: Sequence, iou_threshold: float = 0.5) -> dict:
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    for f in sorted(set(gt) | set(pred)):
        gid, gbox = _arrays(gt.get(f, []))
        pid, pbox = _arrays(pred.get(f, []))
        dist = 1.0 - box_iou(gbox, pbox)
        dist[dist > 1.0 - iou_threshold] = np.nan
        acc.update(gid.tolist(), pid.tolist(), dist, frameid=int(f))
    mh = mm.metrics.create()
    names = ["mota", "motp", "idf1", "idp", "idr", "num_switches", "num_fragmentations", "mostly_tracked",
             "partially_tracked", "mostly_lost", "num_false_positives", "num_misses", "precision", "recall",
             "num_unique_objects"]
    s = mh.compute(acc, metrics=names, name="seq")
    out = {k: float(s[k].iloc[0]) for k in names}
    out["motp_iou"] = 1.0 - out.pop("motp") if np.isfinite(out["motp"]) else float("nan")
    return out


def hota(gt: Sequence, pred: Sequence) -> dict:
    alphas = np.arange(0.05, 0.99, 0.05)
    gt_ids = sorted({r[0] for rows in gt.values() for r in rows})
    pr_ids = sorted({r[0] for rows in pred.values() for r in rows})
    gmap = {g: i for i, g in enumerate(gt_ids)}
    pmap = {p: i for i, p in enumerate(pr_ids)}
    G, P = len(gt_ids), len(pr_ids)
    frames = sorted(set(gt) | set(pred))
    if G == 0 or P == 0:
        n_gt = sum(len(v) for v in gt.values())
        n_pr = sum(len(v) for v in pred.values())
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0, "LocA": 1.0 if n_gt == n_pr == 0 else 0.0,
                "DetRe": 0.0, "DetPr": 0.0}

    per_frame = []
    pot = np.zeros((G, P))
    gcount = np.zeros((G, 1))
    pcount = np.zeros((1, P))
    for f in frames:
        gi, gb = _arrays(gt.get(f, []))
        pi, pb = _arrays(pred.get(f, []))
        gi = np.array([gmap[x] for x in gi], int)
        pi = np.array([pmap[x] for x in pi], int)
        sim = box_iou(gb, pb)
        per_frame.append((gi, pi, sim))
        if len(gi) and len(pi):
            denom = sim.sum(0, keepdims=True) + sim.sum(1, keepdims=True) - sim
            sim_iou = np.where(denom > 1e-10, sim / np.maximum(denom, 1e-10), 0.0)
            pot[gi[:, None], pi[None, :]] += sim_iou
        gcount[gi] += 1
        pcount[0, pi] += 1
    global_align = pot / np.maximum(gcount + pcount - pot, 1e-10)

    A = len(alphas)
    tp, fn, fp, loc = np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A)
    matches = [np.zeros((G, P)) for _ in range(A)]
    for gi, pi, sim in per_frame:
        if len(gi) == 0:
            fp += len(pi)
            continue
        if len(pi) == 0:
            fn += len(gi)
            continue
        score = global_align[gi[:, None], pi[None, :]] * sim
        r, c = linear_sum_assignment(-score)
        for a, alpha in enumerate(alphas):
            ok = sim[r, c] >= alpha - 1e-10
            rr, cc = r[ok], c[ok]
            n = len(rr)
            tp[a] += n
            fn[a] += len(gi) - n
            fp[a] += len(pi) - n
            if n:
                loc[a] += sim[rr, cc].sum()
                matches[a][gi[rr], pi[cc]] += 1
    ass = np.zeros(A)
    for a in range(A):
        m = matches[a]
        ass_iou = m / np.maximum(gcount + pcount - m, 1e-10)
        ass[a] = (m * ass_iou).sum() / max(tp[a], 1e-10)
    det = tp / np.maximum(tp + fn + fp, 1e-10)
    det_re = tp / np.maximum(tp + fn, 1e-10)
    det_pr = tp / np.maximum(tp + fp, 1e-10)
    loc_a = np.maximum(loc, 1e-10) / np.maximum(tp, 1e-10)
    h = np.sqrt(det * ass)
    return {"HOTA": float(h.mean()), "DetA": float(det.mean()), "AssA": float(ass.mean()),
            "LocA": float(loc_a.mean()), "DetRe": float(det_re.mean()), "DetPr": float(det_pr.mean())}


def track_persistence(gt: Sequence, pred: Sequence, iou_threshold: float = 0.5) -> dict:
    """Per-GT-identity coverage by its dominant predicted ID."""
    matched: dict[int, Counter] = defaultdict(Counter)
    lengths: Counter = Counter()
    for f in sorted(gt):
        gi, gb = _arrays(gt.get(f, []))
        pi, pb = _arrays(pred.get(f, []))
        for g in gi:
            lengths[int(g)] += 1
        if len(gi) and len(pi):
            iou = box_iou(gb, pb)
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= iou_threshold:
                    matched[int(gi[i])][int(pi[j])] += 1
    per_id = {g: (matched[g].most_common(1)[0][1] / n if matched[g] else 0.0) for g, n in lengths.items()}
    ids_per_gt = {g: len(matched[g]) for g in lengths}
    return {
        "mean_persistence": float(np.mean(list(per_id.values()))) if per_id else 0.0,
        "mean_pred_ids_per_gt": float(np.mean(list(ids_per_gt.values()))) if ids_per_gt else 0.0,
        "per_gt": per_id,
    }


def evaluate_tracking(gt: Sequence, pred: Sequence, iou_threshold: float = 0.5) -> dict:
    out = {**clear_mot(gt, pred, iou_threshold), **hota(gt, pred)}
    p = track_persistence(gt, pred, iou_threshold)
    out["track_persistence"] = p["mean_persistence"]
    out["pred_ids_per_gt"] = p["mean_pred_ids_per_gt"]
    return out


def load_mot_file(path: str | Path) -> Sequence:
    """Read MOTChallenge-style ``frame,id,x,y,w,h,...`` (GT or tracker output)."""
    out: Sequence = defaultdict(list)
    with Path(path).open() as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            fr, tid = int(float(row[0])), int(float(row[1]))
            x, y, w, h = (float(v) for v in row[2:6])
            if len(row) > 6 and float(row[6]) == 0:  # MOT GT "ignore" flag
                continue
            out[fr].append((tid, x, y, x + w, y + h))
    return dict(out)


def load_tracks_csv(path: str | Path, frame_offset: int = 1) -> Sequence:
    """Read this project's ``tracks.csv`` (0-based source frames) -> 1-based MOT frames by default."""
    out: Sequence = defaultdict(list)
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            out[int(row["frame"]) + frame_offset].append(
                (int(row["track_id"]), float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])))
    return dict(out)


def write_mot_file(seq: Sequence, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for fr in sorted(seq):
            for tid, x1, y1, x2, y2 in seq[fr]:
                f.write(f"{fr},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1\n")
    return path
