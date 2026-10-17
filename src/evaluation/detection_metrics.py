"""Detection metrics: precision, recall, F1, TP/FP/FN counts, AP@0.5 and COCO-style AP@[0.5:0.95].

Matching follows the COCO/VOC convention: predictions sorted by confidence are greedily matched to the
highest-IoU unmatched ground truth in the same image; AP uses 101-point interpolation (COCO).
Boxes are pixel ``xyxy``. Tested against hand-computed cases in ``tests/test_metrics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


@dataclass
class DetectionResult:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    ap50: float
    ap50_95: float
    num_images: int
    num_gt: int
    num_pred: int
    conf_threshold: float

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _match(preds: dict, gts: dict, iou_thr: float):
    """Return (scores, is_tp) over all predictions, and total GT count."""
    scores, tps = [], []
    n_gt = 0
    for img_id in set(gts) | set(preds):
        g = np.asarray(gts.get(img_id, np.zeros((0, 4))), np.float64).reshape(-1, 4)
        p_boxes, p_scores = preds.get(img_id, (np.zeros((0, 4)), np.zeros(0)))
        p_boxes = np.asarray(p_boxes, np.float64).reshape(-1, 4)
        p_scores = np.asarray(p_scores, np.float64).reshape(-1)
        n_gt += len(g)
        order = np.argsort(-p_scores, kind="stable")
        ious = box_iou(p_boxes[order], g)
        used = np.zeros(len(g), bool)
        for i, pi in enumerate(order):
            ok = False
            if len(g):
                cand = np.where(~used, ious[i], -1.0)
                j = int(np.argmax(cand))
                if cand[j] >= iou_thr:
                    used[j] = True
                    ok = True
            scores.append(p_scores[pi])
            tps.append(ok)
    return np.asarray(scores), np.asarray(tps, bool), n_gt


def average_precision(scores: np.ndarray, tps: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(tps[order])
    fp = np.cumsum(~tps[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    # precision envelope
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    rec_points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, rec_points, side="left")
    q = np.array([precision[i] if i < len(precision) else 0.0 for i in idx])
    return float(q.mean())


def evaluate_detections(preds: dict, gts: dict, conf_threshold: float = 0.25, iou_threshold: float = 0.5) -> DetectionResult:
    """``preds``: {image_id: (boxes Nx4, scores N)}; ``gts``: {image_id: boxes Mx4}.

    P/R/F1/TP/FP/FN are reported at ``conf_threshold``; AP metrics use all predictions.
    """
    s, t, n_gt = _match(preds, gts, iou_threshold)
    ap50 = average_precision(s, t, n_gt)
    aps = [ap50 if thr == 0.5 else average_precision(*_match(preds, gts, float(thr))) for thr in IOU_THRESHOLDS]
    keep = s >= conf_threshold
    tp = int(t[keep].sum())
    fp = int(keep.sum() - tp)
    fn = int(n_gt - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionResult(
        precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn,
        ap50=ap50, ap50_95=float(np.nanmean(aps)) if n_gt else float("nan"),
        num_images=len(set(gts) | set(preds)), num_gt=n_gt, num_pred=len(s), conf_threshold=conf_threshold,
    )


def load_yolo_ground_truth(images: list, labels_dir=None) -> dict:
    """Read YOLO labels for a list of image paths -> {image_path: pixel xyxy boxes}."""
    from pathlib import Path

    import cv2

    from src.training.augmentations import label_path_for, read_yolo_boxes_px

    out = {}
    for img_path in images:
        p = Path(img_path)
        lbl = (Path(labels_dir) / f"{p.stem}.txt") if labels_dir else label_path_for(p)
        im = cv2.imread(str(p))
        if im is None:
            continue
        out[str(p)] = read_yolo_boxes_px(lbl, im.shape[1], im.shape[0]) if lbl.exists() else np.zeros((0, 4))
    return out
