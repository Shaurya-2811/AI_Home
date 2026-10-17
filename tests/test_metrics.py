import numpy as np
import pytest

from src.evaluation.detection_metrics import average_precision, box_iou, evaluate_detections
from src.evaluation.tracking_metrics import evaluate_tracking, hota, track_persistence


def test_box_iou():
    assert box_iou([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == pytest.approx(50 / 150)
    assert box_iou([[0, 0, 1, 1]], [[2, 2, 3, 3]])[0, 0] == 0


def test_detection_metrics_hand_computed():
    gts = {"a": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]), "b": np.array([[0, 0, 10, 10]])}
    preds = {
        "a": (np.array([[0, 0, 10, 10], [50, 50, 60, 60]]), np.array([0.9, 0.8])),  # 1 TP, 1 FP
        "b": (np.array([[1, 1, 10, 10], [0, 0, 10, 10]]), np.array([0.7, 0.6])),  # 1 TP, 1 duplicate FP
    }
    r = evaluate_detections(preds, gts, conf_threshold=0.5)
    assert (r.tp, r.fp, r.fn) == (2, 2, 1)
    assert r.precision == pytest.approx(0.5) and r.recall == pytest.approx(2 / 3)
    assert r.f1 == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert 0 < r.ap50 < 1 and r.ap50_95 <= r.ap50


def test_perfect_detection_ap_is_one():
    gts = {i: np.array([[0, 0, 10, 10]]) for i in range(5)}
    preds = {i: (np.array([[0, 0, 10, 10]]), np.array([0.9])) for i in range(5)}
    r = evaluate_detections(preds, gts)
    assert r.ap50 == pytest.approx(1.0) and r.ap50_95 == pytest.approx(1.0) and r.f1 == 1.0


def test_ap_no_predictions():
    assert average_precision(np.array([]), np.array([], bool), 3) == 0.0


def _seq(n=20, swap_at=None):
    gt, pr = {}, {}
    for f in range(1, n + 1):
        gt[f] = [(1, f, 0, f + 10, 20), (2, 100 + f, 0, 110 + f, 20)]
        a, b = (11, 12) if swap_at is None or f < swap_at else (12, 11)
        pr[f] = [(a, f, 0, f + 10, 20), (b, 100 + f, 0, 110 + f, 20)]
    return gt, pr


def test_perfect_tracking():
    gt, pr = _seq()
    m = evaluate_tracking(gt, pr)
    for k in ("HOTA", "DetA", "AssA", "LocA", "idf1", "mota", "track_persistence"):
        assert m[k] == pytest.approx(1.0), k
    assert m["num_switches"] == 0


def test_identity_swap_hurts_association_not_detection():
    gt, pr = _seq(swap_at=11)
    m = evaluate_tracking(gt, pr)
    assert m["DetA"] == pytest.approx(1.0)
    assert m["AssA"] < 0.7 and m["HOTA"] < 0.85
    assert m["num_switches"] == 2
    assert m["idf1"] == pytest.approx(0.5)
    assert track_persistence(gt, pr)["mean_persistence"] == pytest.approx(0.5)


def test_hota_empty_prediction():
    gt, _ = _seq()
    assert hota(gt, {})["HOTA"] == 0.0


def test_hota_matches_trackeval_if_installed():
    pytest.importorskip("trackeval")
    from trackeval.metrics import HOTA

    from src.evaluation.detection_metrics import box_iou as iou

    rng = np.random.default_rng(0)
    gt, pr = {}, {}
    for f in range(1, 41):
        gt[f] = [(i, 10 + i * 80 + f * 2, 50, 60 + i * 80 + f * 2, 200) for i in range(5) if not (i == 2 and 15 < f < 20)]
        pr[f] = []
        for i, x1, y1, x2, y2 in gt[f]:
            if rng.random() < 0.1:
                continue
            pid = (i + 1) % 5 if (f >= 25 and i in (0, 1)) else i
            n = rng.normal(0, 4, 4)
            pr[f].append((pid + 10, x1 + n[0], y1 + n[1], x2 + n[2], y2 + n[3]))
        if rng.random() < 0.2:
            pr[f].append((99, 300, 300, 340, 400))
    frames = sorted(gt)
    gids = sorted({r[0] for v in gt.values() for r in v})
    pids = sorted({r[0] for v in pr.values() for r in v})
    gm, pm = {g: i for i, g in enumerate(gids)}, {p: i for i, p in enumerate(pids)}
    data = {"num_gt_ids": len(gids), "num_tracker_ids": len(pids), "gt_ids": [], "tracker_ids": [], "similarity_scores": [],
            "num_gt_dets": 0, "num_tracker_dets": 0, "num_timesteps": len(frames)}
    for f in frames:
        g = np.array([gm[r[0]] for r in gt[f]], int)
        p = np.array([pm[r[0]] for r in pr[f]], int)
        data["gt_ids"].append(g)
        data["tracker_ids"].append(p)
        data["similarity_scores"].append(iou(np.array([r[1:] for r in gt[f]]).reshape(-1, 4), np.array([r[1:] for r in pr[f]]).reshape(-1, 4)))
        data["num_gt_dets"] += len(g)
        data["num_tracker_dets"] += len(p)
    ref = HOTA().eval_sequence(data)
    ours = hota(gt, pr)
    for k in ("HOTA", "DetA", "AssA", "LocA"):
        assert ours[k] == pytest.approx(float(np.mean(ref[k])), abs=1e-6), k
