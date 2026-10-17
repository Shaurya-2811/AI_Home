import numpy as np
import pytest

from src.detection.detector import Detections
from src.tracking.track_manager import TrackManager
from src.tracking.tracker import PersonTracker, available_trackers
from tests.conftest import gt_to_detections

BASE = {"track_high_thresh": 0.25, "track_low_thresh": 0.1, "new_track_thresh": 0.25, "track_buffer": 30,
        "match_thresh": 0.8, "fuse_score": True, "gmc_method": "none"}


def _run(tracker, frames, gt, drop=None):
    """Return {gt_id: set(track_ids)} over the sequence."""
    mapping: dict[int, set] = {}
    for f, (img, rows) in enumerate(zip(frames, gt)):
        dets = gt_to_detections(rows)
        if drop and drop[0] <= f < drop[1]:
            dets = Detections()
        tracks = tracker.update(dets, img)
        for t in tracks:
            # associate track to the GT object with the highest IoU
            best, best_iou = None, 0.0
            for r in rows:
                if not r[5]:
                    continue
                x1, y1, x2, y2 = r[1:5]
                ix = max(0, min(x2, t.xyxy[2]) - max(x1, t.xyxy[0]))
                iy = max(0, min(y2, t.xyxy[3]) - max(y1, t.xyxy[1]))
                inter = ix * iy
                iou = inter / ((x2 - x1) * (y2 - y1) + (t.xyxy[2] - t.xyxy[0]) * (t.xyxy[3] - t.xyxy[1]) - inter)
                if iou > best_iou:
                    best, best_iou = r[0], iou
            if best is not None and best_iou > 0.5:
                mapping.setdefault(best, set()).add(t.track_id)
    return mapping


@pytest.mark.parametrize("ttype", ["bytetrack", "botsort", "ocsort"])
def test_ids_stable_in_simple_sequence(synthetic, ttype):
    frames, gt = synthetic
    mapping = _run(PersonTracker({**BASE, "type": ttype}), frames, gt)
    assert set(mapping) == {1, 2, 3}
    for gid, ids in mapping.items():
        assert len(ids) == 1, f"{ttype}: object {gid} got several track ids {ids}"
    assert len(set().union(*mapping.values())) == 3


def test_id_survives_short_occlusion(synthetic):
    """Object 2 is hidden for 4 frames (18-21); its ID must be the same before and after."""
    frames, gt = synthetic
    mapping = _run(PersonTracker({**BASE, "type": "bytetrack"}), frames, gt)
    assert len(mapping[2]) == 1


def test_all_detections_missing_does_not_crash_and_recovers(synthetic):
    frames, gt = synthetic
    tracker = PersonTracker({**BASE, "type": "bytetrack"})
    mapping = _run(tracker, frames, gt, drop=(10, 14))
    assert all(len(v) == 1 for v in mapping.values())


def test_empty_frames_only():
    tracker = PersonTracker({**BASE, "type": "bytetrack"})
    for _ in range(5):
        assert tracker.update(Detections(), np.zeros((10, 10, 3), np.uint8)) == []


def test_track_buffer_rescaled_for_frame_stride():
    assert PersonTracker({**BASE, "type": "bytetrack", "track_buffer": 30}, frame_stride=3).args.track_buffer == 10


def test_unknown_tracker_rejected():
    with pytest.raises(ValueError):
        PersonTracker({**BASE, "type": "nope"})


def test_all_ultralytics_trackers_run(synthetic):
    frames, gt = synthetic
    for ttype in available_trackers():
        tr = PersonTracker({**BASE, "type": ttype})
        for img, rows in zip(frames[:10], gt[:10]):
            tr.update(gt_to_detections(rows), img)


def test_track_manager_history_events_and_exports(tmp_path, synthetic):
    import csv
    import json

    frames, gt = synthetic
    tracker = PersonTracker({**BASE, "type": "bytetrack"})
    mgr = TrackManager()
    for f, (img, rows) in enumerate(zip(frames, gt)):
        dets = gt_to_detections(rows)
        mgr.update(f, f / 10, tracker.update(dets, img), len(dets))
    s = mgr.summary()
    assert s["unique_track_ids"] == 3 and s["processed_frames"] == 40
    assert "not the number of unique real-world people" in s["unique_track_ids_note"]
    traj = next(iter(mgr.trajectories.values()))
    assert traj.num_observations > 10 and traj.duration_s > 0 and traj.path_length_px > 0
    assert any(e["type"] == "track_start" for e in mgr.events)
    c = mgr.export_csv(tmp_path / "t.csv")
    rows = list(csv.DictReader(c.open()))
    assert len(rows) == sum(1 for _ in mgr.rows) and {"frame", "timestamp_s", "track_id", "x1", "confidence"} <= set(rows[0])
    j = json.loads(mgr.export_json(tmp_path / "t.json").read_text())
    assert len(j["tracks"]) == 3 and j["tracks"][0]["trajectory"][0]["bbox"]
