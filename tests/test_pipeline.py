import csv
import json

import pytest

from src.utils.config import ConfigError, load_config, parse_overrides
from src.video.inference_pipeline import InferencePipeline
from src.video.video_reader import VideoError
from tests.conftest import FailingDetector, ScriptedDetector, gt_to_detections


def _check_outputs(res, expect_rows=True):
    rows = list(csv.DictReader(res.csv_path.open()))
    data = json.loads(res.json_path.read_text())
    assert {"video", "summary", "tracks", "events", "timing", "disclaimers"} <= set(data)
    if expect_rows:
        assert rows and all(float(r["confidence"]) <= 1 for r in rows)
        assert all(float(r["x2"]) > float(r["x1"]) for r in rows)
    return rows, data


def test_pipeline_with_scripted_detector(tmp_path, synthetic_video, base_cfg):
    path, gt = synthetic_video
    det = ScriptedDetector([gt_to_detections(r) for r in gt])
    res = InferencePipeline(base_cfg, detector=det).run(path, output_dir=tmp_path, run_name="r")
    rows, data = _check_outputs(res)
    assert res.summary["unique_track_ids"] == 3
    assert res.video_path.exists() and res.video_path.stat().st_size > 0
    assert data["video"]["width"] == 640 and len(data["tracks"]) == 3
    assert res.timing["processed_frames"] == 40 and res.timing["processing_fps"] > 0


def test_pipeline_frame_stride_timestamps(tmp_path, synthetic_video, base_cfg):
    path, gt = synthetic_video
    base_cfg["video"]["frame_stride"] = 2
    det = ScriptedDetector([gt_to_detections(r) for r in gt[::2]])
    res = InferencePipeline(base_cfg, detector=det).run(path, output_dir=tmp_path, run_name="s")
    rows, _ = _check_outputs(res)
    frames = sorted({int(r["frame"]) for r in rows})
    assert all(f % 2 == 0 for f in frames)
    r = next(r for r in rows if int(r["frame"]) == 10)
    assert float(r["timestamp_s"]) == pytest.approx(1.0)


def test_pipeline_survives_detector_failures(tmp_path, synthetic_video, base_cfg):
    path, _ = synthetic_video
    res = InferencePipeline(base_cfg, detector=FailingDetector(fail_every=3)).run(path, output_dir=tmp_path, run_name="f")
    assert res.timing["processed_frames"] == 40
    assert any("Detector failed" in w for w in res.warnings)
    _check_outputs(res)


def test_pipeline_resize_exports_original_coordinates(tmp_path, synthetic_video, base_cfg):
    path, gt = synthetic_video
    base_cfg["video"]["resize_max_side"] = 320
    from src.detection.detector import Detections

    scripted = [gt_to_detections(r) for r in gt]
    det = ScriptedDetector([Detections(d.xyxy * 0.5, d.conf, d.cls) for d in scripted])
    res = InferencePipeline(base_cfg, detector=det).run(path, output_dir=tmp_path, run_name="z")
    rows, _ = _check_outputs(res)
    assert max(float(r["x2"]) for r in rows) > 330  # back in 640-wide coordinates


def test_pipeline_with_reid_labels(tmp_path, synthetic_video, base_cfg):
    path, gt = synthetic_video
    base_cfg["reid"].update(enabled=True, min_track_frames=3, sample_every=2, min_crop_height=20)
    det = ScriptedDetector([gt_to_detections(r) for r in gt])
    res = InferencePipeline(base_cfg, detector=det).run(path, output_dir=tmp_path, run_name="reid")
    rows, data = _check_outputs(res)
    labels = {r["reid_label"] for r in rows}
    assert "unknown" in labels  # before enough samples
    assert any(l.startswith("V-") for l in labels)
    assert any("not verified identities" in d for d in data["disclaimers"])


def test_pipeline_rejects_invalid_video(tmp_path, base_cfg):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(VideoError):
        InferencePipeline(base_cfg, detector=ScriptedDetector([])).run(bad, output_dir=tmp_path)


def test_config_validation():
    with pytest.raises(ConfigError):
        load_config(overrides={"detector": {"conf": 1.5}})
    with pytest.raises(ConfigError):
        load_config(overrides={"detector": {"imgsz": 100}})
    assert parse_overrides(["tracker.type=botsort", "detector.conf=0.3"]) == {"tracker": {"type": "botsort"}, "detector": {"conf": 0.3}}


@pytest.mark.integration
def test_end_to_end_real_model(tmp_path, pan_clip, real_detector):
    """Full pipeline with the real pretrained YOLO on a clip panning over a photo of people."""
    cfg = load_config(overrides={"device": "cpu", "output": {"h264": True, "save_crops": True, "crop_every": 5}})
    res = InferencePipeline(cfg, detector=real_detector).run(pan_clip, output_dir=tmp_path, run_name="e2e")
    rows, data = _check_outputs(res)
    assert res.summary["unique_track_ids"] >= 3
    assert res.summary["max_concurrent_tracks"] >= 3
    # the same people stay in view for the whole pan: tracks should be long, not fragmented
    assert res.summary["mean_track_length_frames"] >= 10
    assert any(res.crops_dir.iterdir())
    assert res.video_path.suffix == ".mp4" and res.video_path.stat().st_size > 1000


@pytest.mark.integration
def test_cli_track_command(tmp_path, pan_clip):
    from tests.conftest import WEIGHTS

    if not WEIGHTS.exists():
        pytest.skip("weights not present")
    import cli

    assert cli.main(["track", str(pan_clip), "--out", str(tmp_path), "--set", "output.h264=false", "video.max_frames=5"]) == 0
    assert list(tmp_path.glob("run_*/tracks.json"))
