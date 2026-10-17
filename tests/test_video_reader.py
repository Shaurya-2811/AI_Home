import numpy as np
import pytest

from src.video.video_reader import VideoError, VideoReader, validate_video_file
from src.video.video_writer import VideoWriter


def test_reads_all_frames_with_timestamps(synthetic_video):
    path, _ = synthetic_video
    with VideoReader(path) as r:
        assert r.metadata.width == 640 and r.metadata.height == 480
        assert r.metadata.fps == pytest.approx(10, rel=0.01)
        frames = list(r)
    assert len(frames) == 40
    assert frames[0].index == 0 and frames[-1].index == 39
    assert frames[5].timestamp_s == pytest.approx(0.5)


def test_frame_stride_keeps_source_timestamps(synthetic_video):
    path, _ = synthetic_video
    with VideoReader(path, frame_stride=3) as r:
        frames = list(r)
        assert r.processed_fps == pytest.approx(10 / 3)
    assert [f.index for f in frames[:4]] == [0, 3, 6, 9]
    assert frames[2].timestamp_s == pytest.approx(0.6)


def test_target_fps_and_max_frames(synthetic_video):
    path, _ = synthetic_video
    with VideoReader(path, target_fps=5, max_frames=4) as r:
        assert r.stride == 2
        assert len(list(r)) == 4


def test_resize_preserves_aspect_ratio(synthetic_video):
    path, _ = synthetic_video
    with VideoReader(path, resize_max_side=320, max_frames=1) as r:
        f = next(iter(r))
    assert f.image.shape[:2] == (240, 320)
    assert f.scale == pytest.approx(0.5)


def test_missing_empty_and_corrupt_files(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        VideoReader(tmp_path / "nope.mp4")
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(VideoError, match="empty"):
        VideoReader(empty)
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(np.random.default_rng(0).bytes(4096))
    with pytest.raises(VideoError):
        VideoReader(junk)


def test_rejects_extension_and_size(tmp_path, synthetic_video):
    bad = tmp_path / "video.exe"
    bad.write_bytes(b"x" * 10)
    with pytest.raises(VideoError, match="Unsupported"):
        validate_video_file(bad)
    path, _ = synthetic_video
    with pytest.raises(VideoError, match="limit"):
        validate_video_file(path, max_mb=0.00001)


def test_writer_roundtrip(tmp_path):
    out = tmp_path / "w.mp4"
    with VideoWriter(out, 10, (64, 48), h264=False) as w:
        for i in range(5):
            w.write(np.full((48, 64, 3), i * 40, np.uint8))
    with VideoReader(out) as r:
        assert len(list(r)) == 5
