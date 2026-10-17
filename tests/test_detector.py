import numpy as np
import pytest

from src.detection.detector import Detections, DetectorError, resolve_weights
from src.detection.preprocessing import Preprocessor, apply_clahe, apply_dehaze, apply_gamma, mean_luma


def test_detections_sanitize_clips_and_drops_degenerate():
    d = Detections([[-10, -5, 50, 60], [10, 10, 10, 40], [0, 0, 5000, 5000], [np.nan, 0, 3, 3]], [1.2, 0.5, 0.4, 0.9], [0, 0, 0, 0])
    s = d.sanitize(100, 80)
    assert len(s) == 2
    assert (s.xyxy[:, [0, 2]] >= 0).all() and (s.xyxy[:, [0, 2]] <= 100).all()
    assert (s.xyxy[:, [1, 3]] <= 80).all()
    assert (s.conf <= 1.0).all()


def test_xywh_conversion_and_masking():
    d = Detections([[0, 0, 10, 20], [10, 10, 30, 30]], [0.9, 0.3], [0, 0])
    np.testing.assert_allclose(d.xywh[0], [5, 10, 10, 20])
    assert len(d[d.conf > 0.5]) == 1


def test_missing_weights_without_download_raises():
    with pytest.raises(DetectorError):
        resolve_weights("does_not_exist_xyz.pt", allow_download=False)


def test_preprocessing_methods_keep_shape():
    img = np.random.default_rng(0).integers(0, 60, (120, 160, 3), dtype=np.uint8)
    for fn in (apply_clahe, apply_gamma, apply_dehaze):
        out = fn(img)
        assert out.shape == img.shape and out.dtype == np.uint8
    assert mean_luma(apply_gamma(img, 2.0)) > mean_luma(img)


def test_auto_preprocessing_only_on_dark_frames():
    pre = Preprocessor(enabled=True, method="auto", low_light_threshold=60, check_every=1)
    dark = np.full((64, 64, 3), 20, np.uint8)
    bright = np.full((64, 64, 3), 200, np.uint8)
    assert pre(dark)[1] != "none"
    assert pre(bright)[1] == "none"
    assert Preprocessor(enabled=False)(dark)[1] == "none"


@pytest.mark.integration
def test_real_detector_outputs_valid_boxes(real_detector):
    import cv2

    from src.utils.sample_video import bundled_image

    img = cv2.imread(str(bundled_image("bus.jpg")))
    d = real_detector.detect(img)
    assert len(d) >= 3, "bus.jpg contains at least 3 clearly visible people"
    h, w = img.shape[:2]
    assert (d.xyxy[:, 0] >= 0).all() and (d.xyxy[:, 2] <= w).all()
    assert (d.xyxy[:, 1] >= 0).all() and (d.xyxy[:, 3] <= h).all()
    assert (d.xyxy[:, 2] > d.xyxy[:, 0]).all() and (d.xyxy[:, 3] > d.xyxy[:, 1]).all()
    assert ((d.conf > 0) & (d.conf <= 1)).all()
    assert (d.cls == 0).all()


@pytest.mark.integration
def test_real_detector_on_blank_frame(real_detector):
    d = real_detector.detect(np.zeros((480, 640, 3), np.uint8))
    assert len(d) == 0


@pytest.mark.integration
def test_real_detector_batch_matches_single(real_detector):
    import cv2

    from src.utils.sample_video import bundled_image

    img = cv2.imread(str(bundled_image("zidane.jpg")))
    single = real_detector.detect(img)
    batch = real_detector.detect_batch([img, img])
    assert len(batch) == 2 and len(batch[0]) == len(single)
