import numpy as np

from src.recognition.embeddings import HistogramEmbedder, crop_person
from src.recognition.person_reid import UNKNOWN, ReIDGallery, TrackReID
from src.tracking.tracker import TrackedObject


def _person(color_top, color_bottom, h=128, w=64, noise=0):
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = color_top
    img[h // 2:] = color_bottom
    if noise:
        img = np.clip(img.astype(int) + np.random.default_rng(noise).integers(-15, 15, img.shape), 0, 255).astype(np.uint8)
    return img


def test_histogram_embeddings_are_normalised_and_discriminative():
    e = HistogramEmbedder()
    a1, a2 = _person((0, 0, 200), (200, 0, 0), noise=1), _person((0, 0, 200), (200, 0, 0), noise=2)
    b = _person((0, 200, 0), (50, 50, 50), noise=3)
    em = e.embed([a1, a2, b])
    assert np.allclose(np.linalg.norm(em, axis=1), 1, atol=1e-5)
    assert em[0] @ em[1] > em[0] @ em[2] + 0.3


def test_gallery_threshold_unknown_and_persistence(tmp_path):
    g = ReIDGallery(similarity_threshold=0.8)
    v = np.zeros(8, np.float32)
    v[0] = 1
    l1, _ = g.assign(v)
    w = v.copy()
    w[1] = 0.1
    l2, s = g.assign(w)
    assert l1 == l2 and s > 0.8
    other = np.zeros(8, np.float32)
    other[3] = 1
    l3, s3 = g.assign(other)
    assert l3 != l1 and s3 < 0.8
    # an identity held by another visible track is excluded
    l4, _ = g.assign(v, exclude={l1})
    assert l4 not in (l1,)
    p = g.save(tmp_path / "g.npz")
    g2 = ReIDGallery.load(p)
    assert len(g2) == len(g) and g2.query(other)[0] == l3 and g2._next == g._next


def test_track_reid_needs_enough_frames():
    img = np.zeros((300, 300, 3), np.uint8)
    img[50:250, 50:110] = (0, 0, 255)
    tr = TrackReID(HistogramEmbedder(), ReIDGallery(0.8), min_track_frames=3, sample_every=1, min_crop_height=10)
    t = TrackedObject(1, np.array([50, 50, 110, 250], np.float32), 0.9, 0, 0)
    assert tr.update(img, [t])[1][0] == UNKNOWN
    tr.update(img, [t])
    assert tr.update(img, [t])[1][0].startswith("V-")


def test_crop_person_bounds():
    img = np.zeros((100, 100, 3), np.uint8)
    assert crop_person(img, [-10, -10, 50, 50]).shape[:2] == (53, 53)  # padded, then clipped to the image
    assert crop_person(img, [99, 99, 99.5, 99.5]) is None
