"""Environmental augmentations for robustness training, and fixed-severity corruptions for evaluation.

Training: ``EnvironmentalAugmenter`` applies each transform with its own probability and moderate random
strength. It is used offline (``augment_split``) to add augmented copies of TRAIN images only; validation
and test images are never augmented. All transforms are photometric or box-preserving, so YOLO labels are
copied unchanged (occlusion patches are limited to cover at most ``max_occlusion`` of any box).

Evaluation: ``CORRUPTIONS`` applies one condition at a fixed severity (1-3) so robustness can be measured
per condition. These are synthetic approximations, reported separately from real-condition results.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

Image = np.ndarray


def _u8(x: np.ndarray) -> Image:
    return np.clip(x, 0, 255).astype(np.uint8)


def brightness_contrast(img: Image, rng: np.random.Generator, b: float = 0.3, c: float = 0.3) -> Image:
    alpha = 1.0 + rng.uniform(-c, c)
    beta = 255 * rng.uniform(-b, b)
    return _u8(img.astype(np.float32) * alpha + beta)


def gamma(img: Image, rng: np.random.Generator, lo: float = 0.6, hi: float = 1.6) -> Image:
    g = rng.uniform(lo, hi)
    lut = _u8(((np.arange(256) / 255.0) ** g) * 255)
    return cv2.LUT(img, lut)


def color_temperature(img: Image, rng: np.random.Generator, strength: float = 0.15) -> Image:
    t = rng.uniform(-strength, strength)  # >0 warm, <0 cool
    gains = np.array([1 - t, 1.0, 1 + t], np.float32)  # BGR
    return _u8(img.astype(np.float32) * gains)


def motion_blur(img: Image, rng: np.random.Generator, max_k: int = 15) -> Image:
    k = int(rng.integers(3, max_k + 1)) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    return cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))


def defocus_blur(img: Image, rng: np.random.Generator, max_r: int = 5) -> Image:
    r = int(rng.integers(1, max_r + 1))
    kernel = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(kernel, (r, r), r, 1.0, -1)
    return cv2.filter2D(img, -1, kernel / kernel.sum())


def sensor_noise(img: Image, rng: np.random.Generator, max_sigma: float = 20.0) -> Image:
    sigma = rng.uniform(3, max_sigma)
    return _u8(img.astype(np.float32) + rng.normal(0, sigma, img.shape))


def shadow(img: Image, rng: np.random.Generator) -> Image:
    h, w = img.shape[:2]
    pts = np.array([[rng.integers(0, w), 0], [rng.integers(0, w), h], [rng.integers(0, w), h], [rng.integers(0, w), 0]], np.int32)
    mask = np.zeros((h, w), np.float32)
    cv2.fillPoly(mask, [pts], 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), max(h, w) / 50)
    factor = 1 - rng.uniform(0.3, 0.6) * mask[..., None]
    return _u8(img.astype(np.float32) * factor)


def glare(img: Image, rng: np.random.Generator, radius_range: tuple[float, float] = (0.15, 0.4),
          intensity: tuple[float, float] = (120, 230)) -> Image:
    """Bright light source / lens flare at a random point in the upper part of the frame."""
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h * 0.6)
    radius = rng.uniform(*radius_range) * max(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    flare = np.clip(1 - d / radius, 0, 1) ** 2 * rng.uniform(*intensity)
    return _u8(img.astype(np.float32) + flare[..., None])


def fog(img: Image, rng: np.random.Generator, lo: float = 0.25, hi: float = 0.6) -> Image:
    """Atmospheric scattering model I = J*t + A*(1-t) with depth increasing towards the top of the frame."""
    h, w = img.shape[:2]
    beta = rng.uniform(lo, hi)
    depth = np.linspace(1.0, 0.3, h, dtype=np.float32)[:, None]
    t = np.exp(-beta * 3 * depth)[..., None]
    a = rng.uniform(200, 240)
    return _u8(img.astype(np.float32) * t + a * (1 - t))


def rain(img: Image, rng: np.random.Generator, density: float = 0.002) -> Image:
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    n = int(density * h * w * rng.uniform(0.5, 1.5))
    length = int(rng.uniform(0.02, 0.05) * h)
    angle = rng.uniform(-0.3, 0.3)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    for x, y in zip(xs, ys):
        cv2.line(layer, (int(x), int(y)), (int(x + angle * length), int(y + length)), 1.0, 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    out = img.astype(np.float32) * 0.85 + layer[..., None] * 180
    return _u8(out)


def low_light(img: Image, rng: np.random.Generator, lo: float = 0.15, hi: float = 0.45) -> Image:
    """Night-like: darken with gamma, add shot/read noise, slight blue shift."""
    f = img.astype(np.float32) / 255.0
    f = (f ** rng.uniform(1.5, 2.5)) * rng.uniform(lo, hi)
    f = f + rng.normal(0, rng.uniform(0.005, 0.02), f.shape) + np.sqrt(np.clip(f, 0, 1)) * rng.normal(0, 0.02, f.shape)
    f[..., 0] *= 1.1
    return _u8(f * 255)


def resolution_degradation(img: Image, rng: np.random.Generator, lo: float = 0.25, hi: float = 0.6) -> Image:
    h, w = img.shape[:2]
    s = rng.uniform(lo, hi)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    q = int(rng.integers(30, 80))
    ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])
    small = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else small
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def occlusion(img: Image, rng: np.random.Generator, boxes: np.ndarray | None = None, max_occlusion: float = 0.4) -> Image:
    """Random grey/noise patches. With boxes (pixel xyxy), a patch never covers more than max_occlusion of a box."""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(int(rng.integers(1, 4))):
        for _try in range(10):
            pw, ph = int(rng.uniform(0.05, 0.2) * w), int(rng.uniform(0.05, 0.2) * h)
            x, y = int(rng.integers(0, max(1, w - pw))), int(rng.integers(0, max(1, h - ph)))
            if boxes is None or len(boxes) == 0 or _max_cover(boxes, (x, y, x + pw, y + ph)) <= max_occlusion:
                out[y:y + ph, x:x + pw] = rng.integers(0, 255, (ph, pw, 3), dtype=np.uint8) if rng.random() < 0.5 else rng.integers(60, 200)
                break
    return out


def _max_cover(boxes: np.ndarray, patch: tuple[int, int, int, int]) -> float:
    px1, py1, px2, py2 = patch
    ix = np.clip(np.minimum(boxes[:, 2], px2) - np.maximum(boxes[:, 0], px1), 0, None)
    iy = np.clip(np.minimum(boxes[:, 3], py2) - np.maximum(boxes[:, 1], py1), 0, None)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6)
    return float((ix * iy / area).max())


# Default per-transform probabilities: moderate, so most augmented images get 1-3 effects.
DEFAULT_PROBS = {
    "brightness_contrast": 0.3, "gamma": 0.2, "color_temperature": 0.2, "motion_blur": 0.1, "defocus_blur": 0.1,
    "sensor_noise": 0.15, "shadow": 0.1, "glare": 0.05, "fog": 0.1, "rain": 0.1, "low_light": 0.15,
    "resolution_degradation": 0.1, "occlusion": 0.1,
}
TRANSFORMS: dict[str, Callable] = {
    "brightness_contrast": brightness_contrast, "gamma": gamma, "color_temperature": color_temperature,
    "motion_blur": motion_blur, "defocus_blur": defocus_blur, "sensor_noise": sensor_noise, "shadow": shadow,
    "glare": glare, "fog": fog, "rain": rain, "low_light": low_light,
    "resolution_degradation": resolution_degradation, "occlusion": occlusion,
}
# Mutually exclusive groups: don't stack fog with low light etc., which yields unrealistic images.
EXCLUSIVE = [{"fog", "low_light", "glare"}, {"motion_blur", "defocus_blur"}]


class EnvironmentalAugmenter:
    def __init__(self, probs: dict[str, float] | None = None, seed: int = 0):
        self.probs = {**DEFAULT_PROBS, **(probs or {})}
        unknown = set(self.probs) - set(TRANSFORMS)
        if unknown:
            raise ValueError(f"Unknown augmentations: {sorted(unknown)}")
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: Image, boxes: np.ndarray | None = None) -> tuple[Image, list[str]]:
        applied: list[str] = []
        names = list(self.probs)
        self.rng.shuffle(names)
        for name in names:
            if self.rng.random() >= self.probs[name]:
                continue
            if any(name in g and g & set(applied) for g in EXCLUSIVE):
                continue
            fn = TRANSFORMS[name]
            img = fn(img, self.rng, boxes) if name == "occlusion" else fn(img, self.rng)
            applied.append(name)
        return img, applied


def augment_split(image_list: str | Path, out_dir: str | Path, copies: int = 1, seed: int = 0,
                  probs: dict[str, float] | None = None) -> Path:
    """Write augmented copies of the images listed in a split file (train only!) plus their labels.

    Returns a new split file listing originals + augmented copies. Labels are found by the YOLO
    convention (``.../images/x.jpg`` -> ``.../labels/x.txt``).
    """
    image_list, out_dir = Path(image_list), Path(out_dir)
    if image_list.stem in ("val", "test"):
        raise ValueError("Refusing to augment a validation/test split: augmentations are for training only.")
    aug = EnvironmentalAugmenter(probs, seed)
    img_out, lbl_out = out_dir / "images", out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    from src.data.dataset_splitter import read_split

    originals = read_split(image_list)
    new_items = list(originals)
    log_rows = []
    for src in originals:
        p = Path(src)
        img = cv2.imread(str(p))
        if img is None:
            continue
        lbl = label_path_for(p)
        boxes = read_yolo_boxes_px(lbl, img.shape[1], img.shape[0]) if lbl.exists() else np.zeros((0, 4))
        for k in range(copies):
            out, applied = aug(img, boxes)
            if not applied:
                continue
            name = f"{p.stem}__aug{k}"
            cv2.imwrite(str(img_out / f"{name}.jpg"), out)
            if lbl.exists():
                shutil.copy2(lbl, lbl_out / f"{name}.txt")
            else:
                (lbl_out / f"{name}.txt").write_text("")
            new_items.append(str((img_out / f"{name}.jpg").resolve()))
            log_rows.append(f"{name},{'|'.join(applied)}")
    (out_dir / "augmentations.csv").write_text("image,augmentations\n" + "\n".join(log_rows) + "\n")
    out_list = out_dir / "train_augmented.txt"
    out_list.write_text("".join(f"{x}\n" for x in new_items))
    return out_list


def label_path_for(img_path: Path) -> Path:
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def read_yolo_boxes_px(label: Path, w: int, h: int) -> np.ndarray:
    rows = [l.split() for l in label.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32)
    a = np.asarray([[float(v) for v in r[1:5]] for r in rows], np.float32)
    return np.stack([(a[:, 0] - a[:, 2] / 2) * w, (a[:, 1] - a[:, 3] / 2) * h,
                     (a[:, 0] + a[:, 2] / 2) * w, (a[:, 1] + a[:, 3] / 2) * h], 1)


# ------------------------------------------------------------ evaluation corruptions
def _fixed(fn: Callable, **kw_by_sev) -> Callable:
    def apply(img: Image, severity: int = 2, seed: int = 0) -> Image:
        rng = np.random.default_rng(seed)
        kwargs = {k: v[severity - 1] for k, v in kw_by_sev.items()}
        return fn(img, rng, **kwargs)
    return apply


CORRUPTIONS: dict[str, Callable] = {
    "low_light": _fixed(low_light, lo=[0.45, 0.3, 0.15], hi=[0.45, 0.3, 0.15]),
    "fog": _fixed(fog, lo=[0.2, 0.4, 0.7], hi=[0.2, 0.4, 0.7]),
    "rain": _fixed(rain, density=[0.001, 0.002, 0.004]),
    "motion_blur": _fixed(motion_blur, max_k=[5, 11, 21]),
    "sensor_noise": _fixed(sensor_noise, max_sigma=[8, 16, 30]),
    "low_resolution": _fixed(resolution_degradation, lo=[0.6, 0.4, 0.25], hi=[0.6, 0.4, 0.25]),
    # glare centred per image (seeded), growing radius/intensity: severity 3 washes out ~half the frame
    "glare": _fixed(glare, radius_range=[(0.3, 0.3), (0.5, 0.5), (0.8, 0.8)], intensity=[(150, 150), (200, 200), (255, 255)]),
}
