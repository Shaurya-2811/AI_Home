"""Optional, configurable image enhancement applied before detection.

Enhancement is never assumed to help: ``src.evaluation.environmental_benchmarks.compare_enhancement``
measures detection metrics with and without it. In ``auto`` mode the (cheap) brightness check runs only
every ``check_every`` frames, and enhancement is applied only while the scene is dark.
"""

from __future__ import annotations

import cv2
import numpy as np

METHODS = ("none", "auto", "clahe", "gamma", "dehaze", "clahe_gamma")


def mean_luma(image: np.ndarray) -> float:
    small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def apply_clahe(image: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


_GAMMA_LUTS: dict[float, np.ndarray] = {}


def apply_gamma(image: np.ndarray, gamma: float = 1.6) -> np.ndarray:
    """gamma > 1 brightens."""
    lut = _GAMMA_LUTS.get(gamma)
    if lut is None:
        lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255).astype(np.uint8)
        _GAMMA_LUTS[gamma] = lut
    return cv2.LUT(image, lut)


def apply_dehaze(image: np.ndarray, omega: float = 0.9, t_min: float = 0.2, patch: int = 15) -> np.ndarray:
    """Fast dark-channel-prior dehazing (He et al. 2009) computed at reduced resolution."""
    img = image.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    scale = 320 / max(h, w) if max(h, w) > 320 else 1.0
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1 else img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(small.min(axis=2), kernel)
    n = max(1, dark.size // 1000)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    atmosphere = np.maximum(small.reshape(-1, 3)[idx].max(axis=0), 1e-3)
    trans = 1.0 - omega * cv2.erode((small / atmosphere).min(axis=2), kernel)
    trans = cv2.resize(cv2.GaussianBlur(trans, (0, 0), 3), (w, h)) if scale < 1 else trans
    trans = np.clip(trans, t_min, 1.0)[..., None]
    out = (img - atmosphere) / trans + atmosphere
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


class Preprocessor:
    def __init__(
        self,
        enabled: bool = False,
        method: str = "auto",
        low_light_threshold: float = 60.0,
        check_every: int = 15,
        gamma: float = 1.6,
        clahe_clip: float = 2.0,
    ):
        if method not in METHODS:
            raise ValueError(f"Unknown preprocessing method '{method}'. Choose from {METHODS}")
        self.enabled = enabled and method != "none"
        self.method = method
        self.low_light_threshold = low_light_threshold
        self.check_every = max(1, int(check_every))
        self.gamma = gamma
        self.clahe_clip = clahe_clip
        self._counter = 0
        self._dark = False
        self.applied_frames = 0

    @classmethod
    def from_config(cls, cfg: dict) -> "Preprocessor":
        p = cfg.get("preprocessing", {})
        return cls(
            enabled=p.get("enabled", False),
            method=p.get("method", "auto"),
            low_light_threshold=p.get("low_light_threshold", 60),
            check_every=p.get("check_every", 15),
            gamma=p.get("gamma", 1.6),
            clahe_clip=p.get("clahe_clip", 2.0),
        )

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        """Return (image_for_detection, method_applied). The caller keeps the original for rendering."""
        if not self.enabled:
            return image, "none"
        method = self.method
        if method == "auto":
            if self._counter % self.check_every == 0:
                self._dark = mean_luma(image) < self.low_light_threshold
            self._counter += 1
            if not self._dark:
                return image, "none"
            method = "clahe_gamma"
        out = self.enhance(image, method)
        self.applied_frames += 1
        return out, method

    def enhance(self, image: np.ndarray, method: str) -> np.ndarray:
        if method == "clahe":
            return apply_clahe(image, self.clahe_clip)
        if method == "gamma":
            return apply_gamma(image, self.gamma)
        if method == "clahe_gamma":
            return apply_clahe(apply_gamma(image, self.gamma), self.clahe_clip)
        if method == "dehaze":
            return apply_dehaze(image)
        return image
