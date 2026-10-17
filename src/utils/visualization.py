"""Drawing helpers for annotated output video."""

from __future__ import annotations

import colorsys
from collections.abc import Iterable, Sequence

import cv2
import numpy as np


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR color per track id (golden-ratio hue stepping)."""
    hue = (int(track_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def resize_keep_aspect(image: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is ``max_side``; never distorts. Returns (image, scale)."""
    if not max_side:
        return image, 1.0
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image, 1.0
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def draw_tracks(
    image: np.ndarray,
    tracks: Iterable,
    trails: dict[int, Sequence[tuple[float, float]]] | None = None,
    draw_confidence: bool = True,
    labels: dict[int, str] | None = None,
) -> np.ndarray:
    """Draw boxes, ``ID n`` labels, confidence and optional trajectory trails in place."""
    h, w = image.shape[:2]
    thickness = max(1, round(min(h, w) / 360))
    font_scale = max(0.4, min(h, w) / 1100)
    for tr in tracks:
        tid = int(tr.track_id)
        color = color_for_id(tid)
        x1, y1, x2, y2 = (int(round(v)) for v in tr.xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        text = f"ID {tid}"
        if draw_confidence:
            text += f" {tr.score:.2f}"
        if labels and tid in labels:
            text += f" | {labels[tid]}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(th + base, y1)
        cv2.rectangle(image, (x1, ty - th - base), (x1 + tw + 2, ty), color, -1)
        cv2.putText(image, text, (x1 + 1, ty - base), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        if trails and tid in trails and len(trails[tid]) > 1:
            pts = np.asarray(trails[tid], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], False, color, thickness)
    return image


def draw_header(image: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    h, w = image.shape[:2]
    font_scale = max(0.45, min(h, w) / 1000)
    y = 0
    for line in lines:
        (tw, th), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y += th + base + 4
        cv2.rectangle(image, (0, y - th - base - 4), (tw + 8, y), (0, 0, 0), -1)
        cv2.putText(image, line, (4, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return image
