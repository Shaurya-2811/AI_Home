"""Video ingestion: files, webcams and streams, with validation, frame sampling and correct timestamps."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils.visualization import resize_keep_aspect

log = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv")
MAX_CONSECUTIVE_DECODE_FAILURES = 25


class VideoError(Exception):
    """Raised for missing, empty, corrupted or undecodable video sources."""


@dataclass
class VideoMetadata:
    source: str
    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_s: float | None
    codec: str
    is_live: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Frame:
    index: int            # index in the SOURCE stream (not the processed-frame counter)
    timestamp_s: float    # seconds from stream start
    image: np.ndarray     # BGR, possibly downscaled (aspect ratio preserved)
    scale: float = 1.0    # processed / original size


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return chars if chars.isprintable() and chars.strip() else "unknown"


def validate_video_file(
    path: str | Path,
    max_mb: float | None = None,
    allowed_extensions: tuple[str, ...] | list[str] | None = DEFAULT_EXTENSIONS,
) -> None:
    """Cheap pre-checks before handing a file to the decoder."""
    p = Path(path)
    if not p.exists():
        raise VideoError(f"Video file not found: {p}")
    if not p.is_file():
        raise VideoError(f"Not a regular file: {p}")
    size = p.stat().st_size
    if size == 0:
        raise VideoError(f"Video file is empty (0 bytes): {p.name}")
    if max_mb is not None and size > max_mb * 1024 * 1024:
        raise VideoError(f"Video is {size / 1e6:.0f} MB, above the {max_mb:.0f} MB limit: {p.name}")
    if allowed_extensions and p.suffix.lower() not in {e.lower() for e in allowed_extensions}:
        raise VideoError(
            f"Unsupported file extension '{p.suffix}'. Allowed: {', '.join(sorted(allowed_extensions))}"
        )


class VideoReader:
    """Iterate over frames of a video file, camera index or stream URL.

    Frame skipping uses ``grab()`` (no decode) for skipped frames, and timestamps are derived from the
    source frame index and FPS for files, so trackers and exports stay time-correct when sampling.
    """

    def __init__(
        self,
        source: str | int | Path,
        frame_stride: int = 1,
        target_fps: float | None = None,
        max_frames: int | None = None,
        resize_max_side: int | None = None,
        max_mb: float | None = None,
        max_duration_s: float | None = None,
        allowed_extensions: tuple[str, ...] | list[str] | None = DEFAULT_EXTENSIONS,
    ):
        self.source = source
        self.is_live = isinstance(source, int) or (isinstance(source, str) and source.isdigit()) or (
            isinstance(source, str) and source.split(":", 1)[0].lower() in {"rtsp", "rtmp", "http", "https"}
        )
        if not self.is_live:
            validate_video_file(source, max_mb=max_mb, allowed_extensions=allowed_extensions)
        self.requested_stride = max(1, int(frame_stride))
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.resize_max_side = resize_max_side
        self.max_duration_s = max_duration_s
        self.decode_errors = 0
        self._cap: cv2.VideoCapture | None = None
        self.metadata = self._open()
        self.stride = self._compute_stride()

    # ------------------------------------------------------------------ setup
    def _open(self) -> VideoMetadata:
        src = int(self.source) if self.is_live and str(self.source).isdigit() else str(self.source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            if self.is_live:
                raise VideoError(f"Could not open camera/stream '{self.source}'. Check the device index or URL.")
            raise VideoError(
                f"Could not open '{Path(str(self.source)).name}'. The file may be corrupted or use a codec "
                "that this OpenCV/FFmpeg build cannot decode. Try re-encoding to H.264 MP4."
            )
        ok, first = cap.read()
        if not ok or first is None or first.size == 0:
            cap.release()
            raise VideoError(
                f"Opened '{self.source}' but could not decode any frame (corrupted file or unsupported codec)."
            )
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps <= 0 or fps > 1000:
            log.warning("Source reports invalid FPS (%s); assuming 30.", fps)
            fps = 30.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_count = count if count > 0 and not self.is_live else None
        duration = frame_count / fps if frame_count else None
        if self.max_duration_s and duration and duration > self.max_duration_s:
            cap.release()
            raise VideoError(f"Video is {duration:.0f}s long, above the {self.max_duration_s:.0f}s limit.")
        h, w = first.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) if not self.is_live else None
        self._first_frame = first if self.is_live else None
        self._cap = cap
        return VideoMetadata(
            source=str(self.source),
            width=w,
            height=h,
            fps=fps,
            frame_count=frame_count,
            duration_s=duration,
            codec=_fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
            is_live=self.is_live,
        )

    def _compute_stride(self) -> int:
        if self.target_fps and self.target_fps < self.metadata.fps:
            return max(self.requested_stride, int(round(self.metadata.fps / self.target_fps)))
        return self.requested_stride

    @property
    def processed_fps(self) -> float:
        return self.metadata.fps / self.stride

    @property
    def expected_processed_frames(self) -> int | None:
        if self.metadata.frame_count is None:
            return self.max_frames
        n = (self.metadata.frame_count + self.stride - 1) // self.stride
        return min(n, self.max_frames) if self.max_frames else n

    # -------------------------------------------------------------- iteration
    def __iter__(self) -> Iterator[Frame]:
        if self._cap is None:
            raise VideoError("Reader is closed.")
        cap = self._cap
        src_idx = -1
        emitted = 0
        failures = 0
        t0 = time.monotonic()
        pending_first = self._first_frame
        while True:
            if self.max_frames is not None and emitted >= self.max_frames:
                break
            src_idx += 1
            want = src_idx % self.stride == 0
            if pending_first is not None:
                ok, img = True, pending_first
                pending_first = None
            elif want:
                ok, img = cap.read()
            else:
                ok, img = cap.grab(), None
            if not ok:
                if self.metadata.frame_count is not None and src_idx >= self.metadata.frame_count:
                    break
                failures += 1
                self.decode_errors += 1
                if failures >= MAX_CONSECUTIVE_DECODE_FAILURES or self.metadata.frame_count is None and not self.is_live:
                    if self.metadata.frame_count and src_idx < self.metadata.frame_count - 1:
                        log.warning("Stopped at source frame %d: decoding failed repeatedly.", src_idx)
                    break
                continue
            failures = 0
            if not want:
                continue
            if img is None or img.size == 0:
                self.decode_errors += 1
                continue
            ts = (time.monotonic() - t0) if self.is_live else src_idx / self.metadata.fps
            img, scale = resize_keep_aspect(img, self.resize_max_side)
            emitted += 1
            yield Frame(index=src_idx, timestamp_s=ts, image=img, scale=scale)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def probe_video(path: str | Path, **kwargs) -> VideoMetadata:
    with VideoReader(path, **kwargs) as r:
        return r.metadata


def list_cameras(max_index: int = 4) -> list[int]:
    """Best-effort probe for attached cameras (returns indices that open and deliver a frame)."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            found.append(i)
        cap.release()
    return found
