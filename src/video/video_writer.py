"""Annotated video output. Writes with OpenCV, then (optionally) re-encodes to browser-playable H.264."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoWriterError(Exception):
    pass


class VideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], h264: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.size = (int(size[0]), int(size[1]))
        self.h264 = h264 and shutil.which("ffmpeg") is not None
        self._raw_path = self.path.with_name(self.path.stem + ".raw.mp4") if self.h264 else self.path
        self._writer = cv2.VideoWriter(str(self._raw_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        if not self._writer.isOpened():
            raise VideoWriterError(f"Could not open video writer for {self._raw_path}")
        self.frames_written = 0

    def write(self, image: np.ndarray) -> None:
        if (image.shape[1], image.shape[0]) != self.size:
            image = cv2.resize(image, self.size)
        self._writer.write(image)
        self.frames_written += 1

    def close(self) -> Path:
        if self._writer is None:
            return self.path
        self._writer.release()
        self._writer = None
        if self.h264:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(self._raw_path),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
                self._raw_path.unlink(missing_ok=True)
            except (subprocess.SubprocessError, OSError) as exc:
                log.warning("H.264 re-encode failed (%s); keeping mp4v output.", exc)
                shutil.move(str(self._raw_path), self.path)
        return self.path

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
