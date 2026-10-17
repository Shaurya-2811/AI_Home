"""Hardware detection and device selection (CUDA when available, CPU fallback)."""

from __future__ import annotations

import os
import platform
import sys

import logging

log = logging.getLogger(__name__)


def resolve_device(preference: str | None = "auto") -> str:
    """Return a torch device string. ``auto`` picks cuda:0, then mps, then cpu.

    An explicit ``cuda`` request on a machine without CUDA falls back to CPU with a warning
    rather than crashing, so configs stay portable.
    """
    pref = (preference or "auto").lower()
    try:
        import torch
    except ImportError:
        return "cpu"
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            log.warning("CUDA requested but not available; falling back to CPU.")
            return "cpu"
        return pref if ":" in pref else "cuda:0"
    if pref == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            log.warning("MPS requested but not available; falling back to CPU.")
            return "cpu"
    return pref


def use_half(device: str, setting: str | bool = "auto") -> bool:
    if isinstance(setting, bool):
        return setting and device.startswith("cuda")
    return device.startswith("cuda")


def hardware_report() -> dict:
    """Collect the facts needed to reason about speed and training feasibility."""
    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1),
                }
                for i in range(torch.cuda.device_count())
            ]
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        info["torch"] = None
    for mod in ("ultralytics", "cv2", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    return info


def _ram_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * size / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


def peak_memory_mb() -> dict:
    """Peak resident memory of this process (and CUDA allocator peak when on GPU)."""
    out: dict = {}
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["peak_rss_mb"] = round(rss / 1024 if sys.platform != "darwin" else rss / 1e6, 1)
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            out["peak_cuda_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except ImportError:
        pass
    return out
