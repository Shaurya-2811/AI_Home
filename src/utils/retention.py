"""Retention policy: delete uploads and generated outputs older than a configured age."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

log = logging.getLogger(__name__)


def cleanup_older_than(root: str | Path, max_age_s: float, keep: tuple[str, ...] = (".gitkeep",)) -> list[Path]:
    """Remove files/dirs directly under ``root`` whose mtime is older than ``max_age_s``. Returns removed paths."""
    root = Path(root)
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    cutoff = time.time() - max_age_s
    for entry in root.iterdir():
        if entry.name in keep:
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed.append(entry)
        except OSError as exc:
            log.warning("Retention cleanup could not remove %s: %s", entry, exc)
    return removed


def apply_retention(cfg: dict, uploads_dir: str | Path, outputs_dir: str | Path) -> dict:
    ret = cfg.get("retention", {})
    if not ret.get("enabled", True):
        return {"uploads": 0, "outputs": 0}
    up = cleanup_older_than(uploads_dir, float(ret.get("uploads_hours", 24)) * 3600)
    out = cleanup_older_than(outputs_dir, float(ret.get("outputs_days", 7)) * 86400)
    if up or out:
        log.info("Retention removed %d upload(s) and %d output item(s)", len(up), len(out))
    return {"uploads": len(up), "outputs": len(out)}
