"""Central logging setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str | None = None, log_file: str | Path | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = (level or os.environ.get("HTA_LOG_LEVEL", "INFO")).upper()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
    )
    # Ultralytics is chatty at INFO; keep our own logs readable.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    _CONFIGURED = True
