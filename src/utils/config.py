"""YAML configuration loading with deep-merge overrides and dotted-key access."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with ``override`` merged recursively into ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load the default config, merge an optional user config file, then explicit overrides."""
    cfg = _read_yaml(DEFAULT_CONFIG)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        cfg = deep_merge(cfg, _read_yaml(Path(path)))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return data


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_overrides(items: list[str] | None) -> dict:
    """Turn ``["tracker.type=botsort", "detector.conf=0.3"]`` into a nested dict (values parsed as YAML)."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"Override must look like key.sub=value, got: {item!r}")
        key, raw = item.split("=", 1)
        set_dotted(out, key.strip(), yaml.safe_load(raw))
    return out


def validate_config(cfg: dict) -> None:
    """Fail fast on values that would otherwise surface as confusing runtime errors."""
    checks = [
        ("detector.conf", 0.0, 1.0),
        ("detector.iou", 0.0, 1.0),
        ("tracker.track_high_thresh", 0.0, 1.0),
        ("tracker.track_low_thresh", 0.0, 1.0),
        ("tracker.new_track_thresh", 0.0, 1.0),
        ("tracker.match_thresh", 0.0, 1.0),
        ("reid.similarity_threshold", -1.0, 1.0),
    ]
    for key, lo, hi in checks:
        val = get(cfg, key)
        if val is None or not isinstance(val, (int, float)) or not lo <= float(val) <= hi:
            raise ConfigError(f"{key} must be a number in [{lo}, {hi}], got {val!r}")
    for key in ("video.frame_stride", "tracker.track_buffer", "detector.imgsz", "video.batch_size"):
        val = get(cfg, key)
        if not isinstance(val, int) or val < 1:
            raise ConfigError(f"{key} must be a positive integer, got {val!r}")
    if get(cfg, "detector.imgsz") % 32:
        raise ConfigError("detector.imgsz must be a multiple of 32")
    tf = get(cfg, "video.target_fps")
    if tf is not None and (not isinstance(tf, (int, float)) or tf <= 0):
        raise ConfigError(f"video.target_fps must be positive or null, got {tf!r}")
