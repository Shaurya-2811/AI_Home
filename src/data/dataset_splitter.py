"""Reproducible, leakage-free train/val/test splits.

Frames from the same video/sequence/camera are near-duplicates, so splitting them independently leaks
test content into training. Images are therefore grouped (by a regex on the file name, by a CSV mapping,
or by a callable) and whole groups are assigned to splits. Exact duplicate files are also forced into the
same group. Assignment is deterministic for a given seed.
"""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from src.data.annotation_converter import IMG_EXTS
from src.data.dataset_validator import find_duplicates

# Default grouping: "<group>__<frame>" (our MOT converter), else "<group>_<digits>" (common video-frame
# naming), else the whole stem (each image is its own group).
DEFAULT_GROUP_PATTERNS = (r"^(?P<g>.+?)__", r"^(?P<g>.+?)[_-](?:frame[_-]?)?\d{3,}$")


def group_key(stem: str, patterns: tuple[str, ...] = DEFAULT_GROUP_PATTERNS) -> str:
    for pat in patterns:
        m = re.match(pat, stem)
        if m:
            return m.group("g")
    return stem


def split_groups(group_sizes: dict[str, int], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    """Greedy, seeded assignment of groups to splits so image counts approach the target ratios."""
    if abs(sum(ratios) - 1.0) > 1e-6 or any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative and sum to 1, got {ratios}")
    names = ("train", "val", "test")
    groups = sorted(group_sizes)
    rng = random.Random(seed)
    rng.shuffle(groups)
    # Big groups first so they do not unbalance small splits at the end (stable for equal sizes).
    groups.sort(key=lambda g: -group_sizes[g])
    total = sum(group_sizes.values())
    target = {n: r * total for n, r in zip(names, ratios)}
    active = [n for n in names if target[n] > 0]
    filled = {n: 0 for n in names}
    out: dict[str, str] = {}
    for g in groups:
        split = max(active, key=lambda n: (target[n] - filled[n]) / target[n])
        out[g] = split
        filled[split] += group_sizes[g]
    # Make sure every requested split is non-empty when there are enough groups: move the smallest group
    # from the most over-filled split.
    for n in active:
        if filled[n] == 0:
            donors = [d for d in active if sum(1 for v in out.values() if v == d) > 1]
            if not donors:
                break
            donor = max(donors, key=lambda d: filled[d] - target[d])
            g = min((g for g, v in out.items() if v == donor), key=lambda g: group_sizes[g])
            out[g] = n
            filled[donor] -= group_sizes[g]
            filled[n] += group_sizes[g]
    return out


def split_dataset(images_dir: str | Path, out_dir: str | Path, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                  seed: int = 42, group_fn: Callable[[str], str] | None = None, group_csv: str | Path | None = None,
                  dedupe: bool = True) -> dict:
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    mapping: dict[str, str] = {}
    if group_csv:
        with Path(group_csv).open() as f:
            mapping = {r["image_stem"]: r["group"] for r in csv.DictReader(f)}
    fn = group_fn or (lambda s: mapping.get(s) or group_key(s))
    groups = {p: fn(p.stem) for p in images}
    if dedupe:
        # Union duplicate files into one group so identical images cannot straddle splits.
        for dup in find_duplicates(images):
            key = groups[Path(dup[0])]
            old = {groups[Path(d)] for d in dup}
            for p, g in groups.items():
                if g in old:
                    groups[p] = key
    sizes: dict[str, int] = defaultdict(int)
    for g in groups.values():
        sizes[g] += 1
    assignment = split_groups(dict(sizes), ratios, seed)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for p in images:
        splits[assignment[groups[p]]].append(str(p.resolve()))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        (out_dir / f"{name}.txt").write_text("".join(f"{_portable(x, out_dir)}\n" for x in items))
    leaks = check_group_leakage(splits, {str(p.resolve()): g for p, g in groups.items()})
    return {
        "seed": seed, "ratios": ratios, "groups": len(sizes),
        "counts": {k: len(v) for k, v in splits.items()},
        "group_counts": {k: sum(1 for v in assignment.values() if v == k) for k in splits},
        "leaking_groups": leaks,
        "files": {k: f"{k}.txt" for k in splits},
    }


def check_group_leakage(splits: dict[str, list[str]], group_of: dict[str, str]) -> list[str]:
    seen: dict[str, set[str]] = defaultdict(set)
    for name, items in splits.items():
        for x in items:
            seen[group_of[x]].add(name)
    return sorted(g for g, s in seen.items() if len(s) > 1)


def _portable(path: str, base: Path) -> str:
    """Paths inside ``base`` are written as ``./relative`` (Ultralytics resolves these against the list file)."""
    try:
        return "./" + Path(path).relative_to(base.resolve()).as_posix()
    except ValueError:
        return path


def read_split(list_file: str | Path) -> list[str]:
    """Read a split file, resolving ``./relative`` entries against the file's directory."""
    list_file = Path(list_file)
    out = []
    for line in list_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(str((list_file.parent / line).resolve()) if line.startswith("./") else line)
    return out
