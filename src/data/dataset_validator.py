"""Validate a YOLO-format person dataset and produce a statistics report.

Checks: unreadable/corrupt images, missing or orphan labels, malformed label lines, out-of-range or
degenerate boxes, unknown classes, duplicate image files (content hash). Statistics: class counts, image
resolutions, box sizes (COCO small/medium/large), people per image, and environmental conditions when a
``conditions.csv`` (``image_stem,condition``) is supplied. Without it, a brightness-based *estimate* of
low-light images is reported and clearly labelled as a heuristic.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.data.annotation_converter import IMG_EXTS


@dataclass
class ValidationReport:
    images: int = 0
    labels: int = 0
    boxes: int = 0
    background_images: int = 0
    corrupt_images: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    invalid_lines: list[str] = field(default_factory=list)
    duplicate_groups: list[list[str]] = field(default_factory=list)
    class_counts: dict = field(default_factory=dict)
    resolutions: dict = field(default_factory=dict)
    box_size_buckets: dict = field(default_factory=dict)
    people_per_image: dict = field(default_factory=dict)
    conditions: dict = field(default_factory=dict)
    conditions_source: str = "none"

    @property
    def ok(self) -> bool:
        return not (self.corrupt_images or self.invalid_lines)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["ok"] = self.ok
        return d

    def to_markdown(self) -> str:
        lines = [
            "# Dataset validation report", "",
            f"- Images: {self.images} (background / no people: {self.background_images})",
            f"- Label files: {self.labels}, boxes: {self.boxes}",
            f"- Corrupt images: {len(self.corrupt_images)}",
            f"- Missing labels: {len(self.missing_labels)}; orphan labels: {len(self.orphan_labels)}",
            f"- Invalid label lines: {len(self.invalid_lines)}",
            f"- Duplicate image groups: {len(self.duplicate_groups)}",
            f"- Class counts: {self.class_counts}",
            f"- Box sizes (COCO buckets, pixels): {self.box_size_buckets}",
            f"- People per image: {self.people_per_image}",
            f"- Top resolutions: {dict(list(self.resolutions.items())[:5])}",
            f"- Conditions ({self.conditions_source}): {self.conditions}",
            "", f"**Status: {'OK' if self.ok else 'PROBLEMS FOUND'}**",
        ]
        for title, items in (("Corrupt images", self.corrupt_images), ("Invalid lines", self.invalid_lines)):
            if items:
                lines += ["", f"## {title} (first 20)", *[f"- {x}" for x in items[:20]]]
        return "\n".join(lines) + "\n"


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def find_duplicates(paths: list[Path]) -> list[list[str]]:
    by_size: dict[int, list[Path]] = defaultdict(list)
    for p in paths:
        by_size[p.stat().st_size].append(p)
    groups: dict[str, list[str]] = defaultdict(list)
    for same in by_size.values():
        if len(same) > 1:
            for p in same:
                groups[file_hash(p)].append(str(p))
    return [g for g in groups.values() if len(g) > 1]


def _parse_label(path: Path, num_classes: int, errors: list[str]) -> list[tuple[int, float, float, float, float]]:
    out = []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        where = f"{path.name}:{ln}"
        if len(parts) != 5:
            errors.append(f"{where}: expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{where}: non-numeric value")
            continue
        if not 0 <= cls < num_classes:
            errors.append(f"{where}: class {cls} outside [0, {num_classes - 1}]")
            continue
        if not all(np.isfinite([cx, cy, w, h])) or w <= 0 or h <= 0:
            errors.append(f"{where}: non-positive or non-finite box size")
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and w <= 1.0001 and h <= 1.0001):
            errors.append(f"{where}: coordinates outside [0, 1]")
            continue
        if cx - w / 2 < -0.01 or cx + w / 2 > 1.01 or cy - h / 2 < -0.01 or cy + h / 2 > 1.01:
            errors.append(f"{where}: box extends outside the image")
            continue
        out.append((cls, cx, cy, w, h))
    return out


def load_conditions(path: str | Path | None) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    with Path(path).open() as f:
        return {row["image_stem"]: row["condition"] for row in csv.DictReader(f)}


def validate_yolo_dataset(images_dir: str | Path, labels_dir: str | Path, num_classes: int = 1,
                          conditions_csv: str | Path | None = None, check_duplicates: bool = True,
                          low_light_luma: float = 50.0) -> ValidationReport:
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    labels = {p.stem: p for p in labels_dir.rglob("*.txt")} if labels_dir.is_dir() else {}
    rep = ValidationReport(images=len(images), labels=len(labels))
    cond_map = load_conditions(conditions_csv)
    rep.conditions_source = "conditions.csv" if cond_map else "heuristic: mean brightness (estimate only)"
    classes: Counter = Counter()
    res: Counter = Counter()
    sizes: Counter = Counter()
    ppl: Counter = Counter()
    conds: Counter = Counter()
    stems = set()
    for img_path in images:
        stems.add(img_path.stem)
        img = cv2.imread(str(img_path))
        if img is None:
            rep.corrupt_images.append(str(img_path))
            continue
        h, w = img.shape[:2]
        res[f"{w}x{h}"] += 1
        if cond_map:
            conds[cond_map.get(img_path.stem, "unlabelled")] += 1
        else:
            luma = float(cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2GRAY).mean())
            conds["low_light_estimate" if luma < low_light_luma else "normal_light_estimate"] += 1
        lp = labels.get(img_path.stem)
        if lp is None:
            rep.missing_labels.append(str(img_path))
            continue
        boxes = _parse_label(lp, num_classes, rep.invalid_lines)
        if not boxes:
            rep.background_images += 1
        ppl["0" if not boxes else "1" if len(boxes) == 1 else "2-5" if len(boxes) <= 5 else "6-20" if len(boxes) <= 20 else ">20"] += 1
        for cls, _, _, bw, bh in boxes:
            classes[cls] += 1
            area = bw * w * bh * h
            sizes["small(<32^2)" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large(>96^2)"] += 1
        rep.boxes += len(boxes)
    rep.orphan_labels = sorted(str(p) for s, p in labels.items() if s not in stems)
    rep.class_counts = {str(k): v for k, v in sorted(classes.items())}
    rep.resolutions = dict(res.most_common())
    rep.box_size_buckets = dict(sizes)
    rep.people_per_image = dict(ppl)
    rep.conditions = dict(conds)
    if check_duplicates:
        rep.duplicate_groups = find_duplicates(images)
    return rep


def write_report(rep: ValidationReport, out_dir: str | Path, name: str = "dataset_report") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    j = out_dir / f"{name}.json"
    m = out_dir / f"{name}.md"
    j.write_text(json.dumps(rep.to_dict(), indent=2))
    m.write_text(rep.to_markdown())
    return j, m
