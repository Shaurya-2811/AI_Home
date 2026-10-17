"""Convert person annotations from common dataset formats into YOLO detection labels (single class: person).

YOLO label line: ``<class> <cx> <cy> <w> <h>`` normalised to [0, 1]. Every converter writes one ``.txt`` per
image (empty file = background image with no people, which YOLO uses as negatives; keep such images to
reduce false positives on mannequins/posters if your data has them).

Supported inputs: COCO JSON, Pascal VOC XML, CrowdHuman ``.odgt``, MOTChallenge ``gt.txt``,
WiderPerson ``.txt`` and ExDark ``.txt``.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

PERSON_NAMES = {"person", "people", "pedestrian", "human", "rider", "persons"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class AnnotationError(ValueError):
    """Malformed annotation. The message names the file (and line/record) that failed."""


@dataclass
class ConversionReport:
    images: int = 0
    boxes: int = 0
    skipped_boxes: int = 0
    images_without_file: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[float, float, float, float] | None:
    """Clip to the image and normalise; returns None for degenerate boxes."""
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    return (x1 + bw / 2) / w, (y1 + bh / 2) / h, bw / w, bh / h


def write_yolo_label(path: Path, boxes: Iterable[tuple[float, float, float, float]], cls: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for cx, cy, bw, bh in boxes))


def _image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise AnnotationError(f"Cannot read image {path}")
    return img.shape[1], img.shape[0]


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        for cand in (images_dir / f"{stem}{ext}", images_dir / f"{stem}{ext.upper()}"):
            if cand.exists():
                return cand
    return None


# ---------------------------------------------------------------------------- COCO
def coco_to_yolo(json_path: str | Path, out_labels: str | Path, person_names: set[str] = PERSON_NAMES,
                 include_crowd: bool = False) -> ConversionReport:
    json_path, out_labels = Path(json_path), Path(out_labels)
    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"{json_path}: invalid JSON ({exc})") from exc
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise AnnotationError(f"{json_path}: missing '{key}' (not a COCO detection file?)")
    cat_ids = {c["id"] for c in data["categories"] if str(c.get("name", "")).lower() in person_names}
    if not cat_ids:
        raise AnnotationError(f"{json_path}: no person-like category among {[c.get('name') for c in data['categories']]}")
    images = {im["id"]: im for im in data["images"]}
    per_image: dict[int, list] = {i: [] for i in images}
    rep = ConversionReport(images=len(images))
    for k, ann in enumerate(data["annotations"]):
        if ann.get("category_id") not in cat_ids:
            continue
        if ann.get("iscrowd") and not include_crowd:
            rep.skipped_boxes += 1
            continue
        im = images.get(ann.get("image_id"))
        bbox = ann.get("bbox")
        if im is None or not isinstance(bbox, list) or len(bbox) != 4:
            raise AnnotationError(f"{json_path}: annotation #{k} has a bad image_id or bbox: {ann!r:.120}")
        x, y, bw, bh = (float(v) for v in bbox)
        box = xyxy_to_yolo(x, y, x + bw, y + bh, im["width"], im["height"])
        if box is None:
            rep.skipped_boxes += 1
            continue
        per_image[im["id"]].append(box)
        rep.boxes += 1
    for iid, boxes in per_image.items():
        write_yolo_label(out_labels / (Path(images[iid]["file_name"]).stem + ".txt"), boxes)
    return rep


# ----------------------------------------------------------------------------- VOC
def voc_to_yolo(xml_dir: str | Path, out_labels: str | Path, person_names: set[str] = PERSON_NAMES) -> ConversionReport:
    rep = ConversionReport()
    for xml_path in sorted(Path(xml_dir).glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
            size = root.find("size")
            w, h = int(size.find("width").text), int(size.find("height").text)
        except (ET.ParseError, AttributeError, TypeError, ValueError) as exc:
            raise AnnotationError(f"{xml_path}: malformed VOC XML ({exc})") from exc
        boxes = []
        for obj in root.findall("object"):
            if (obj.findtext("name") or "").strip().lower() not in person_names:
                continue
            bb = obj.find("bndbox")
            try:
                coords = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
            except (AttributeError, TypeError, ValueError) as exc:
                raise AnnotationError(f"{xml_path}: bad bndbox in object '{obj.findtext('name')}'") from exc
            box = xyxy_to_yolo(*coords, w, h)
            if box is None:
                rep.skipped_boxes += 1
            else:
                boxes.append(box)
        write_yolo_label(Path(out_labels) / f"{xml_path.stem}.txt", boxes)
        rep.images += 1
        rep.boxes += len(boxes)
    return rep


# ---------------------------------------------------------------------- CrowdHuman
def crowdhuman_to_yolo(odgt_path: str | Path, images_dir: str | Path, out_labels: str | Path,
                       box_type: str = "fbox") -> ConversionReport:
    """``box_type``: ``fbox`` (full body incl. occluded part), ``vbox`` (visible part) or ``hbox`` (head)."""
    if box_type not in ("fbox", "vbox", "hbox"):
        raise ValueError("box_type must be fbox, vbox or hbox")
    rep = ConversionReport()
    images_dir = Path(images_dir)
    for ln, line in enumerate(Path(odgt_path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            stem = rec["ID"]
            gtboxes = rec["gtboxes"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise AnnotationError(f"{odgt_path}:{ln}: malformed record ({exc})") from exc
        img = _find_image(images_dir, stem)
        if img is None:
            rep.images_without_file += 1
            continue
        w, h = _image_size(img)
        boxes = []
        for g in gtboxes:
            if g.get("tag") != "person" or g.get("extra", {}).get("ignore", 0) == 1 or box_type not in g:
                rep.skipped_boxes += 1
                continue
            x, y, bw, bh = g[box_type]
            box = xyxy_to_yolo(x, y, x + bw, y + bh, w, h)
            if box is None:
                rep.skipped_boxes += 1
            else:
                boxes.append(box)
        write_yolo_label(Path(out_labels) / f"{stem}.txt", boxes)
        rep.images += 1
        rep.boxes += len(boxes)
    return rep


# ---------------------------------------------------------------------------- MOT
def read_mot_gt(gt_path: str | Path, pedestrian_classes: tuple[int, ...] = (1,), min_visibility: float = 0.0) -> dict[int, list[tuple]]:
    """Parse MOTChallenge gt.txt -> {frame: [(track_id, x1, y1, x2, y2), ...]} (1-based frames)."""
    out: dict[int, list[tuple]] = {}
    for ln, line in enumerate(Path(gt_path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) < 6:
            raise AnnotationError(f"{gt_path}:{ln}: expected >=6 comma-separated values, got {len(parts)}")
        try:
            frame, tid = int(float(parts[0])), int(float(parts[1]))
            x, y, w, h = (float(v) for v in parts[2:6])
            flag = float(parts[6]) if len(parts) > 6 else 1.0
            cls = int(float(parts[7])) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0
        except ValueError as exc:
            raise AnnotationError(f"{gt_path}:{ln}: non-numeric value ({exc})") from exc
        if flag == 0 or cls not in pedestrian_classes or vis < min_visibility:
            continue
        out.setdefault(frame, []).append((tid, x, y, x + w, y + h))
    return out


def mot_to_yolo(sequence_dir: str | Path, out_images: str | Path, out_labels: str | Path, every_n: int = 1,
                min_visibility: float = 0.0) -> ConversionReport:
    """Convert one MOT sequence (``img1/`` + ``gt/gt.txt``). File names are prefixed with the sequence name,
    which ``dataset_splitter`` uses as the group key so frames of one sequence never span splits."""
    seq = Path(sequence_dir)
    gt = read_mot_gt(seq / "gt" / "gt.txt", min_visibility=min_visibility)
    rep = ConversionReport()
    out_images, out_labels = Path(out_images), Path(out_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    frames = sorted((seq / "img1").glob("*.jpg"))
    for img_path in frames[::every_n]:
        fidx = int(img_path.stem)
        w, h = _image_size(img_path)
        boxes = [b for b in (xyxy_to_yolo(x1, y1, x2, y2, w, h) for _, x1, y1, x2, y2 in gt.get(fidx, [])) if b]
        name = f"{seq.name}__{img_path.stem}"
        target = out_images / f"{name}.jpg"
        if not target.exists():
            try:
                target.symlink_to(img_path.resolve())
            except OSError:
                import shutil

                shutil.copy2(img_path, target)
        write_yolo_label(out_labels / f"{name}.txt", boxes)
        rep.images += 1
        rep.boxes += len(boxes)
    return rep


# --------------------------------------------------------------------- WiderPerson
def widerperson_to_yolo(ann_dir: str | Path, images_dir: str | Path, out_labels: str | Path,
                        keep_classes: tuple[int, ...] = (1, 2, 3)) -> ConversionReport:
    """WiderPerson classes: 1 pedestrian, 2 rider, 3 partially-visible, 4 ignore region, 5 crowd."""
    rep = ConversionReport()
    for ann in sorted(Path(ann_dir).glob("*.txt")):
        stem = Path(ann.stem).stem  # files are named like 000040.jpg.txt
        img = _find_image(Path(images_dir), stem)
        if img is None:
            rep.images_without_file += 1
            continue
        w, h = _image_size(img)
        lines = [l for l in ann.read_text().splitlines() if l.strip()]
        boxes = []
        for ln, line in enumerate(lines[1:], 2):
            parts = line.split()
            if len(parts) != 5:
                raise AnnotationError(f"{ann}:{ln}: expected 5 values, got {len(parts)}")
            cls, x1, y1, x2, y2 = int(parts[0]), *(float(v) for v in parts[1:])
            if cls not in keep_classes:
                rep.skipped_boxes += 1
                continue
            box = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            if box:
                boxes.append(box)
        write_yolo_label(Path(out_labels) / f"{stem}.txt", boxes)
        rep.images += 1
        rep.boxes += len(boxes)
    return rep


# --------------------------------------------------------------------------- ExDark
def exdark_to_yolo(ann_dir: str | Path, images_dir: str | Path, out_labels: str | Path) -> ConversionReport:
    """ExDark bbGt format: header line, then ``<class> l t w h ...`` per object; keep class 'People'."""
    rep = ConversionReport()
    for ann in sorted(Path(ann_dir).glob("*.txt")):
        stem = Path(ann.stem).stem
        img = _find_image(Path(images_dir), stem)
        if img is None:
            rep.images_without_file += 1
            continue
        w, h = _image_size(img)
        boxes = []
        for ln, line in enumerate(ann.read_text().splitlines(), 1):
            if ln == 1 or not line.strip() or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 5:
                raise AnnotationError(f"{ann}:{ln}: expected at least 5 values")
            if parts[0].lower() != "people":
                continue
            l, t, bw, bh = (float(v) for v in parts[1:5])
            box = xyxy_to_yolo(l, t, l + bw, t + bh, w, h)
            if box:
                boxes.append(box)
        write_yolo_label(Path(out_labels) / f"{stem}.txt", boxes)
        rep.images += 1
        rep.boxes += len(boxes)
    return rep


def write_data_yaml(path: str | Path, root: str | Path, train: str, val: str, test: str | None = None) -> Path:
    import yaml

    path = Path(path)
    # ``path`` is resolved at write time; re-run `datasets split` (or edit this line) after moving the dataset.
    data = {"path": str(Path(root).resolve()), "train": train, "val": val, "names": {0: "person"}}
    if test:
        data["test"] = test
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


CONVERTERS = {
    "coco": coco_to_yolo,
    "voc": voc_to_yolo,
    "crowdhuman": crowdhuman_to_yolo,
    "mot": mot_to_yolo,
    "widerperson": widerperson_to_yolo,
    "exdark": exdark_to_yolo,
}
