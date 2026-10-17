"""Tiny public sample data for smoke tests and benchmarks (no Kaggle account needed).

``coco128`` (Ultralytics, ~7 MB): the first 128 COCO train2017 images with YOLO labels. We keep only the
person class. It is far too small for meaningful fine-tuning or per-condition evaluation, but it gives
real, human-made person annotations for exercising the detection metrics, corruption benchmark,
enhancement A/B and the training pipeline end to end. COCO annotations are CC BY 4.0; images are from
Flickr under their individual licenses (see https://cocodataset.org/#termsofuse).
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from src.utils.config import PROJECT_ROOT

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def prepare_coco128_person(root: str | Path = PROJECT_ROOT / "data", seed: int = 42) -> Path:
    """Download (once), keep person labels only, split into train/val/test. Returns the data.yaml path."""
    from src.data.annotation_converter import write_data_yaml
    from src.data.dataset_registry import DatasetRegistry
    from src.data.dataset_splitter import split_dataset

    root = Path(root)
    raw = root / "raw" / "coco128"
    out = root / "processed" / "coco128_person"
    data_yaml = out / "data.yaml"
    if data_yaml.exists():
        return data_yaml
    if not (raw / "images").exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        zpath = raw.parent / "coco128.zip"
        if not zpath.exists():
            import urllib.request

            urllib.request.urlretrieve(COCO128_URL, zpath)  # noqa: S310 - fixed https URL
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw.parent)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for img in sorted((raw / "images" / "train2017").glob("*.jpg")):
        shutil.copy2(img, out / "images" / img.name)
        lbl = raw / "labels" / "train2017" / f"{img.stem}.txt"
        lines = []
        if lbl.exists():
            lines = [l for l in lbl.read_text().splitlines() if l.split() and l.split()[0] == "0"]
        (out / "labels" / f"{img.stem}.txt").write_text("".join(f"{l}\n" for l in lines))
    res = split_dataset(out / "images", out, (0.7, 0.15, 0.15), seed)
    write_data_yaml(data_yaml, out, res["files"]["train"], res["files"]["val"], res["files"]["test"])
    DatasetRegistry().register(
        name="coco128-person", source_url=COCO128_URL, license="COCO: annotations CC BY 4.0, images Flickr (various)",
        annotation_format="yolo", local_path=str(out), version="ultralytics v0.0.0",
        notes="Person-only subset of the first 128 COCO train2017 images. For smoke tests/benchmarks only.",
    )
    return data_yaml
