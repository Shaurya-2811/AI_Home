"""Provenance registry for datasets used in this project, plus a researched catalogue of candidates.

Every dataset that is downloaded or imported is recorded in ``data/registry.json`` with its source URL,
version, license, download date and annotation format, so training runs can cite exactly what they used.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import PROJECT_ROOT

REGISTRY_PATH = PROJECT_ROOT / "data" / "registry.json"

# Candidate datasets. Licenses are from the ORIGINAL publishers. Kaggle mirrors are usually community
# re-uploads: their slugs change, and a mirror cannot grant rights beyond the original license. Always
# confirm with `python -m src.data.kaggle_client search "<name>"` and read the dataset page before use.
CANDIDATE_DATASETS: list[dict] = [
    {
        "name": "CrowdHuman", "purpose": "crowded-scene person detection (full/visible/head boxes)",
        "official_url": "https://www.crowdhuman.org/", "annotation_format": "crowdhuman_odgt",
        "license": "Non-commercial research use only (CrowdHuman terms)", "modality": "RGB",
        "conditions": ["crowded", "occlusion", "outdoor", "indoor"],
        "kaggle_search": "crowdhuman",
    },
    {
        "name": "WiderPerson", "purpose": "dense pedestrian detection in the wild",
        "official_url": "http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/", "annotation_format": "widerperson_txt",
        "license": "Academic research only (per authors)", "modality": "RGB",
        "conditions": ["crowded", "occlusion", "outdoor"], "kaggle_search": "widerperson",
    },
    {
        "name": "MOT17", "purpose": "pedestrian multi-object tracking benchmark",
        "official_url": "https://motchallenge.net/data/MOT17/", "annotation_format": "mot_gt",
        "license": "CC BY-NC-SA 3.0", "modality": "RGB",
        "conditions": ["crowded", "daylight", "indoor", "moving camera"],
        "kaggle_search": "MOT17",
    },
    {
        "name": "MOT20", "purpose": "very crowded tracking (incl. night scenes)",
        "official_url": "https://motchallenge.net/data/MOT20/", "annotation_format": "mot_gt",
        "license": "CC BY-NC-SA 3.0", "modality": "RGB",
        "conditions": ["crowded", "low_light", "occlusion"], "kaggle_search": "MOT20",
    },
    {
        "name": "BDD100K", "purpose": "driving scenes with weather/time-of-day tags (pedestrian class)",
        "official_url": "https://bdd-data.berkeley.edu/", "annotation_format": "bdd_json",
        "license": "BDD100K license (non-commercial research); verify current terms", "modality": "RGB",
        "conditions": ["daylight", "night", "rain", "fog", "snow", "dawn/dusk"],
        "kaggle_search": "bdd100k",
        "notes": "Per-image weather and timeofday attributes make it the best source for per-condition evaluation.",
    },
    {
        "name": "ExDark", "purpose": "low-light object detection (People class among 12)",
        "official_url": "https://github.com/cs-chan/Exclusively-Dark-Image-Dataset", "annotation_format": "exdark_txt",
        "license": "BSD-3-Clause (repository); research use, cite the paper", "modality": "RGB",
        "conditions": ["low_light", "night", "indoor", "outdoor"], "kaggle_search": "exdark",
    },
    {
        "name": "LLVIP", "purpose": "paired visible/infrared pedestrian detection at night",
        "official_url": "https://github.com/bupt-ai-cz/LLVIP", "annotation_format": "voc_xml",
        "license": "Research use; see repository terms", "modality": "RGB + thermal IR (paired)",
        "conditions": ["low_light", "night", "thermal"], "kaggle_search": "LLVIP",
        "notes": "Thermal images are a separate modality: train/evaluate them separately from RGB.",
    },
    {
        "name": "Market-1501", "purpose": "person re-identification across 6 cameras",
        "official_url": "https://zheng-lab-anu.github.io/Project/project_reid.html", "annotation_format": "reid_market",
        "license": "Research use only", "modality": "RGB crops", "conditions": ["daylight", "outdoor"],
        "kaggle_search": "market-1501",
    },
]


class DatasetRegistry:
    def __init__(self, path: str | Path = REGISTRY_PATH):
        self.path = Path(path)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text())

    def _save(self, entries: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2))

    def all(self) -> list[dict]:
        return self._load()

    def get(self, name: str) -> dict | None:
        return next((e for e in self._load() if e["name"] == name or e.get("slug") == name), None)

    def register(self, name: str, source_url: str, license: str, annotation_format: str, local_path: str,
                 slug: str | None = None, version: str = "unknown", download_date: str | None = None,
                 modality: str = "RGB", conditions: list[str] | None = None, notes: str = "") -> dict:
        entries = [e for e in self._load() if e["name"] != name]
        entry = {
            "name": name, "slug": slug, "source_url": source_url, "version": version, "license": license,
            "download_date": download_date or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "annotation_format": annotation_format, "local_path": local_path, "modality": modality,
            "conditions": conditions or [], "notes": notes,
        }
        entries.append(entry)
        self._save(entries)
        return entry
