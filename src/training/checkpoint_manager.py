"""Locate, record and promote training checkpoints."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import PROJECT_ROOT


class CheckpointManager:
    def __init__(self, project: str | Path = PROJECT_ROOT / "models" / "runs"):
        self.project = Path(project)

    def runs(self) -> list[Path]:
        if not self.project.is_dir():
            return []
        return sorted((p for p in self.project.iterdir() if (p / "weights").is_dir()), key=lambda p: p.stat().st_mtime)

    def latest_run(self, name: str | None = None) -> Path | None:
        runs = [r for r in self.runs() if name is None or r.name.startswith(name)]
        return runs[-1] if runs else None

    @staticmethod
    def last(run_dir: str | Path) -> Path | None:
        p = Path(run_dir) / "weights" / "last.pt"
        return p if p.exists() else None

    @staticmethod
    def best(run_dir: str | Path) -> Path | None:
        p = Path(run_dir) / "weights" / "best.pt"
        return p if p.exists() else None

    def resume_path(self, resume: bool | str, name: str | None = None) -> Path | None:
        """Resolve ``resume`` (True -> latest run's last.pt, or an explicit path)."""
        if not resume:
            return None
        if isinstance(resume, str) and resume.lower() not in ("true", "yes"):
            p = Path(resume)
            if not p.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {p}")
            return p
        run = self.latest_run(name)
        ckpt = self.last(run) if run else None
        if ckpt is None:
            raise FileNotFoundError(f"No last.pt found to resume under {self.project}")
        return ckpt

    @staticmethod
    def write_metadata(run_dir: str | Path, info: dict) -> Path:
        path = Path(run_dir) / "run_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info}
        path.write_text(json.dumps(info, indent=2, default=str))
        return path

    @staticmethod
    def promote(best: str | Path, target: str | Path = PROJECT_ROOT / "models" / "person_finetuned.pt") -> Path:
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)
        return target
