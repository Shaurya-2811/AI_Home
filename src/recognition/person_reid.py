"""Optional person re-identification by body appearance.

Two different things are kept apart on purpose:

* **Instance recognition** (same person later in this video) is the tracker's job: ``track_id``.
* **Re-identification** here assigns *visual identity* labels (``V-0001`` ...) to tracks whose aggregated
  appearance embedding is similar to a gallery entry. A label means "looks similar above a threshold",
  never a verified real-world identity. Tracks without enough reliable samples stay ``unknown``.

The gallery can be saved and loaded (``.npz``) to compare across videos or cameras.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.recognition.embeddings import EmbeddingExtractor, l2_normalize

log = logging.getLogger(__name__)

UNKNOWN = "unknown"


@dataclass
class VisualIdentity:
    label: str
    embedding: np.ndarray
    observations: int = 1
    sources: set[str] = field(default_factory=set)


class ReIDGallery:
    def __init__(self, similarity_threshold: float = 0.8, ema: float = 0.9):
        self.threshold = similarity_threshold
        self.ema = ema
        self.identities: list[VisualIdentity] = []
        self._next = 1

    def __len__(self) -> int:
        return len(self.identities)

    def query(self, emb: np.ndarray, exclude: set[str] | None = None) -> tuple[str | None, float]:
        """Best matching label and cosine similarity (None if the gallery is empty)."""
        cands = [i for i in self.identities if not exclude or i.label not in exclude]
        if not cands:
            return None, -1.0
        sims = np.stack([c.embedding for c in cands]) @ emb
        j = int(np.argmax(sims))
        return cands[j].label, float(sims[j])

    def assign(self, emb: np.ndarray, source: str = "", exclude: set[str] | None = None) -> tuple[str, float]:
        """Match ``emb`` against the gallery; reuse the identity above threshold, else enrol a new one."""
        emb = l2_normalize(emb.astype(np.float32))
        label, sim = self.query(emb, exclude)
        if label is not None and sim >= self.threshold:
            ident = next(i for i in self.identities if i.label == label)
            ident.embedding = l2_normalize(self.ema * ident.embedding + (1 - self.ema) * emb)
            ident.observations += 1
            ident.sources.add(source)
            return label, sim
        new = VisualIdentity(label=f"V-{self._next:04d}", embedding=emb, sources={source})
        self._next += 1
        self.identities.append(new)
        return new.label, sim if label is not None else 1.0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            labels=np.array([i.label for i in self.identities]),
            embeddings=np.stack([i.embedding for i in self.identities]) if self.identities else np.zeros((0, 0)),
            observations=np.array([i.observations for i in self.identities]),
            threshold=np.array(self.threshold),
            next_id=np.array(self._next),
        )
        return path

    @classmethod
    def load(cls, path: str | Path, similarity_threshold: float | None = None) -> "ReIDGallery":
        data = np.load(path, allow_pickle=False)
        g = cls(float(similarity_threshold if similarity_threshold is not None else data["threshold"]))
        for label, emb, obs in zip(data["labels"], data["embeddings"], data["observations"]):
            g.identities.append(VisualIdentity(str(label), emb.astype(np.float32), int(obs)))
        g._next = int(data["next_id"])
        return g


class TrackReID:
    """Collects embeddings per track and assigns gallery labels once a track has enough samples."""

    def __init__(self, embedder: EmbeddingExtractor, gallery: ReIDGallery, min_track_frames: int = 5,
                 sample_every: int = 5, min_crop_height: int = 48, max_samples: int = 20, source: str = ""):
        self.embedder = embedder
        self.gallery = gallery
        self.min_track_frames = min_track_frames
        self.sample_every = max(1, sample_every)
        self.min_crop_height = min_crop_height
        self.max_samples = max_samples
        self.source = source
        self.samples: dict[int, list[np.ndarray]] = {}
        self.seen: dict[int, int] = {}
        self.labels: dict[int, tuple[str, float | None]] = {}

    @classmethod
    def from_config(cls, cfg: dict, embedder: EmbeddingExtractor, source: str = "") -> "TrackReID":
        r = cfg["reid"]
        gp = r.get("gallery_path")
        gallery = ReIDGallery.load(gp, r["similarity_threshold"]) if gp and Path(gp).exists() else ReIDGallery(r["similarity_threshold"])
        return cls(embedder, gallery, r.get("min_track_frames", 5), r.get("sample_every", 5), r.get("min_crop_height", 48),
                   r.get("max_samples_per_track", 20), source)

    def update(self, image: np.ndarray, tracks: list, det_feats: np.ndarray | None = None) -> dict[int, tuple[str, float | None]]:
        """Sample embeddings for active tracks; returns the current label map for every track seen so far."""
        need = []
        for tr in tracks:
            n = self.seen.get(tr.track_id, 0)
            self.seen[tr.track_id] = n + 1
            h = tr.xyxy[3] - tr.xyxy[1]
            if n % self.sample_every == 0 and h >= self.min_crop_height and len(self.samples.get(tr.track_id, [])) < self.max_samples:
                need.append(tr)
        if need:
            if det_feats is not None and all(0 <= t.det_index < len(det_feats) for t in need):
                embs = det_feats[[t.det_index for t in need]]
            else:
                embs = self.embedder.embed_boxes(image, np.stack([t.xyxy for t in need]))
            for tr, e in zip(need, embs):
                if np.linalg.norm(e) > 0:
                    self.samples.setdefault(tr.track_id, []).append(e)
        active_labels = {self.labels[t.track_id][0] for t in tracks if t.track_id in self.labels}
        for tr in tracks:
            tid = tr.track_id
            s = self.samples.get(tid, [])
            if tid not in self.labels and s and self.seen[tid] >= self.min_track_frames:
                # Exclude identities currently held by other visible tracks: one person cannot be in two places.
                label, sim = self.gallery.assign(np.mean(s, axis=0), self.source, exclude=active_labels)
                self.labels[tid] = (label, sim)
                active_labels.add(label)
        return {t.track_id: self.labels.get(t.track_id, (UNKNOWN, None)) for t in tracks}
