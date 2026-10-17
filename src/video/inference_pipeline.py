"""End-to-end inference: video -> preprocessing -> detection -> tracking -> (ReID) -> analysis -> outputs.

Each stage is a separate component chosen by configuration. Stage failures are contained: a detector
error on one frame yields an empty detection set for that frame, a tracker error resets the tracker, and a
ReID error disables ReID for the rest of the run. All are recorded as warnings in the results.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import BaseDetector, Detections, build_detector
from src.detection.preprocessing import Preprocessor
from src.tracking.track_manager import TrackManager
from src.tracking.tracker import PersonTracker, TrackedObject
from src.utils.config import PROJECT_ROOT
from src.utils.device import peak_memory_mb, resolve_device
from src.utils.visualization import draw_header, draw_tracks
from src.video.video_reader import Frame, VideoReader
from src.video.video_writer import VideoWriter

log = logging.getLogger(__name__)

ProgressFn = Callable[[float | None, int, dict], None]
FrameFn = Callable[[np.ndarray, dict], None]


@dataclass
class PipelineResult:
    run_dir: Path
    video_path: Path | None
    csv_path: Path
    json_path: Path
    summary: dict
    warnings: list[str]
    timing: dict
    metadata: dict
    crops_dir: Path | None = None
    frames_dir: Path | None = None
    extra: dict = field(default_factory=dict)


class StageTimer:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, stage: str, seconds: float) -> None:
        self.totals[stage] = self.totals.get(stage, 0.0) + seconds
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def report(self) -> dict:
        return {k: round(1000 * v / max(1, self.counts[k]), 2) for k, v in self.totals.items()}


class InferencePipeline:
    def __init__(self, cfg: dict, detector: BaseDetector | None = None, embedder=None):
        self.cfg = cfg
        self.device = resolve_device(cfg.get("device", "auto"))
        self.detector = detector or build_detector(cfg, self.device)
        self.preprocessor = Preprocessor.from_config(cfg)
        self._embedder = embedder
        self.run_warnings: list[str] = []

    # ------------------------------------------------------------------ helpers
    def _get_embedder(self):
        if self._embedder is None:
            from src.recognition.embeddings import build_embedder

            self._embedder = build_embedder(self.cfg, self.device)
        return self._embedder

    def _detect(self, images: list[np.ndarray], failures: list[int]) -> list[Detections]:
        try:
            return self.detector.detect_batch(images)
        except Exception as exc:  # noqa: BLE001 - keep the run alive
            failures[0] += len(images)
            log.warning("Detector failed on %d frame(s): %s", len(images), exc)
            if failures[0] == len(images):
                self.run_warnings.append(f"Detector error (first occurrence): {exc}")
            return [Detections() for _ in images]

    # ---------------------------------------------------------------------- run
    def run(
        self,
        source: str | int | Path,
        output_dir: str | Path | None = None,
        run_name: str | None = None,
        progress: ProgressFn | None = None,
        on_frame: FrameFn | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> PipelineResult:
        cfg, vcfg, ocfg = self.cfg, self.cfg["video"], self.cfg["output"]
        out_root = Path(output_dir or PROJECT_ROOT / ocfg.get("dir", "outputs"))
        run_dir = out_root / (run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_warnings = []
        lim = cfg.get("limits", {})

        reader = VideoReader(
            source,
            frame_stride=vcfg.get("frame_stride", 1),
            target_fps=vcfg.get("target_fps"),
            max_frames=vcfg.get("max_frames"),
            resize_max_side=vcfg.get("resize_max_side"),
            max_mb=lim.get("max_upload_mb"),
            max_duration_s=lim.get("max_duration_s"),
            allowed_extensions=lim.get("allowed_extensions"),
        )
        meta = reader.metadata
        tracker = PersonTracker(cfg["tracker"], frame_stride=reader.stride)
        manager = TrackManager(frame_stride=reader.stride, quality_cfg=cfg.get("quality"))

        reid_on = bool(cfg["reid"].get("enabled"))
        need_feats = tracker.uses_appearance
        track_reid = None
        if reid_on or need_feats:
            try:
                emb = self._get_embedder()
                if reid_on:
                    from src.recognition.person_reid import TrackReID

                    track_reid = TrackReID.from_config(cfg, emb, source=str(Path(str(source)).name))
            except Exception as exc:  # noqa: BLE001
                self.run_warnings.append(f"ReID disabled: {exc}")
                reid_on, need_feats, track_reid = False, False, None
                if tracker.uses_appearance:
                    tracker.uses_appearance = False

        writer = None
        video_path = run_dir / "annotated.mp4"
        crops_dir = run_dir / "crops" if ocfg.get("save_crops") else None
        frames_dir = run_dir / "frames" if ocfg.get("save_frames") else None
        for d in (crops_dir, frames_dir):
            if d:
                d.mkdir(exist_ok=True)

        timer = StageTimer()
        det_failures = [0]
        processed = 0
        total = reader.expected_processed_frames
        batch_size = max(1, int(vcfg.get("batch_size", 1)))
        t_start = time.perf_counter()
        latencies: list[float] = []
        stopped_early = False

        def process_batch(batch: list[tuple[Frame, float]]) -> None:
            nonlocal processed, writer, reid_on, track_reid, need_feats
            t = time.perf_counter()
            det_inputs = []
            for fr, _ in batch:
                img, _m = self.preprocessor(fr.image)
                det_inputs.append(img)
            timer.add("preprocess", (time.perf_counter() - t) / len(batch))
            t = time.perf_counter()
            dets_list = self._detect(det_inputs, det_failures)
            timer.add("detect", (time.perf_counter() - t) / len(batch))

            for (fr, t_read), dets, det_img in zip(batch, dets_list, det_inputs):
                feats = None
                if need_feats and len(dets):
                    t = time.perf_counter()
                    try:
                        feats = self._get_embedder().embed_boxes(det_img, dets.xyxy)
                    except Exception as exc:  # noqa: BLE001
                        self.run_warnings.append(f"Appearance features disabled after error: {exc}")
                        need_feats, tracker.uses_appearance = False, False
                    timer.add("embed", time.perf_counter() - t)
                t = time.perf_counter()
                try:
                    tracks = tracker.update(dets, fr.image, feats)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Tracker failed at frame %d: %s; resetting tracker", fr.index, exc)
                    self.run_warnings.append(f"Tracker reset at frame {fr.index}: {exc}")
                    tracker.reset()
                    tracks = []
                timer.add("track", time.perf_counter() - t)

                labels = None
                if reid_on and track_reid is not None and tracks:
                    t = time.perf_counter()
                    try:
                        labels = track_reid.update(det_img, tracks, feats)
                    except Exception as exc:  # noqa: BLE001
                        self.run_warnings.append(f"ReID disabled after error: {exc}")
                        reid_on, track_reid = False, None
                    timer.add("reid", time.perf_counter() - t)

                # Exports use ORIGINAL video coordinates.
                inv = 1.0 / fr.scale
                export_tracks = [TrackedObject(tr.track_id, tr.xyxy * inv, tr.score, tr.cls, tr.det_index) for tr in tracks] if inv != 1.0 else tracks
                manager.update(fr.index, fr.timestamp_s, export_tracks, len(dets), labels)

                t = time.perf_counter()
                need_render = ocfg.get("save_video", True) or on_frame is not None or frames_dir is not None
                if need_render:
                    canvas = fr.image.copy()
                    trails = None
                    if ocfg.get("draw_trails", True):
                        trails = {k: [(x * fr.scale, y * fr.scale) for x, y in v]
                                  for k, v in manager.trails([t.track_id for t in tracks], ocfg.get("trail_length", 30)).items()}
                    lbl = {k: v[0] for k, v in labels.items() if v[0] != "unknown"} if labels else None
                    draw_tracks(canvas, tracks, trails, ocfg.get("draw_confidence", True), lbl)
                    elapsed = time.perf_counter() - t_start
                    draw_header(canvas, [
                        f"t={fr.timestamp_s:7.2f}s  frame {fr.index}",
                        f"active tracks: {len(tracks)}  track IDs so far: {len(manager.trajectories)}",
                        f"proc fps: {(processed + 1) / max(elapsed, 1e-6):.1f}",
                    ])
                    if ocfg.get("save_video", True):
                        if writer is None:
                            writer = VideoWriter(video_path, reader.processed_fps, (canvas.shape[1], canvas.shape[0]), ocfg.get("h264", True))
                        writer.write(canvas)
                    if frames_dir is not None and processed % max(1, ocfg.get("frame_every", 30)) == 0:
                        cv2.imwrite(str(frames_dir / f"frame_{fr.index:07d}.jpg"), canvas)
                    if on_frame is not None:
                        on_frame(canvas, {"frame": fr.index, "active": len(tracks)})
                if crops_dir is not None and processed % max(1, ocfg.get("crop_every", 15)) == 0:
                    from src.recognition.embeddings import crop_person

                    for tr in tracks:
                        c = crop_person(fr.image, tr.xyxy, pad=0.0)
                        if c is not None:
                            cv2.imwrite(str(crops_dir / f"id{tr.track_id:05d}_f{fr.index:07d}.jpg"), c)
                timer.add("render_write", time.perf_counter() - t)
                latencies.append(time.perf_counter() - t_read)
                processed += 1
                if progress is not None:
                    progress(processed / total if total else None, processed, {"frame": fr.index, "active": len(tracks)})

        try:
            batch: list[tuple[Frame, float]] = []
            t_read = time.perf_counter()
            for fr in reader:
                batch.append((fr, t_read))
                if len(batch) >= batch_size:
                    process_batch(batch)
                    batch = []
                if should_stop is not None and should_stop():
                    stopped_early = True
                    break
                t_read = time.perf_counter()
            if batch:
                process_batch(batch)
        finally:
            reader.close()
            final_video = writer.close() if writer is not None else None

        wall = time.perf_counter() - t_start
        if det_failures[0]:
            self.run_warnings.append(f"Detector failed on {det_failures[0]} frame(s); those frames have no detections.")
        if reader.decode_errors:
            self.run_warnings.append(f"{reader.decode_errors} frame(s) could not be decoded and were skipped.")
        if processed == 0:
            self.run_warnings.append("No frames were processed.")
        lat = np.asarray(latencies) * 1000 if latencies else np.zeros(1)
        timing = {
            "wall_time_s": round(wall, 3),
            "processed_frames": processed,
            "processing_fps": round(processed / wall, 2) if wall > 0 else 0.0,
            "source_fps": round(meta.fps, 3),
            "frame_stride": reader.stride,
            "realtime_factor": round((processed / wall) / reader.processed_fps, 3) if wall > 0 else 0.0,
            "stage_ms_per_frame": timer.report(),
            "latency_ms": {"mean": round(float(lat.mean()), 2), "p50": round(float(np.percentile(lat, 50)), 2),
                           "p95": round(float(np.percentile(lat, 95)), 2)},
            "device": self.device,
            "stopped_early": stopped_early,
            **peak_memory_mb(),
        }
        if self.preprocessor.enabled:
            timing["enhanced_frames"] = self.preprocessor.applied_frames
        md = meta.to_dict()
        md["source"] = Path(md["source"]).name if not meta.is_live else md["source"]
        cfg_snapshot = {"detector": self.detector.info(), "tracker": cfg["tracker"], "reid": cfg["reid"],
                        "preprocessing": cfg["preprocessing"], "video": vcfg}
        csv_path = manager.export_csv(run_dir / "tracks.csv")
        json_path = manager.export_json(run_dir / "tracks.json", metadata=md, config=cfg_snapshot, timing=timing)
        warnings = manager.warnings() + self.run_warnings
        if warnings:
            data = json.loads(json_path.read_text())
            data["warnings"] = warnings
            json_path.write_text(json.dumps(data, indent=2))
        if track_reid is not None and cfg["reid"].get("gallery_path"):
            track_reid.gallery.save(cfg["reid"]["gallery_path"])
        return PipelineResult(
            run_dir=run_dir,
            video_path=final_video,
            csv_path=csv_path,
            json_path=json_path,
            summary=manager.summary(),
            warnings=warnings,
            timing=timing,
            metadata=md,
            crops_dir=crops_dir,
            frames_dir=frames_dir,
            extra={"manager": manager},
        )
