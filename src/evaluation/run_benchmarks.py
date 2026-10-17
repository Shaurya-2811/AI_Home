"""Run every benchmark that is possible with locally available data and write reports/benchmark_report.md.

Data used (all small and public): a pan clip generated from the photo bundled with Ultralytics, the
synthetic crowd from ``tracker_benchmark``, and the person subset of coco128 (~7 MB download on first run).
Every number in the report is measured in this run on this machine.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import yaml

from src.evaluation import environmental_benchmarks as eb
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.device import hardware_report, resolve_device

log = logging.getLogger(__name__)


def _frames(path: Path, n: int = 60):
    from src.video.video_reader import VideoReader

    with VideoReader(path, max_frames=n) as r:
        return [f.image for f in r]


def speed_section(device: str, report_dir: Path, md: list[str], results: dict) -> None:
    from src.detection.detector import UltralyticsPersonDetector
    from src.utils.sample_video import make_pan_clip
    from src.video.inference_pipeline import InferencePipeline

    clip = make_pan_clip(report_dir / "assets" / "pan_clip.mp4", frames=60, fps=15)
    frames = _frames(clip)
    rows = {}
    for imgsz in (320, 480, 640):
        det = UltralyticsPersonDetector("models/yolo26n.pt", device, imgsz=imgsz)
        rows[f"yolo26n pytorch imgsz={imgsz}"] = eb.benchmark_speed(det, frames)
        rows[f"yolo26n pytorch imgsz={imgsz}"]["detections_per_frame"] = round(sum(len(det.detect(f)) for f in frames[:20]) / 20, 2)
    try:
        from ultralytics import YOLO

        onnx_path = Path(YOLO(str(PROJECT_ROOT / "models" / "yolo26n.pt")).export(format="onnx", imgsz=640, verbose=False))
        det = UltralyticsPersonDetector(str(onnx_path), device, imgsz=640)
        rows["yolo26n onnxruntime imgsz=640"] = eb.benchmark_speed(det, frames)
        rows["yolo26n onnxruntime imgsz=640"]["detections_per_frame"] = round(sum(len(det.detect(f)) for f in frames[:20]) / 20, 2)
    except Exception as exc:  # noqa: BLE001
        rows["yolo26n onnxruntime imgsz=640"] = {"error": str(exc)[:200]}
    results["detector_speed"] = rows
    md += ["## Detector speed (640x480 frames)", "", eb.table(rows, ["fps", "latency_ms_mean", "latency_ms_p95", "detections_per_frame"]), ""]

    pipe_rows = {}
    variants = {
        "bytetrack": {},
        "botsort (GMC sparseOptFlow)": {"tracker": {"type": "botsort"}},
        "botsort (no GMC)": {"tracker": {"type": "botsort", "gmc_method": "none"}},
        "botsort + histogram appearance": {"tracker": {"type": "botsort", "with_reid": True}},
        "bytetrack + ReID labels (histogram)": {"reid": {"enabled": True}},
        "bytetrack, frame_stride=2": {"video": {"frame_stride": 2}},
        "bytetrack, imgsz=320": {"detector": {"imgsz": 320}},
    }
    for name, ov in variants.items():
        cfg = load_config(overrides={"device": device, **ov, "output": {"h264": False}})
        res = InferencePipeline(cfg).run(clip, output_dir=report_dir / "assets" / "runs", run_name=name.split()[0] + str(abs(hash(name)) % 1000))
        t = res.timing
        pipe_rows[name] = {"processing_fps": t["processing_fps"], "realtime_factor": t["realtime_factor"],
                           "detect_ms": t["stage_ms_per_frame"].get("detect"), "track_ms": t["stage_ms_per_frame"].get("track"),
                           "embed_or_reid_ms": t["stage_ms_per_frame"].get("embed", t["stage_ms_per_frame"].get("reid")),
                           "latency_p95_ms": t["latency_ms"]["p95"], "peak_rss_mb": t.get("peak_rss_mb"),
                           "track_ids": res.summary["unique_track_ids"]}
    results["pipeline_speed"] = pipe_rows
    md += ["## End-to-end pipeline speed (60-frame 640x480 pan clip, 15 FPS source)", "",
           "realtime_factor = processed FPS / FPS needed to keep up with the (sampled) source. Peak RSS is the process peak so far, so it only grows across rows.", "",
           eb.table(pipe_rows, ["processing_fps", "realtime_factor", "detect_ms", "track_ms", "embed_or_reid_ms", "latency_p95_ms", "peak_rss_mb", "track_ids"]), ""]


def tracker_section(md: list[str], results: dict, report_dir: Path) -> None:
    from src.evaluation.tracker_benchmark import run_tracker_benchmark
    from src.recognition.embeddings import HistogramEmbedder

    base = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text())["tracker"]
    cfgs = {
        "bytetrack": {**base, "type": "bytetrack"},
        "botsort": {**base, "type": "botsort", "gmc_method": "none"},
        "botsort + appearance (histogram)": {**base, "type": "botsort", "gmc_method": "none", "with_reid": True},
        "ocsort": {**base, "type": "ocsort"},
        "deepocsort + appearance (histogram)": {**base, "type": "deepocsort", "with_reid": True},
        "bytetrack buffer=60": {**base, "type": "bytetrack", "track_buffer": 60},
    }
    rows = run_tracker_benchmark(cfgs, seeds=(0, 1, 2), embedder=HistogramEmbedder())
    results["tracker_comparison_synthetic"] = rows
    md += ["## Tracker comparison (SYNTHETIC crowd, simulated detector, mean of 3 seeds)", "",
           "8 people on crossing paths, a pillar occluder, 10% misses, 4% box jitter, occasional false positives. "
           "Isolates association quality from detector quality; it is not a substitute for MOT17/MOT20.", "",
           eb.table(rows, ["HOTA", "DetA", "AssA", "idf1", "mota", "num_switches", "num_fragmentations", "track_persistence", "tracker_fps"]), ""]
    eb.bar_plot(rows, "HOTA", report_dir / "plots" / "tracker_hota_synthetic.png", "HOTA by tracker (synthetic)")
    md += ["![tracker HOTA](plots/tracker_hota_synthetic.png)", ""]


def robustness_section(device: str, md: list[str], results: dict, report_dir: Path) -> list[str]:
    from src.data.sample_datasets import prepare_coco128_person
    from src.detection.detector import UltralyticsPersonDetector

    data_yaml = prepare_coco128_person()
    d = yaml.safe_load(Path(data_yaml).read_text())
    all_imgs = sorted(str(p) for p in (Path(d["path"]) / "images").glob("*.jpg"))
    det = UltralyticsPersonDetector("models/yolo26n.pt", device, conf=0.001)
    clean = eb.compare_models({"yolo26n": det}, all_imgs)["yolo26n"]
    syn = eb.synthetic_corruption_benchmark(det, all_imgs, severities=(1, 2, 3))
    results["coco128_person_clean"] = clean
    results["synthetic_corruptions"] = syn
    md += ["## Detection robustness (coco128 person subset, 128 images)", "",
           "CAVEAT: coco128 images come from COCO train2017, which the pretrained detector was trained on, so absolute "
           "numbers are optimistic. The relative drop under each synthetic corruption is the informative part.", "",
           eb.table({"clean": clean}, ["precision", "recall", "f1", "ap50", "ap50_95", "num_gt", "tp", "fp", "fn"]), "",
           "### Synthetic corruptions (severity 1-3)", "",
           eb.table(syn["corrupted"], ["precision", "recall", "f1", "ap50", "ap50_95"]), ""]
    eb.bar_plot({"clean": clean, **{k: v for k, v in syn["corrupted"].items() if k.endswith("@2")}}, "ap50",
                report_dir / "plots" / "corruption_ap50.png", "AP50 under synthetic corruptions (severity 2)")
    md += ["![corruptions](plots/corruption_ap50.png)", ""]

    enh = {"original images": eb.compare_enhancement(det, all_imgs),
           "synthetic low light @2": eb.compare_enhancement(det, all_imgs, darken="low_light", severity=2),
           "synthetic low light @3": eb.compare_enhancement(det, all_imgs, darken="low_light", severity=3),
           "synthetic fog @3": eb.compare_enhancement(det, all_imgs, darken="fog", severity=3, methods=("dehaze", "clahe"))}
    results["enhancement_ab"] = enh
    md += ["## Enhancement A/B (does preprocessing help?)", ""]
    for name, block in enh.items():
        rows = {k: v for k, v in block.items() if isinstance(v, dict)}
        md += [f"### {name}", "", eb.table(rows, ["ap50", "ap50_95", "recall", "precision"]), ""]
    return d["test"] if isinstance(d.get("test"), str) else []


def training_section(device: str, md: list[str], results: dict, report_dir: Path) -> None:
    """Fine-tuning smoke test: proves the pipeline trains, checkpoints and compares; not a quality claim."""
    from src.data.sample_datasets import prepare_coco128_person
    from src.detection.detector import UltralyticsPersonDetector
    from src.training.train import load_training_config, train

    data_yaml = prepare_coco128_person()
    d = yaml.safe_load(Path(data_yaml).read_text())
    cfg = load_training_config(overrides={"data": str(data_yaml), "epochs": 3, "batch": 8, "imgsz": 320, "workers": 0, "lr0": 0.0001, "warmup_epochs": 0,
                                          "device": device, "name": "coco128_smoke", "plots": False, "patience": 0})
    t0 = time.perf_counter()
    res = train(cfg, confirm=True)
    minutes = (time.perf_counter() - t0) / 60
    from src.data.dataset_splitter import read_split

    test_imgs = read_split(Path(d["path"]) / d["test"])
    comp = eb.compare_models({
        "baseline yolo26n (imgsz 320)": UltralyticsPersonDetector("models/yolo26n.pt", device, conf=0.001, imgsz=320),
        "fine-tuned 3 epochs (imgsz 320)": UltralyticsPersonDetector(str(res["best"]), device, conf=0.001, imgsz=320),
    }, test_imgs)
    results["finetune_smoke"] = {"minutes": minutes, "run_dir": str(res["run_dir"]), "comparison": comp}
    md += ["## Fine-tuning smoke test (coco128 person, 3 epochs, CPU, imgsz 320, AdamW lr0=1e-4)", "",
           f"Training ran for {minutes:.1f} min and wrote checkpoints to `{Path(res['run_dir']).relative_to(PROJECT_ROOT)}`. "
           f"Test split: {len(test_imgs)} images (group-aware split, seed 42). This only demonstrates that training, "
           "checkpointing and baseline-vs-fine-tuned comparison work; 3 epochs on ~90 images is not expected to improve "
           "a COCO-pretrained model, and the test images were part of its pretraining data.", "",
           eb.table(comp, ["precision", "recall", "f1", "ap50", "ap50_95", "num_gt"]), ""]


def run_all(args) -> int:
    device = resolve_device(getattr(args, "device", "auto"))
    what = getattr(args, "what", "all")
    report_dir = Path(getattr(args, "report_dir", None) or PROJECT_ROOT / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"hardware": hardware_report(), "device": device, "date": time.strftime("%Y-%m-%d %H:%M")}
    hw = results["hardware"]
    md = ["# Benchmark report", "",
          f"Generated {results['date']} by `python cli.py benchmark {what}`. All numbers measured in this run.", "",
          f"Hardware: {hw.get('cpu_count')} CPU cores, {hw.get('ram_gb')} GB RAM, CUDA available: {hw.get('cuda_available')}, "
          f"device used: `{device}`. Python {hw.get('python')}, torch {hw.get('torch')}, ultralytics {hw.get('ultralytics')}, "
          f"OpenCV {hw.get('cv2')}.", ""]
    name = "benchmark_report" if what == "all" else f"benchmark_{what}"
    sections = [
        ("speed", what in ("all", "speed"), lambda: speed_section(device, report_dir, md, results)),
        ("trackers", what in ("all", "trackers"), lambda: tracker_section(md, results, report_dir)),
        ("robustness", what in ("all", "robustness"), lambda: robustness_section(device, md, results, report_dir)),
        ("training", what == "training" or (what == "all" and getattr(args, "with_training", False)),
         lambda: training_section(device, md, results, report_dir)),
    ]
    for sec, enabled, fn in sections:
        if not enabled:
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report the failure, keep the other sections
            log.exception("Benchmark section %s failed", sec)
            md += [f"## {sec}: FAILED", "", f"`{type(exc).__name__}: {exc}`", ""]
            results[f"{sec}_error"] = str(exc)
        # write after every section so a late failure never loses earlier results
        (report_dir / f"{name}.md").write_text("\n".join(md))
        eb.write_json(results, report_dir / f"{name}.json")
    print("\n".join(md))
    print(f"\nWrote {report_dir / (name + '.md')}")
    return 0
