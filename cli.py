#!/usr/bin/env python3
"""Command-line entry point for human-tracking-ai.

Examples:
  python cli.py info
  python cli.py sample-video data/samples/pan.mp4
  python cli.py track video.mp4 --set tracker.type=botsort reid.enabled=true
  python cli.py track 0 --set video.max_frames=300          # webcam 0
  python cli.py datasets catalog
  python cli.py datasets search "pedestrian detection"
  python cli.py datasets download owner/slug --yes
  python cli.py datasets convert coco --input ann.json --labels data/processed/person/labels
  python cli.py datasets validate data/processed/person/images data/processed/person/labels
  python cli.py datasets split data/processed/person/images data/processed/person
  python cli.py train --yes --set epochs=5
  python cli.py evaluate detection --split data/processed/person/test.txt --weights models/yolo26n.pt
  python cli.py evaluate tracking --gt MOT17-02/gt/gt.txt --pred outputs/run_x/tracks.csv
  python cli.py benchmark all
  python cli.py cleanup
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.utils.config import PROJECT_ROOT, load_config, parse_overrides
from src.utils.logging_config import setup_logging

log = logging.getLogger("cli")


def _cfg(args) -> dict:
    return load_config(getattr(args, "config", None), parse_overrides(getattr(args, "set", None)))


def cmd_info(args) -> int:
    from src.tracking.tracker import available_trackers
    from src.utils.device import hardware_report, resolve_device

    print(json.dumps({**hardware_report(), "selected_device": resolve_device("auto"), "trackers": available_trackers()}, indent=2))
    return 0


def cmd_sample(args) -> int:
    from src.utils.sample_video import make_pan_clip

    print(make_pan_clip(args.out, args.image, frames=args.frames, darken=args.darken))
    return 0


def cmd_track(args) -> int:
    from src.video.inference_pipeline import InferencePipeline

    cfg = _cfg(args)
    source = int(args.source) if args.source.isdigit() else args.source
    last = [-1]

    def progress(frac, n, info):
        pct = int(frac * 100) if frac is not None else None
        if pct is not None and pct // 10 != last[0]:
            last[0] = pct // 10
            log.info("progress %3d%% (%d frames, %d active tracks)", pct, n, info["active"])

    res = InferencePipeline(cfg).run(source, output_dir=args.out, progress=progress)
    print(json.dumps({"run_dir": str(res.run_dir), "video": str(res.video_path), "csv": str(res.csv_path),
                      "json": str(res.json_path), "summary": res.summary, "timing": res.timing,
                      "warnings": res.warnings}, indent=2))
    return 0


# ------------------------------------------------------------------- datasets
def cmd_datasets(args) -> int:
    from src.data import annotation_converter as ac
    from src.data.dataset_registry import CANDIDATE_DATASETS, DatasetRegistry

    if args.dcmd == "catalog":
        for d in CANDIDATE_DATASETS:
            print(f"- {d['name']}: {d['purpose']}\n    license: {d['license']}\n    official: {d['official_url']}\n"
                  f"    format: {d['annotation_format']}  modality: {d['modality']}\n"
                  f"    find on Kaggle: python cli.py datasets search \"{d['kaggle_search']}\"")
        return 0
    if args.dcmd in ("search", "download", "check-auth"):
        from src.data.kaggle_client import main as kmain

        argv = [args.dcmd]
        if args.dcmd == "search":
            argv += [args.query, "-n", str(args.n)]
        elif args.dcmd == "download":
            argv += [args.slug] + (["--yes"] if args.yes else []) + (["--dest", args.dest] if args.dest else [])
            argv += ["--format", args.format] if args.format else []
        return kmain(argv)
    if args.dcmd == "register":
        e = DatasetRegistry().register(name=args.name, source_url=args.url, license=args.license,
                                       annotation_format=args.format, local_path=args.path, version=args.version,
                                       modality=args.modality, notes=args.notes)
        print(json.dumps(e, indent=2))
        return 0
    if args.dcmd == "list":
        print(json.dumps(DatasetRegistry().all(), indent=2))
        return 0
    if args.dcmd == "convert":
        f = args.format
        if f == "coco":
            rep = ac.coco_to_yolo(args.input, args.labels)
        elif f == "voc":
            rep = ac.voc_to_yolo(args.input, args.labels)
        elif f == "crowdhuman":
            rep = ac.crowdhuman_to_yolo(args.input, args.images, args.labels, args.box_type)
        elif f == "mot":
            rep = ac.mot_to_yolo(args.input, args.images, args.labels, every_n=args.every_n)
        elif f == "widerperson":
            rep = ac.widerperson_to_yolo(args.input, args.images, args.labels)
        elif f == "exdark":
            rep = ac.exdark_to_yolo(args.input, args.images, args.labels)
        else:
            raise SystemExit(f"unknown format {f}")
        print(json.dumps(rep.to_dict(), indent=2))
        return 0
    if args.dcmd == "validate":
        from src.data.dataset_validator import validate_yolo_dataset, write_report

        rep = validate_yolo_dataset(args.images, args.labels, conditions_csv=args.conditions)
        j, m = write_report(rep, args.report_dir or PROJECT_ROOT / "reports", args.name)
        print(rep.to_markdown())
        print(f"Report: {j}, {m}")
        return 0 if rep.ok else 1
    if args.dcmd == "split":
        from src.data.dataset_splitter import split_dataset

        ratios = tuple(float(x) for x in args.ratios.split(","))
        res = split_dataset(args.images, args.out, ratios, args.seed, group_csv=args.groups)
        ac.write_data_yaml(Path(args.out) / "data.yaml", args.out, res["files"]["train"], res["files"]["val"], res["files"]["test"])
        print(json.dumps(res, indent=2))
        if res["leaking_groups"]:
            print("ERROR: groups leak across splits:", res["leaking_groups"][:10])
            return 1
        return 0
    return 1


# --------------------------------------------------------------------- training
def cmd_train(args) -> int:
    from src.training.train import load_training_config, train

    cfg = load_training_config(args.config, parse_overrides(args.set))
    res = train(cfg, confirm=args.yes, allow_large=args.allow_large)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in res.items()}, indent=2, default=str))
    if res.get("best") and args.promote:
        from src.training.checkpoint_manager import CheckpointManager

        print("Promoted to", CheckpointManager.promote(res["best"]))
    return 0


# ------------------------------------------------------------------- evaluation
def _detector(weights: str, args) -> object:
    from src.detection.detector import UltralyticsPersonDetector
    from src.utils.device import resolve_device

    return UltralyticsPersonDetector(weights, resolve_device(args.device), conf=0.001, imgsz=args.imgsz)


def cmd_evaluate(args) -> int:
    from src.evaluation import environmental_benchmarks as eb

    out_dir = Path(args.report_dir or PROJECT_ROOT / "reports")
    if args.ecmd == "detection":
        from src.data.dataset_splitter import read_split

        images = read_split(args.split)
        if args.max_images:
            images = images[: args.max_images]
        dets = {Path(w).stem: _detector(w, args) for w in args.weights}
        res = {"models": eb.compare_models(dets, images, conf=args.conf)}
        if args.conditions:
            from src.data.dataset_validator import load_conditions

            cond = load_conditions(args.conditions)
            res["per_condition"] = {k: eb.per_condition_detection(d, images, cond, args.min_images, args.conf) for k, d in dets.items()}
        if args.synthetic:
            first = next(iter(dets.values()))
            res["synthetic_corruptions"] = eb.synthetic_corruption_benchmark(first, images, severities=(2,), conf=args.conf)
        if args.enhancement:
            first = next(iter(dets.values()))
            res["enhancement"] = {"original": eb.compare_enhancement(first, images, conf=args.conf),
                                  "synthetic_low_light": eb.compare_enhancement(first, images, darken="low_light", conf=args.conf)}
        eb.write_json(res, out_dir / "detection_evaluation.json")
        print(eb.table(res["models"], ["precision", "recall", "f1", "ap50", "ap50_95", "tp", "fp", "fn"]))
        return 0
    if args.ecmd == "tracking":
        from src.evaluation.tracking_metrics import evaluate_tracking, load_mot_file, load_tracks_csv

        gt = load_mot_file(args.gt)
        pred = load_tracks_csv(args.pred) if args.pred.endswith(".csv") and "tracks" in Path(args.pred).name else load_mot_file(args.pred)
        res = evaluate_tracking(gt, pred, args.iou)
        eb.write_json(res, out_dir / "tracking_evaluation.json")
        print(json.dumps(res, indent=2))
        return 0
    return 1


def cmd_benchmark(args) -> int:
    from src.evaluation.run_benchmarks import run_all

    return run_all(args)


def cmd_export(args) -> int:
    from ultralytics import YOLO

    from src.detection.detector import resolve_weights

    path = YOLO(resolve_weights(args.weights)).export(format=args.format, imgsz=args.imgsz, dynamic=False)
    print(path)
    return 0


def cmd_cleanup(args) -> int:
    from src.utils.retention import apply_retention

    print(apply_retention(_cfg(args), PROJECT_ROOT / "uploads", PROJECT_ROOT / "outputs"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cli.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-level", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_cfg(p):
        p.add_argument("--config", default=None, help="YAML merged over configs/default.yaml")
        p.add_argument("--set", nargs="*", default=[], help="overrides like tracker.type=botsort")
        return p

    sub.add_parser("info", help="hardware and library report").set_defaults(fn=cmd_info)
    s = sub.add_parser("sample-video", help="generate a small demo clip from a bundled photo")
    s.add_argument("out")
    s.add_argument("--image", default=None)
    s.add_argument("--frames", type=int, default=60)
    s.add_argument("--darken", type=float, default=1.0)
    s.set_defaults(fn=cmd_sample)

    t = with_cfg(sub.add_parser("track", help="detect + track a video file, camera index or stream URL"))
    t.add_argument("source")
    t.add_argument("--out", default=None)
    t.set_defaults(fn=cmd_track)

    d = sub.add_parser("datasets", help="dataset discovery, download, conversion, validation, splitting")
    dsub = d.add_subparsers(dest="dcmd", required=True)
    dsub.add_parser("catalog")
    dsub.add_parser("list")
    dsub.add_parser("check-auth")
    x = dsub.add_parser("search")
    x.add_argument("query")
    x.add_argument("-n", type=int, default=20)
    x = dsub.add_parser("download")
    x.add_argument("slug")
    x.add_argument("--dest")
    x.add_argument("--format")
    x.add_argument("--yes", action="store_true")
    x = dsub.add_parser("register", help="record a manually imported dataset")
    for a in ("--name", "--url", "--license", "--format", "--path"):
        x.add_argument(a, required=True)
    x.add_argument("--version", default="unknown")
    x.add_argument("--modality", default="RGB")
    x.add_argument("--notes", default="")
    x = dsub.add_parser("convert")
    x.add_argument("format", choices=["coco", "voc", "crowdhuman", "mot", "widerperson", "exdark"])
    x.add_argument("--input", required=True, help="annotation file/dir (MOT: sequence dir)")
    x.add_argument("--images", help="images dir (MOT: output images dir)")
    x.add_argument("--labels", required=True, help="output YOLO labels dir")
    x.add_argument("--box-type", default="fbox")
    x.add_argument("--every-n", type=int, default=1)
    x = dsub.add_parser("validate")
    x.add_argument("images")
    x.add_argument("labels")
    x.add_argument("--conditions", default=None)
    x.add_argument("--report-dir", default=None)
    x.add_argument("--name", default="dataset_report")
    x = dsub.add_parser("split")
    x.add_argument("images")
    x.add_argument("out")
    x.add_argument("--ratios", default="0.8,0.1,0.1")
    x.add_argument("--seed", type=int, default=42)
    x.add_argument("--groups", default=None, help="CSV image_stem,group (e.g. video or camera id)")
    d.set_defaults(fn=cmd_datasets)

    tr = sub.add_parser("train", help="fine-tune the detector (requires --yes)")
    tr.add_argument("--config", default=None)
    tr.add_argument("--set", nargs="*", default=[])
    tr.add_argument("--yes", action="store_true")
    tr.add_argument("--allow-large", action="store_true")
    tr.add_argument("--promote", action="store_true", help="copy best.pt to models/person_finetuned.pt")
    tr.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate")
    esub = e.add_subparsers(dest="ecmd", required=True)
    x = esub.add_parser("detection")
    x.add_argument("--split", required=True, help="text file listing test images (YOLO labels alongside)")
    x.add_argument("--weights", nargs="+", default=["models/yolo26n.pt"], help="one or more models to compare")
    x.add_argument("--conditions", default=None)
    x.add_argument("--min-images", type=int, default=20)
    x.add_argument("--synthetic", action="store_true", help="also run synthetic corruption benchmark")
    x.add_argument("--enhancement", action="store_true", help="also compare enhancement on/off")
    x.add_argument("--conf", type=float, default=0.25)
    x.add_argument("--imgsz", type=int, default=640)
    x.add_argument("--device", default="auto")
    x.add_argument("--max-images", type=int, default=None)
    x.add_argument("--report-dir", default=None)
    x = esub.add_parser("tracking")
    x.add_argument("--gt", required=True)
    x.add_argument("--pred", required=True, help="MOT txt or this project's tracks.csv")
    x.add_argument("--iou", type=float, default=0.5)
    x.add_argument("--report-dir", default=None)
    e.set_defaults(fn=cmd_evaluate)

    b = sub.add_parser("benchmark", help="speed / tracker / ReID / enhancement benchmarks on available data")
    b.add_argument("what", choices=["all", "speed", "trackers", "robustness", "training"], nargs="?", default="all")
    b.add_argument("--device", default="auto")
    b.add_argument("--report-dir", default=None)
    b.add_argument("--with-training", action="store_true", help="include a 3-epoch CPU fine-tuning smoke test")
    b.set_defaults(fn=cmd_benchmark)

    x = sub.add_parser("export", help="export the detector (onnx, engine, openvino ...)")
    x.add_argument("--weights", default="models/yolo26n.pt")
    x.add_argument("--format", default="onnx")
    x.add_argument("--imgsz", type=int, default=640)
    x.set_defaults(fn=cmd_export)

    c = with_cfg(sub.add_parser("cleanup", help="apply the retention policy to uploads/ and outputs/"))
    c.set_defaults(fn=cmd_cleanup)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - one clear line instead of a traceback for expected errors
        from src.data.annotation_converter import AnnotationError
        from src.data.kaggle_client import KaggleAuthError, KaggleClientError
        from src.detection.detector import DetectorError
        from src.training.train import TrainingError
        from src.utils.config import ConfigError
        from src.video.video_reader import VideoError

        if isinstance(exc, (VideoError, ConfigError, DetectorError, AnnotationError, KaggleAuthError, KaggleClientError, TrainingError, FileNotFoundError)):
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
