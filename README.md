<<<<<<< HEAD
# human-tracking-ai

Detect, track and (optionally) re-identify people in video. Pretrained Ultralytics YOLO person detection,
BoT-SORT / ByteTrack / OC-SORT tracking, optional body-appearance ReID, Kaggle dataset tooling, a
fine-tuning pipeline with environmental augmentations, detection + tracking metrics (incl. HOTA, IDF1,
MOTA), benchmarks, a Streamlit UI and a CLI. Everything runs locally. No face recognition.

> Status (2026-10-16): the inference pipeline, UI, dataset tools, training pipeline and metrics are
> implemented and tested on CPU (65 automated tests passing, see [Tests](#tests)). Not yet done because they need
> resources this build environment did not have: training on a real Kaggle dataset (needs your Kaggle
> credentials and a GPU), real-condition benchmarks (needs labelled day/night/rain/fog test data), a
> trained person-ReID model (needs weights), and GPU/TensorRT runs (needs an NVIDIA GPU).

## Contents
- [Quick start](#quick-start)
- [What is implemented](#what-is-implemented)
- [Architecture and model choices](#architecture-and-model-choices)
- [Usage](#usage) (UI, CLI, configuration)
- [Datasets](#datasets) (Kaggle setup, manual import, conversion, validation, splitting)
- [Training](#training)
- [Evaluation and benchmarks](#evaluation-and-benchmarks)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Hardware requirements](#hardware-requirements)
- [Privacy, security and responsible use](#privacy-security-and-responsible-use)
- [Troubleshooting](#troubleshooting)
- [Known limitations and next steps](#known-limitations-and-next-steps)

## Quick start

```bash
cd human-tracking-ai
python3 -m venv .venv && source .venv/bin/activate
# GPU users: first install the CUDA build of torch/torchvision from https://pytorch.org/get-started/locally/
pip install -r requirements.txt            # add -r requirements-dev.txt for tests/ONNX/TrackEval

python cli.py info                          # hardware + library report, selected device
python cli.py sample-video data/samples/demo.mp4   # 4 s demo clip from a photo bundled with Ultralytics
python cli.py track data/samples/demo.mp4   # downloads yolo26n.pt (5.3 MB) to models/ on first use
streamlit run app.py                        # UI at http://127.0.0.1:8501
```

`cli.py track` writes `outputs/run_<time>_<id>/` with `annotated.mp4` (H.264), `tracks.csv`, `tracks.json`,
and optionally `crops/` and `frames/`.

## What is implemented

| Area | Implementation | Verified here |
|---|---|---|
| Video input | `src/video/video_reader.py`: MP4/AVI/MOV/MKV/WebM/... via OpenCV+FFmpeg, webcam index, RTSP/HTTP streams; frame stride or target FPS with correct source timestamps; aspect-preserving downscale; size/duration/extension limits; clear errors for missing, empty, corrupt or undecodable files | Files: yes (tests). Webcam/RTSP: code path only, no camera in the build environment |
| Detection | `src/detection/detector.py`: any Ultralytics checkpoint with a `person` class (default YOLO26n), CPU/CUDA/MPS, FP16 on CUDA, batching, box sanitising; `BaseDetector` interface so other detectors can be plugged in | CPU: yes. CUDA/FP16: not tested (no GPU) |
| Preprocessing | `src/detection/preprocessing.py`: CLAHE, gamma, CLAHE+gamma, dark-channel dehaze; `auto` mode enhances only dark scenes and checks brightness every N frames | Yes, plus an A/B benchmark |
| Tracking | `src/tracking/tracker.py`: standalone Ultralytics ByteTrack, BoT-SORT (with camera-motion compensation), OC-SORT, Deep OC-SORT, TrackTrack, FastTrack; all thresholds configurable; `track_buffer` auto-rescaled when skipping frames | Yes (all six run in tests) |
| Track history | `src/tracking/track_manager.py`, `trajectory.py`: per-track boxes, timestamps, confidences, path length, gaps; start/lost/recovered events; statistics and quality warnings; CSV + JSON export in original video coordinates | Yes |
| ReID (optional) | `src/recognition/`: per-track embedding sampling, gallery with cosine threshold, `unknown` until enough evidence, identities held by other visible tracks excluded, save/load gallery for cross-video use; backends `histogram` (no weights), `yolo_cls` (generic ImageNet features), `torchreid` (real ReID weights, e.g. OSNet) | `histogram`: yes. `yolo_cls`/`torchreid`: code only, weights not available here |
| Rendering | boxes, IDs, confidence, trails, header with time/active tracks/FPS; H.264 re-encode for browsers | Yes |
| Datasets | `src/data/`: Kaggle search/list/download (explicit confirmation), provenance registry, converters (COCO, VOC, CrowdHuman, MOT, WiderPerson, ExDark → YOLO), validator + stats report, group-aware leakage-free splitting with duplicate detection | Converters/validator/splitter: yes. Kaggle API: mocked only (Kaggle blocked from the build environment, no credentials) |
| Training | `src/training/`: Ultralytics fine-tuning from pretrained weights, seeds, AMP on CUDA, early stopping, checkpoints, resume, run metadata, download guard; offline environmental augmentation (13 transforms) of the train split only | Yes: 3-epoch CPU smoke run on coco128-person |
| Evaluation | `src/evaluation/`: P/R/F1/TP/FP/FN, AP50, AP50-95 (COCO 101-point); MOTA/IDF1/IDsw/Frag/MT/ML via `motmetrics`; HOTA/DetA/AssA/LocA (matches TrackEval to 1e-6); track persistence; per-condition, synthetic corruption, enhancement A/B, baseline-vs-fine-tuned, tracker comparison, speed; Markdown/JSON/PNG reports | Yes |
| UI | `app.py` (Streamlit): upload + preview + metadata, camera/stream tab, all model/tracker/ReID/preprocessing settings, progress, annotated video, metrics, trajectories, events, CSV/JSON/video downloads, warnings | Yes (headless test + browser smoke test) |
| Optimisation | ONNX export (`cli.py export`) and ONNX Runtime inference through the same detector class; adaptive resolution, frame stride, batching | ONNX on CPU: yes. TensorRT: not tested (no GPU) |

## Architecture and model choices

```
video/camera ─► VideoReader ─► Preprocessor ─► Detector ─► Tracker ─► (TrackReID) ─► TrackManager ─► writer + CSV/JSON
                (stride, ts)   (optional)      (YOLO)      (ByteTrack…)  (optional)     (history, events, stats)
```
`src/video/inference_pipeline.py` wires these together. Each stage is selected by configuration and can
fail independently: a detector error on a frame yields no detections for that frame, a tracker error
resets the tracker, a ReID error disables ReID, and every such event is reported as a warning.

**Detector: Ultralytics YOLO26n (default).** Current, well-supported, CPU-friendly (5.3 MB), NMS-free
head, COCO-pretrained with a person class, and the same API covers larger variants (`yolo26s/m/l/x.pt`),
YOLO11/YOLOv8 checkpoints, your fine-tuned `best.pt`, and ONNX/TensorRT exports. On CPU, n at 640 runs
at real-time-ish speed for a single stream (see benchmarks); on a GPU use `s` or `m` and `imgsz` 960-1280
for small/distant people. Alternatives considered: RT-DETR (better crowded-scene accuracy, much slower
on CPU) and pedestrian-specific detectors (better on CrowdHuman, but less maintained and harder to deploy).
The `BaseDetector` interface keeps that door open.

**Tracker: BoT-SORT default, ByteTrack as the lightweight alternative.** Both associate low-confidence
boxes in a second stage, which keeps partially occluded people tracked. In the synthetic crossing-paths
benchmark BoT-SORT scored HOTA 0.82 / IDF1 0.95 / 0.3 ID switches against ByteTrack's 0.70 / 0.87 / 6
(mean of 3 seeds), at the same tracking cost (~1 ms/frame) without camera-motion compensation. The
default keeps BoT-SORT's `sparseOptFlow` camera-motion compensation on, because uploaded footage is often
handheld; it costs ~7 ms/frame on this CPU. Set `tracker.gmc_method=none` for fixed cameras. Deep SORT is
not included: BoT-SORT and Deep OC-SORT are its maintained successors and accept appearance features.

**ReID: optional and off by default.** For real cross-camera re-identification use the `torchreid`
backend with an OSNet checkpoint trained on Market-1501/MSMT17 (see [ReID weights](#reid-weights)).
The always-available `histogram` backend only compares clothing colours and is a baseline, not a ReID
model. Labels are `V-0001`-style *visual identities*, never names.

## Usage

### UI
`streamlit run app.py`, then upload a video (or use the Camera / stream tab), adjust settings in the
sidebar, and press **Run detection and tracking**. The server binds to 127.0.0.1 (`.streamlit/config.toml`).

### CLI
```bash
python cli.py track video.mp4                                   # defaults from configs/default.yaml
python cli.py track video.mp4 --set tracker.type=botsort detector.imgsz=960 video.frame_stride=2
python cli.py track video.mp4 --set reid.enabled=true reid.gallery_path=outputs/gallery.npz   # cross-video gallery
python cli.py track video.mp4 --set preprocessing.enabled=true preprocessing.method=auto        # low-light aware
python cli.py track 0 --set video.max_frames=600                # webcam 0
python cli.py track rtsp://user:pass@host/stream --set video.max_frames=1000
python cli.py track video.mp4 --config my_overrides.yaml        # YAML merged over the defaults
python cli.py export --format onnx                              # then set detector.weights=models/yolo26n.onnx
python cli.py cleanup                                           # apply the retention policy now
```

### Output format
`tracks.csv`: one row per track per processed frame:
`frame, timestamp_s, track_id, x1, y1, x2, y2, confidence, reid_label, reid_similarity` (pixel
coordinates of the original video). `tracks.json`: video metadata, effective configuration, timing
(FPS, per-stage ms, latency p50/p95, peak memory), summary statistics, warnings, events
(`track_start`, `track_lost`, `track_recovered`) and every track with its full trajectory.

### Configuration
All options are documented inline in [`configs/default.yaml`](configs/default.yaml). The most useful:

| Key | Default | Effect |
|---|---|---|
| `detector.weights` | `models/yolo26n.pt` | any Ultralytics `.pt`/`.onnx`/`.engine` with a person class |
| `detector.imgsz` | 640 | larger finds smaller people, slower |
| `detector.conf` | 0.25 | lowest score passed to the tracker |
| `tracker.type` | `botsort` | `bytetrack`, `ocsort`, `deepocsort`, `tracktrack`, `fasttrack` |
| `tracker.gmc_method` | `sparseOptFlow` | camera-motion compensation; `none` for fixed cameras (faster) |
| `tracker.track_buffer` | 30 | source frames a lost track is kept (occlusion tolerance) |
| `tracker.match_thresh` | 0.8 | association threshold |
| `tracker.with_reid` | false | feed appearance features into BoT-SORT/Deep OC-SORT association |
| `reid.enabled` / `reid.similarity_threshold` | false / 0.80 | visual identity labels and their threshold |
| `video.frame_stride` / `video.target_fps` | 1 / null | frame skipping (timestamps stay correct) |
| `video.resize_max_side` | null | downscale before processing (aspect preserved, exports in original coordinates) |
| `preprocessing.method` | `auto` (disabled) | `clahe`, `gamma`, `clahe_gamma`, `dehaze` |
| `limits.*`, `retention.*` | 1 GB, 1 h, 24 h, 7 d | upload limits and deletion policy |

## Datasets

### Configure Kaggle access
1. Sign in at kaggle.com → *Settings* → *API* → create a token.
2. Provide it in ONE of these ways (never in this repo's files):
   - `export KAGGLE_API_TOKEN=<token>` (or save the token to `~/.kaggle/access_token`), or
   - legacy key: `export KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>`, or save `kaggle.json` to
     `~/.kaggle/kaggle.json` and `chmod 600 ~/.kaggle/kaggle.json`.
3. `python cli.py datasets check-auth`

### Find and download
```bash
python cli.py datasets catalog                       # researched candidates with original licenses
python cli.py datasets search "crowdhuman"           # live Kaggle search: slug, size, license
python cli.py datasets download <owner>/<slug> --format crowdhuman --yes
python cli.py datasets list                          # provenance registry (data/registry.json)
```
Downloads go to `data/raw/<owner>__<slug>/` and are recorded with source URL, version, license and date.
Nothing downloads without `--yes`.

Candidate datasets (license from the original publisher; a Kaggle mirror cannot grant more):

| Dataset | Use | Original license | Notes |
|---|---|---|---|
| CrowdHuman | crowded detection, occlusion | non-commercial research | full/visible/head boxes (`--box-type fbox/vbox`) |
| WiderPerson | dense pedestrians in the wild | academic research | classes 1-3 kept, ignore regions dropped |
| MOT17 / MOT20 | tracking benchmarks, crowded/night (MOT20) | CC BY-NC-SA 3.0 | converter keeps sequence grouping for leak-free splits |
| BDD100K | weather and time-of-day tags (pedestrian) | BDD100K terms (non-commercial) | best source for per-condition evaluation |
| ExDark | low light (People class) | BSD-3 repo, research use | |
| LLVIP | night, paired RGB + thermal | research use | thermal is a separate modality: train/evaluate separately |
| Market-1501 | person ReID | research use | for training/evaluating a ReID model |

Kaggle was not reachable from the environment this was built in, so the Kaggle slugs and the current
license text on each Kaggle page were **not** verified. Run `datasets search` and read the page before use.

### Manual import (no Kaggle access)
Download from the official site, unpack under `data/raw/<name>/`, then:
```bash
python cli.py datasets register --name CrowdHuman --url https://www.crowdhuman.org/ \
    --license "non-commercial research" --format crowdhuman --path data/raw/crowdhuman
```

### Convert, validate, split
```bash
# Convert to YOLO labels (single class: person). Formats: coco voc crowdhuman mot widerperson exdark
python cli.py datasets convert crowdhuman --input data/raw/crowdhuman/annotation_train.odgt \
    --images data/raw/crowdhuman/Images --labels data/processed/person/labels
#   (put/symlink the images into data/processed/person/images; MOT conversion does this for you:)
python cli.py datasets convert mot --input data/raw/MOT17/train/MOT17-02-FRCNN \
    --images data/processed/person/images --labels data/processed/person/labels --every-n 3

python cli.py datasets validate data/processed/person/images data/processed/person/labels \
    --conditions data/processed/person/conditions.csv      # optional: image_stem,condition
python cli.py datasets split data/processed/person/images data/processed/person --ratios 0.8,0.1,0.1 --seed 42 \
    --groups data/processed/person/groups.csv              # optional: image_stem,group (video/camera id)
```
The validator reports corrupt images, missing/orphan labels, malformed lines, out-of-range boxes,
unknown classes, duplicate files, class counts, resolutions, box-size buckets, people per image and
conditions (from `conditions.csv`, otherwise a brightness *estimate* labelled as such). The splitter
assigns whole groups (video/sequence/camera; inferred from names like `MOT17-02__000001` or
`cam1_frame_0001`, or from `--groups`) to one split, keeps duplicate files together, is deterministic
for a seed, and fails if any group leaks across splits.

## Training
```bash
python cli.py train --yes                                   # configs/training.yaml
python cli.py train --yes --set epochs=100 batch=32 imgsz=960 weights=yolo26s.pt
python cli.py train --yes --set environmental.enabled=true environmental.copies=1   # + fog/rain/night... copies of TRAIN only
python cli.py train --yes --set resume=true                 # continue the latest run from last.pt
python cli.py train --yes --promote                         # copy best.pt to models/person_finetuned.pt
```
Starts from pretrained weights; seeds fixed (`seed`, `deterministic=True`); early stopping (`patience`);
AMP on CUDA; checkpoints `models/runs/<name>/weights/{last,best}.pt` every epoch; `run_metadata.json`
records config, hardware, dataset registry entries and final metrics. Weights over
`max_auto_download_mb` need `--allow-large`. On-the-fly Ultralytics augmentations (HSV, scale,
translate, flip, mosaic, random erasing) are configured under `augment:`; the 13 environmental
transforms in `src/training/augmentations.py` (brightness/contrast, gamma, colour temperature, motion
and defocus blur, sensor noise, shadow, glare, fog, rain, low light, resolution loss, occlusion) are
applied probabilistically, never stacked unrealistically (e.g. fog and night are exclusive), and only to
the training split.

Then compare baseline and fine-tuned on the same held-out test split:
```bash
python cli.py evaluate detection --split data/processed/person/test.txt \
    --weights models/yolo26n.pt models/person_finetuned.pt --conditions data/processed/person/conditions.csv \
    --synthetic --enhancement
```

## Evaluation and benchmarks
```bash
python cli.py evaluate detection --split <test.txt> --weights <a.pt> [<b.pt> ...]   # P/R/F1/AP50/AP50-95/TP/FP/FN
python cli.py evaluate tracking --gt MOT17-02-FRCNN/gt/gt.txt --pred outputs/<run>/tracks.csv  # HOTA/IDF1/MOTA/...
python cli.py benchmark all --with-training                 # everything below, writes reports/benchmark_report.md
```
Per-condition numbers are only reported for conditions that have at least `--min-images` labelled test
images; everything else is listed as skipped. Synthetic corruption results are always labelled synthetic.

### Results measured in this build (CPU only)
Full tables and plots: [`reports/benchmark_report.md`](reports/benchmark_report.md) (raw: `benchmark_report.json`).
Machine: 4-core Xeon @ 2.1 GHz, 16.9 GB RAM, no GPU; Python 3.13, torch 2.14.1, ultralytics 8.4.178.

- **Detector speed, YOLO26n, 640x480 frames:** 17.1 FPS at imgsz 640 (58 ms), 23.2 FPS at 480, 51 FPS at 320.
  ONNX Runtime at 640 was not faster on this CPU (15.7 FPS), so PyTorch stays the default here.
- **End-to-end pipeline (decode, detect, track, render, write):** 18.2 FPS with ByteTrack, 15.6 FPS with
  BoT-SORT + camera-motion compensation, 18.8 FPS with BoT-SORT without it, 28.9 FPS at imgsz 320.
  ReID labels (histogram) add ~0.2 ms/frame. Peak RSS ~1.1 GB.
- **Trackers (synthetic crowd, exact ground truth, mean of 3 seeds):** BoT-SORT HOTA 0.824, IDF1 0.946,
  0.3 ID switches; ByteTrack 0.704 / 0.868 / 6.0; OC-SORT 0.663 / 0.826 / 1.7. Adding the histogram
  appearance features changed nothing measurable (0.824 → 0.824), which is expected from a colour-only
  embedding and is why a trained ReID model is the next step. Raising ByteTrack's `track_buffer` from 30
  to 60 also changed nothing here.
- **Detection robustness (coco128 person subset, 254 people in 128 images; optimistic because these
  images are in COCO train2017):** clean AP50 0.776. AP50 at severity 2/3: motion blur 0.503/0.335,
  low resolution 0.520/0.366, rain 0.676/0.561, low light 0.698/0.658, sensor noise 0.741/0.690,
  fog 0.763/0.713, glare 0.766/0.738. Recall is what falls; precision mostly holds.
- **Does enhancement help?** On normal images, no (AP50 0.776 → 0.770-0.779, recall drops slightly), so
  preprocessing is off by default. On synthetic low light (severity 3) gamma raised AP50 0.658 → 0.711 and
  CLAHE to 0.680; on synthetic heavy fog, dehazing raised AP50 0.713 → 0.758. That is why `auto` mode
  only enhances dark frames. Confirm on real night footage before relying on it.
- **Fine-tuning smoke test:** 3 epochs on 89 coco128 images on CPU (0.4 min) produced checkpoints and a
  baseline-vs-fine-tuned table (AP50 0.547 vs 0.539 on 19 test images at imgsz 320). This only proves the
  pipeline works. With the original default learning rate (AdamW 1e-3) the same run fell to AP50 0.30,
  which is why the default is now 2e-4 with no bias warm-up.

Not measured (no data or hardware here): real per-condition metrics (needs labelled condition data such
as BDD100K or ExDark), MOT17/MOT20 HOTA/IDF1, ReID with a trained model, GPU/FP16/TensorRT speed.

## Project structure
```
human-tracking-ai/
├── app.py                     Streamlit UI
├── cli.py                     CLI: info, sample-video, track, datasets, train, evaluate, benchmark, export, cleanup
├── configs/default.yaml       inference config (documented inline)
├── configs/training.yaml      fine-tuning config
├── requirements.txt / requirements-dev.txt / .env.example / .streamlit/config.toml
├── docs/ABOUT.md              privacy notes and limitations (also shown in the UI)
├── src/
│   ├── data/        kaggle_client, dataset_registry, annotation_converter, dataset_validator, dataset_splitter, sample_datasets
│   ├── detection/   detector (BaseDetector, Ultralytics YOLO), preprocessing (CLAHE/gamma/dehaze)
│   ├── tracking/    tracker (Ultralytics trackers, standalone), track_manager, trajectory
│   ├── recognition/ embeddings (histogram, yolo_cls, torchreid), person_reid (gallery, per-track labels)
│   ├── video/       video_reader, video_writer, inference_pipeline
│   ├── training/    train, augmentations (13 environmental transforms + eval corruptions), checkpoint_manager
│   ├── evaluation/  detection_metrics, tracking_metrics (HOTA/IDF1/MOTA), environmental_benchmarks, tracker_benchmark, run_benchmarks
│   └── utils/       config, device, logging_config, visualization, retention, privacy, sample_video
├── tests/                     65 tests (pytest)
├── reports/                   benchmark_report.md/.json + plots (generated)
├── data/raw, data/processed   datasets (git-ignored)
├── models/                    weights and training runs (git-ignored)
├── outputs/, uploads/         results and uploads (git-ignored, retention policy applies)
```

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest -q                 # all tests (~30 s on 4 CPU cores)
python -m pytest -q -m "not integration"   # skip tests that need models/yolo26n.pt
```
Executed in this build (2026-10-16, CPU): **65 passed, 0 failed, 0 skipped** in ~10 s, including the 5
integration tests that use the real YOLO26n weights. Coverage by acceptance criterion:

| Criterion | Tests |
|---|---|
| App starts | `test_app.py::test_app_starts` (headless Streamlit), plus a manual browser run: upload, process, results and downloads rendered with no page errors |
| Videos open, decode, process | `test_video_reader.py` (7), `test_pipeline.py` |
| Valid detector coordinates/confidences | `test_detector.py` (sanitising + real model on `bus.jpg`, blank frame, batch) |
| Stable IDs in controlled sequences | `test_tracker.py` (ByteTrack/BoT-SORT/OC-SORT keep 3 IDs; all 6 trackers run) |
| Missed detections and occlusion | `test_tracker.py` (4-frame occlusion, 4 frames with no detections), `test_pipeline.py` (detector crashing every 3rd frame) |
| Valid CSV/JSON | `test_tracker.py`, `test_pipeline.py` |
| Useful errors for bad videos/annotations | `test_video_reader.py`, `test_dataset_validation.py` (invalid JSON, missing keys, bad MOT line, broken XML) |
| CPU inference | real-model tests run on CPU |
| Kaggle auth failures handled safely | `test_dataset_validation.py` (no credentials, rejected credentials without leaking the key, confirmation required, slug validation) |
| End-to-end on a small clip | `test_pipeline.py::test_end_to_end_real_model`, `test_cli_track_command` |
| Metrics correct | `test_metrics.py` (hand-computed P/R/F1, AP=1 for perfect, IDF1=0.5 for a swap, HOTA equals TrackEval to 1e-6) |

## Hardware requirements
| Task | Minimum | Recommended |
|---|---|---|
| Inference, YOLO26n, 640 px | 4 CPU cores, 4 GB RAM (measured ~0.8 GB peak RSS) | any CUDA GPU with 4 GB+ |
| Inference, larger models / 960-1280 px / multiple streams | CUDA GPU 6 GB | RTX 3060 12 GB or better; TensorRT export |
| Fine-tuning YOLO26n/s on CrowdHuman-scale data (~15k images, 640 px) | CUDA GPU 8 GB, 16 GB RAM, ~50 GB disk | 16-24 GB GPU, 32 GB RAM, NVMe; expect hours per run |
| Fine-tuning m/l models at 960+ px | 24 GB GPU | multi-GPU |
| ReID model training (Market-1501/MSMT17) | 8 GB GPU | 12 GB+ |
CPU-only training works (the smoke test ran on CPU) but is only practical for tiny datasets.

## Privacy, security and responsible use
- Local processing only; no frames are transmitted. Ultralytics usage analytics are disabled in-process (`src/utils/privacy.py`).
- No face detection/recognition. Any future face module must be a separate, explicitly opt-in component with consent, purpose limitation, access control and audit logging.
- ReID labels express visual similarity, not identity; nothing attributes identity to a named person. Track-ID counts are not people counts (stated in the UI and in every JSON export).
- Uploads: extension whitelist, size/duration limits, random server-side file names (user file names are never used as paths), decoder validation before processing.
- Retention: uploads deleted after 24 h and outputs/crops after 7 days by default (`retention.*`), applied at app start and via `cli.py cleanup`.
- Credentials only via Kaggle's own env vars/files; `.env`, `kaggle.json` and tokens are git-ignored and never logged.
- The Streamlit app has no authentication and binds to localhost. Add an authenticating reverse proxy (or Streamlit auth) and per-user storage before multi-user deployment.
- Not validated for safety-critical or high-stakes decisions (law enforcement, employment, access control).

## Troubleshooting
| Symptom | Fix |
|---|---|
| `Could not open ... codec` | Re-encode: `ffmpeg -i in.xyz -c:v libx264 -pix_fmt yuv420p out.mp4` |
| Annotated video does not play in the browser | Install `ffmpeg` (H.264 re-encode); otherwise the output is MPEG-4 Part 2 |
| Weights download fails (offline) | Put `yolo26n.pt` in `models/` and set `detector.allow_download=false` |
| `CUDA requested but not available` | Install the CUDA build of torch; the code falls back to CPU automatically |
| Many short tracks / ID switches | `tracker.type=botsort`, raise `tracker.track_buffer`, lower `video.frame_stride`, lower `detector.conf` slightly |
| Small/distant people missed | raise `detector.imgsz` (960/1280), use a larger model, avoid `resize_max_side` |
| Kaggle `401`/auth errors | `python cli.py datasets check-auth`; regenerate the token; check `~/.kaggle` permissions |
| Training OOM | lower `batch`/`imgsz`, or `batch=-1` (auto-batch, GPU) |

## Known limitations and next steps
Verified vs not verified:
- **Verified here (CPU):** video I/O, detection, all six trackers, ReID pipeline with the histogram
  backend, exports, UI, dataset converters/validator/splitter, training/checkpoint/resume plumbing,
  metrics, benchmarks.
- **Implemented but not run here:** Kaggle API calls against the real service (Kaggle was unreachable
  and no credentials were configured), webcam/RTSP input (no camera), CUDA/FP16/AMP and TensorRT (no
  GPU), the `yolo_cls` and `torchreid` ReID backends (weights not downloadable from this environment).

Limitations:
- Track-ID counts are not people counts; IDs survive occlusions only up to `track_buffer`.
- Accuracy drops sharply with motion blur and low resolution (see benchmarks); small or distant people need a larger `imgsz`.
- The histogram ReID backend is a colour baseline and did not improve tracking; people in similar clothes will be confused.
- Thermal/IR input is not supported by the RGB model.
- Mannequins, posters and screens can be detected as people.
- The upload preview in the UI only plays browser-compatible codecs (H.264/VP9); processing works regardless and the annotated output is always H.264.
- No authentication: single-user, localhost use only as shipped.

Recommended next steps, in order:
1. Configure Kaggle (or download manually), then build a person dataset from CrowdHuman + a night/adverse
   weather source (ExDark, BDD100K), with `conditions.csv` so per-condition metrics become real.
2. On a GPU, fine-tune `yolo26s` at imgsz 960 with `environmental.enabled=true`, add hard negatives
   (mannequins, posters, empty scenes), and compare baseline vs fine-tuned with `evaluate detection`.
3. Evaluate tracking on MOT17/MOT20 train sequences (`evaluate tracking`) to tune `track_buffer`,
   `match_thresh` and BoT-SORT vs ByteTrack on real footage.
4. Install `torchreid` with OSNet MSMT17 weights, enable `tracker.with_reid` + `reid.backend=torchreid`, and measure
   the IDF1/HOTA/ID-switch change on MOT17.
5. Export to TensorRT FP16 on the target GPU and re-run `benchmark speed`.

### ReID weights
`pip install git+https://github.com/KaiyangZhou/deep-person-reid.git`, download an OSNet checkpoint
(e.g. `osnet_x0_25_msmt17`) from the torchreid model zoo into `models/`, then
`--set reid.enabled=true reid.backend=torchreid reid.torchreid_model=osnet_x0_25 reid.weights=models/<file>.pth`.
Check the license of the dataset the weights were trained on before commercial use.
=======
# IRIS
IRIS - Immediate Response Intelligent Surveillance
>>>>>>> fb885fe73f19252ad71de2a3342eae66f4a641fa
