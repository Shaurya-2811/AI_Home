"""Streamlit interface for human detection and tracking.

Run:  streamlit run app.py
Everything is processed locally; no frames leave this machine.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.config import PROJECT_ROOT, ConfigError, load_config
from src.utils.logging_config import setup_logging

setup_logging()
st.set_page_config(page_title="Human Tracking AI", layout="wide")

UPLOADS = PROJECT_ROOT / "uploads"
OUTPUTS = PROJECT_ROOT / "outputs"
BASE_CFG = load_config()


@st.cache_resource(show_spinner=False)
def _startup() -> dict:
    """Runs once per server process: apply the retention policy."""
    from src.utils.retention import apply_retention

    return apply_retention(BASE_CFG, UPLOADS, OUTPUTS)


@st.cache_resource(show_spinner="Loading detector...")
def get_detector(weights: str, device: str, imgsz: int, conf: float, iou: float, allow_download: bool):
    from src.detection.detector import UltralyticsPersonDetector

    return UltralyticsPersonDetector(weights, device, conf=conf, iou=iou, imgsz=imgsz, allow_download=allow_download)


@st.cache_data(show_spinner=False)
def hardware() -> dict:
    from src.utils.device import hardware_report

    return hardware_report()


def sidebar_config() -> dict:
    from src.tracking.tracker import available_trackers
    from src.utils.device import resolve_device

    cfg = copy.deepcopy(BASE_CFG)
    sb = st.sidebar
    sb.header("Model")
    local = sorted(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "models").rglob("*.pt"))
    choices = local + ["yolo26n.pt (download)", "yolo26s.pt (download)", "yolo11n.pt (download)", "custom path..."]
    default = choices.index(cfg["detector"]["weights"]) if cfg["detector"]["weights"] in choices else 0
    pick = sb.selectbox("Detector checkpoint", choices, index=default,
                        help="Any Ultralytics YOLO checkpoint with a 'person' class, e.g. your fine-tuned best.pt")
    if pick == "custom path...":
        pick = sb.text_input("Checkpoint path", value=cfg["detector"]["weights"])
    cfg["detector"]["weights"] = pick.replace(" (download)", "")

    hw = hardware()
    dev_options = ["auto", "cpu"] + (["cuda:0"] if hw.get("cuda_available") else []) + (["mps"] if hw.get("mps_available") else [])
    cfg["device"] = sb.selectbox("Device", dev_options, help=f"auto selects: {resolve_device('auto')}")
    cfg["detector"]["imgsz"] = sb.select_slider("Inference size", [320, 416, 480, 640, 800, 960, 1280], value=cfg["detector"]["imgsz"],
                                                help="Larger finds smaller/distant people but is slower")
    cfg["detector"]["conf"] = sb.slider("Detection confidence", 0.05, 0.9, float(cfg["detector"]["conf"]), 0.05)
    cfg["detector"]["iou"] = sb.slider("NMS IoU", 0.3, 0.9, float(cfg["detector"]["iou"]), 0.05)

    sb.header("Tracking")
    trackers = available_trackers()
    cfg["tracker"]["type"] = sb.selectbox("Tracker", trackers, index=trackers.index(cfg["tracker"]["type"]))
    cfg["tracker"]["track_high_thresh"] = sb.slider("Track high threshold", 0.05, 0.9, float(cfg["tracker"]["track_high_thresh"]), 0.05)
    cfg["tracker"]["new_track_thresh"] = sb.slider("New track threshold", 0.05, 0.95, float(cfg["tracker"]["new_track_thresh"]), 0.05)
    cfg["tracker"]["match_thresh"] = sb.slider("Match threshold", 0.3, 0.99, float(cfg["tracker"]["match_thresh"]), 0.01)
    cfg["tracker"]["track_buffer"] = sb.number_input("Track buffer (frames kept while lost)", 1, 600, int(cfg["tracker"]["track_buffer"]))
    if cfg["tracker"]["type"] in ("botsort", "deepocsort", "tracktrack"):
        cfg["tracker"]["with_reid"] = sb.checkbox("Use appearance features in association", value=False)

    sb.header("Re-identification (optional)")
    cfg["reid"]["enabled"] = sb.checkbox("Assign visual-identity labels", value=False,
                                         help="Body-appearance similarity only. Not identity verification.")
    if cfg["reid"]["enabled"] or cfg["tracker"].get("with_reid"):
        cfg["reid"]["backend"] = sb.selectbox("Embedding backend", ["histogram", "yolo_cls", "torchreid"])
        if cfg["reid"]["backend"] != "histogram":
            cfg["reid"]["weights"] = sb.text_input("Embedding weights", value=cfg["reid"].get("weights") or ("yolo26n-cls.pt" if cfg["reid"]["backend"] == "yolo_cls" else ""))
        cfg["reid"]["similarity_threshold"] = sb.slider("Similarity threshold", 0.3, 0.99, float(cfg["reid"]["similarity_threshold"]), 0.01)

    sb.header("Video processing")
    cfg["video"]["frame_stride"] = sb.number_input("Process every Nth frame", 1, 30, 1)
    rmax = sb.selectbox("Downscale frames (longest side)", ["original", 1920, 1280, 960, 640])
    cfg["video"]["resize_max_side"] = None if rmax == "original" else int(rmax)
    cfg["video"]["max_frames"] = sb.number_input("Max frames (0 = all)", 0, 1_000_000, 0) or None
    pre = sb.selectbox("Environmental preprocessing", ["none", "auto (low-light only)", "clahe", "gamma", "clahe_gamma", "dehaze"])
    cfg["preprocessing"]["enabled"] = pre != "none"
    cfg["preprocessing"]["method"] = "auto" if pre.startswith("auto") else pre
    cfg["output"]["draw_trails"] = sb.checkbox("Draw trails", value=True)
    cfg["output"]["save_crops"] = sb.checkbox("Save person crops", value=False)
    return cfg


def save_upload(uploaded, cfg: dict) -> Path:
    """Write the upload to uploads/ under a random name (the user's filename is never used as a path)."""
    ext = Path(uploaded.name).suffix.lower()
    allowed = cfg["limits"]["allowed_extensions"]
    if ext not in allowed:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}")
    limit = cfg["limits"]["max_upload_mb"] * 1024 * 1024
    if uploaded.size > limit:
        raise ValueError(f"File is {uploaded.size / 1e6:.0f} MB; the limit is {cfg['limits']['max_upload_mb']} MB.")
    if uploaded.size == 0:
        raise ValueError("The uploaded file is empty.")
    UPLOADS.mkdir(exist_ok=True)
    dest = UPLOADS / f"{uuid.uuid4().hex}{ext}"
    with dest.open("wb") as f:
        f.write(uploaded.getbuffer())
    return dest


def run_pipeline(cfg: dict, source, on_frame=None, label: str = "Processing"):
    from src.utils.device import resolve_device
    from src.video.inference_pipeline import InferencePipeline

    device = resolve_device(cfg["device"])
    det = get_detector(cfg["detector"]["weights"], device, cfg["detector"]["imgsz"], cfg["detector"]["conf"],
                       cfg["detector"]["iou"], cfg["detector"].get("allow_download", True))
    bar = st.progress(0.0, text=label)
    status = st.empty()

    def progress(frac, n, info):
        if frac is not None:
            bar.progress(min(frac, 1.0), text=f"{label}: frame {info['frame']} ({frac:.0%})")
        status.caption(f"{n} frames processed, {info['active']} active tracks")

    res = InferencePipeline(cfg, detector=det).run(source, output_dir=OUTPUTS, progress=progress, on_frame=on_frame)
    bar.progress(1.0, text="Done")
    return res


def show_results(res) -> None:
    s, t = res.summary, res.timing
    st.subheader("Results")
    for w in res.warnings:
        st.warning(w)
    c = st.columns(5)
    c[0].metric("Track IDs observed", s["unique_track_ids"], help=s["unique_track_ids_note"])
    c[1].metric("Max concurrent tracks", s["max_concurrent_tracks"])
    c[2].metric("Processing FPS", t["processing_fps"])
    c[3].metric("Processing time", f"{t['wall_time_s']:.1f} s")
    c[4].metric("Mean track confidence", f"{s['mean_track_confidence']:.2f}")
    st.caption("Track IDs are not a count of unique people: ID switches, fragmentation and re-entry change the count.")

    left, right = st.columns([3, 2])
    with left:
        if res.video_path and Path(res.video_path).exists():
            st.video(str(res.video_path))
    with right:
        manager = res.extra.get("manager")
        if manager and manager.active_counts:
            st.markdown("**Active tracks per processed frame**")
            st.line_chart(pd.DataFrame({"active tracks": manager.active_counts, "detections": manager.detection_counts}))
        st.markdown("**Stage timing (ms per frame)**")
        st.json(t["stage_ms_per_frame"], expanded=False)

    data = json.loads(Path(res.json_path).read_text())
    tracks = pd.DataFrame([{k: v for k, v in tr.items() if k != "trajectory"} for tr in data["tracks"]])
    if not tracks.empty:
        st.markdown("**Tracks** (timestamps in seconds from video start)")
        st.dataframe(tracks, use_container_width=True, hide_index=True)
        traj = pd.DataFrame([{"track_id": str(tr["track_id"]), "x": (p["bbox"][0] + p["bbox"][2]) / 2, "y": -p["bbox"][3]}
                             for tr in data["tracks"] for p in tr["trajectory"]])
        st.markdown("**Trajectories** (foot point, image coordinates; y inverted for display)")
        st.scatter_chart(traj, x="x", y="y", color="track_id", height=320)
    with st.expander("Statistics and events"):
        st.json(s)
        st.dataframe(pd.DataFrame(data["events"]), use_container_width=True, hide_index=True)

    d = st.columns(3)
    if res.video_path and Path(res.video_path).exists():
        d[0].download_button("Download annotated video", Path(res.video_path).read_bytes(), "annotated.mp4", "video/mp4")
    d[1].download_button("Download CSV", Path(res.csv_path).read_bytes(), "tracks.csv", "text/csv")
    d[2].download_button("Download JSON", Path(res.json_path).read_bytes(), "tracks.json", "application/json")


def main() -> None:
    _startup()
    st.title("Human detection and tracking")
    st.caption("Pretrained YOLO person detector + multi-object tracker. Runs locally; no video leaves this machine. "
               "No face recognition is performed.")
    try:
        cfg = sidebar_config()
    except ConfigError as exc:
        st.error(str(exc))
        return

    tab_file, tab_cam, tab_about = st.tabs(["Video file", "Camera / stream", "About & limitations"])
    with tab_file:
        up = st.file_uploader("Upload a video", type=[e.lstrip(".") for e in cfg["limits"]["allowed_extensions"]])
        if up is not None:
            key = f"{up.name}:{up.size}"
            if st.session_state.get("upload_key") != key:
                try:
                    st.session_state["upload_path"] = str(save_upload(up, cfg))
                    st.session_state["upload_key"] = key
                    st.session_state.pop("result", None)
                except ValueError as exc:
                    st.error(str(exc))
                    return
            path = Path(st.session_state["upload_path"])
            from src.video.video_reader import VideoError, probe_video

            try:
                meta = probe_video(path, allowed_extensions=cfg["limits"]["allowed_extensions"])
            except VideoError as exc:
                st.error(str(exc))
                return
            c1, c2 = st.columns([3, 2])
            c1.video(str(path))
            c2.markdown("**Video metadata**")
            c2.write({"resolution": f"{meta.width}x{meta.height}", "fps": round(meta.fps, 2), "frames": meta.frame_count,
                      "duration_s": round(meta.duration_s or 0, 2), "codec": meta.codec})
            if c2.button("Run detection and tracking", type="primary"):
                try:
                    st.session_state["result"] = run_pipeline(cfg, path)
                except Exception as exc:  # noqa: BLE001 - show the error instead of a stack trace
                    st.error(f"Processing failed: {exc}")
        if st.session_state.get("result") is not None:
            show_results(st.session_state["result"])

    with tab_cam:
        st.write("Process frames from a webcam (device index) or a stream URL (rtsp/http). Processing stops after the frame limit.")
        src = st.text_input("Camera index or stream URL", value="0")
        n = st.number_input("Frames to process", 10, 100_000, 300)
        if st.button("Start camera"):
            cam_cfg = copy.deepcopy(cfg)
            cam_cfg["video"]["max_frames"] = int(n)
            view = st.empty()

            def on_frame(img, info):
                view.image(img[:, :, ::-1], caption=f"frame {info['frame']}, {info['active']} active tracks")

            try:
                source = int(src) if src.strip().isdigit() else src.strip()
                show_results(run_pipeline(cam_cfg, source, on_frame=on_frame, label="Camera"))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Camera processing failed: {exc}")

    with tab_about:
        st.markdown((PROJECT_ROOT / "docs" / "ABOUT.md").read_text() if (PROJECT_ROOT / "docs" / "ABOUT.md").exists() else "")
        st.json(hardware(), expanded=False)


main()
