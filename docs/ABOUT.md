### What this does

1. **Detects** people in every processed frame with a pretrained Ultralytics YOLO model (person class only).
2. **Tracks** them with a multi-object tracker (ByteTrack by default; BoT-SORT, OC-SORT, Deep OC-SORT and others are selectable). A *track ID* means "the tracker believes this is the same person instance in this video".
3. Optionally assigns **visual-identity labels** (`V-0001`, ...) from body-appearance embeddings. A label means "looks similar above a threshold", not a verified identity.

### Privacy and responsible use

- All processing is local. Video frames are never uploaded anywhere; Ultralytics usage analytics are switched off for this process.
- No face detection or face recognition is performed. Re-identification uses whole-body appearance (mostly clothing) only.
- Nothing in the outputs attributes an identity to a named individual. Do not use visual-identity labels as proof that two observations are the same real person.
- Uploaded videos are deleted after `retention.uploads_hours` (default 24 h) and generated outputs after `retention.outputs_days` (default 7 days), applied when the app starts or with `python cli.py cleanup`.
- The app has no user accounts. Bind it to localhost (the default) or put it behind authentication before sharing it with other users.

### Known limitations

- **Counts:** the number of track IDs is not the number of people. ID switches, fragmentation after long occlusions, and people re-entering the frame all change it.
- **Long occlusions / re-entry:** IDs are kept only while the tracker can associate the person (`tracker.track_buffer`). After that a new ID is issued. ReID labels may reconnect them but can also merge different people dressed alike.
- **Hard conditions:** accuracy drops in very low light, heavy fog/rain, strong backlight, and for very small people. See `reports/benchmark_report.md` for measured drops under synthetic corruptions. Enhancement preprocessing is optional because it does not reliably help.
- **Thermal/IR video** is a different modality; the RGB model is not validated on it.
- **Mannequins, posters, screens and reflections** can be detected as people. Fine-tuning with such negatives helps but is not included in the pretrained model.
- **Not for high-stakes decisions.** This system has not been independently validated for safety-critical, legal, employment, or law-enforcement use.
