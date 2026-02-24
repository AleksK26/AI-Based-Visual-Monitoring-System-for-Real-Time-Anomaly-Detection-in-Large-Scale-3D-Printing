# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

Thesis project for detecting 3D printing defects (Spaghetti, Warping, Layer_shifting, Stringing, Offplatfrom, Cracking) using YOLOv8s. The end goal is an edge-deployable system running on a Raspberry Pi that monitors a large-format printer bed via webcam and automatically pauses the print via OctoPrint/Moonraker API when a defect is detected.

---

## Current State (as of last session)

### Best Trained Model
- **Path:** `runs\detect\runs\detect\3d_print_monitor\yolov8s_improved_v12\weights\best.pt`
- **ONNX export:** `runs\detect\runs\detect\3d_print_monitor\yolov8s_improved_v12\weights\best.onnx` (42.7 MB)
- **mAP@50:** 0.9838 | **mAP@50-95:** 0.7949 | **Precision:** 0.9686 | **Recall:** 0.9726
- **Note:** The double `runs\detect\runs\detect\` prefix is a known path artifact from a train.py bug that has since been fixed (see Known Issues below). Future training runs will save to `runs\detect\3d_print_monitor\`.

### What Was Improved (improvements/model-updates branch, now merged to main)
Compared to the original `main` branch:
1. **Validation split fixed** — `configs/defect_data.yaml` now points `val: images/val` (was `val: images/train`, which gave meaningless 99.5% training metrics)
2. **Dataset split script** — `split_dataset.py` creates an 80/20 train/val split from existing data
3. **Synthetic data pipeline improved** — `data/scripts/syntetic_generated_data.py`:
   - Defect placement is now **randomized** across the full bed (was always centered at 0.5, 0.5)
   - Scale randomized **30–90%** of bed (was 50–80%)
   - `rembg` now pre-caches all background-removed defect images at startup (runs once per source image, not 1000 times) — dramatically faster
   - Forces `CPUExecutionProvider` to suppress the missing `cublasLt64_12.dll` CUDA error
   - Uses lighter `u2netp` model instead of `u2net`
   - **Label filenames fixed:** now saves as `synth_Class_N.txt` (was `synth_Class_N.jpg.txt`)
4. **Training improved** — `train.py`:
   - Starts from clean `yolov8s.pt` base weights
   - Added `hsv_h`, `hsv_s`, `hsv_v` augmentation (lighting variation for webcam/Pi deployment)
   - 50 epochs with `patience=15` early stopping
   - `project='3d_print_monitor'` (fixed double-path bug — previously `'runs/detect/3d_print_monitor'`)
5. **Application modules implemented** — `src/` stubs are now functional:
   - `detector.py` — Detector class with persistence/debounce filter (N consecutive frames required before trigger)
   - `camera.py` — Camera class supporting both mock (video file) and live (USB webcam) modes
   - `printer_interface.py` — Mock mode + OctoPrint/Moonraker HTTP stubs ready to uncomment
   - `main.py` — Full monitoring loop wired up (camera → detector → printer interface)

---

## Known Issues & Open Problems

### 1. Stringing False Negative (Priority: HIGH)
The model fails to detect stringing defects on images shot from a **low side angle** with a **bokeh (blurred) background**. Example: `data/real_world_test/s3.jpg` (a print with clear stringing between two pillars, shot from the side at bed level).

**Root cause:** Training data used top-down clean printer bed backgrounds. The side-angle bokeh view is outside the training distribution.

**Fix needed (data, not code):**
- Add real-world stringing images shot from similar camera angles to `data/processed/images/train/` with hand-labelled YOLO `.txt` files
- Add side-view printer photos to `data/clean_printerbed/` so synthetic data can composite onto them
- Re-run `python data/scripts/syntetic_generated_data.py` and retrain

### 2. False Positives on Perfect Prints (Priority: HIGH)
The model occasionally detects defects on clean/perfect prints. Current mitigation: `CONF_THRESHOLD=0.55` in `test_model.py`.

**Fix needed (data):**
- Run `prepare_negatives.py` with more real perfect-print images (currently only ~52 negatives out of ~900 total, ~6% — target 20-30%)
- Collect frames from successful prints using `src/datacollector.py` (stub — needs hardware)

### 3. Model Path Has Double `runs\detect` Prefix
The current best model is at `runs\detect\runs\detect\3d_print_monitor\yolov8s_improved_v12\` because the old `train.py` had `project='runs/detect/3d_print_monitor'` instead of `project='3d_print_monitor'`. Ultralytics prepends `runs/detect/` automatically.

**This is fixed in the current `train.py`** — new training runs will save to `runs\detect\3d_print_monitor\yolov8s_improved_v1\`.

### 4. `src/detector.py` Persistence Logic Not Used by `test_model.py`
`test_model.py` uses `model.track()` (simple batch mode). The `Detector` class with persistence filter is implemented but only used by `main.py` (the live monitoring loop). This is intentional for now — `test_model.py` is for offline review of recorded videos.

### 5. Video Output Was .mkv (Fixed)
`model.track()` saves annotated videos in the same container format as the input. `.mkv` files from the camera work fine — ultralytics handles the codec.

---

## Common Commands

```bash
# Install dependencies
pip install ultralytics opencv-python rembg torch torchvision tqdm pillow

# ONE-TIME: Split existing dataset into 80% train / 20% val
python split_dataset.py

# Crop new printer bed background photos interactively
python data/scripts/crop_background.py

# Regenerate synthetic training dataset (~1,000 composite images)
python data/scripts/syntetic_generated_data.py

# Validate dataset directory structure
python directory_check.py

# Train YOLOv8s on the defect dataset
python train.py

# Run inference on test images/video (batch mode, simple)
python test_model.py

# Run the full live monitoring pipeline (mock camera mode)
python main.py --source data/real_world_test/your_video.mp4

# Run on live webcam (Raspberry Pi)
python main.py --source 0
```

---

## Architecture

### Data Pipeline

The core challenge: the Kaggle defect dataset contains macro close-up photos, but the real-world use case is global-view detection from a large printer bed. The synthetic data pipeline bridges this domain gap:

1. `data/3D-Printing-Defect-Dataset/` — Kaggle source images per class
2. `data/scripts/crop_background.py` — Interactive tool to crop clean printer bed photos
3. `data/clean_printerbed/` — Clean background images of the actual printer (top-down view currently; **needs side-view backgrounds added**)
4. `data/scripts/syntetic_generated_data.py` — Core pipeline: uses `rembg` (u2netp, CPU) to remove backgrounds from Kaggle defect images, resizes to 30–90% of frame, places at random position on bed, composites onto real printer bed backgrounds, writes YOLO-format labels
5. `data/processed/` — Final training dataset (~900 images: ~850 synthetic + ~52 real negative prints)
   - `images/train/` + `labels/train/` — training split (~80%)
   - `images/val/` + `labels/val/` — validation split (~20%, created by `split_dataset.py`)

### Training

`train.py` — Trains YOLOv8s (`yolov8s.pt`) for 50 epochs with:
- Mosaic (0.5), mixup (0.1), rotation (5°), horizontal flip (0.5)
- HSV augmentation (`hsv_h=0.015`, `hsv_s=0.4`, `hsv_v=0.4`) for lighting variation
- Config: `configs/defect_data.yaml`
- Output: `runs/detect/3d_print_monitor/yolov8s_improved_v1/`

### Inference (Offline Testing)

`test_model.py` — Loads `best.pt`, runs `model.track()` with `conf=0.55`, `iou=0.50`, ByteTrack tracking. Reads from `data/real_world_test/`, saves annotated results to `runs/detect/real_world_test{N}/` (auto-incremented).

### Live Monitoring Pipeline (`src/`)

```
main.py
  └── Camera.grab_frame()          # src/camera.py — cv2.VideoCapture, mock or live
        └── Detector.trigger()     # src/detector.py — YOLOv8 + N-frame persistence filter
              └── PrinterInterface.pause_print()  # src/printer_interface.py — OctoPrint/Moonraker HTTP
```

- **`detector.py`** — `Detector` class: runs YOLOv8 `predict()` on each frame, counts consecutive frames with detections, triggers only after `persistence_frames` (default 8) consecutive hits. Call `detector.reset()` after a pause is triggered.
- **`camera.py`** — `Camera` class: `source=0` for USB webcam, `source="path/to/video.mp4"` for mock mode.
- **`printer_interface.py`** — Set env var `PRINTER_MODE=live` to enable real HTTP calls. OctoPrint and Moonraker code blocks are present, commented out — uncomment the relevant one.
- **`datacollector.py`** — Stub. Will save frames during successful prints for future training data collection.

### Trained Models

| Model | Path | mAP@50 | Notes |
|---|---|---|---|
| yolov8s_centered_synthetic | `runs/detect/3d_print_monitor/.../best.pt` | ~0.995* | Original, overfit (val=train) |
| yolov8s_centered_synthetic2 | `runs/detect/3d_print_monitor/.../best.pt` | ~0.995* | Second original iteration |
| **yolov8s_improved_v12** | `runs/detect/runs/detect/3d_print_monitor/.../best.pt` | **0.9838** | **Current best** — real val split |

*Meaningless — validation set was same as training set.

### Dataset Classes (6)

`configs/defect_data.yaml`:
```
0: Spaghetti
1: Warping
2: Layer_shifting
3: Stringing
4: Offplatfrom   ← typo in class name (intentional, matches Kaggle dataset folder)
5: Cracking
```

---

## Next Steps (Planned)

1. **Collect real-world stringing images** from the same side-angle camera view as `s3.jpg` and add them to training with hand-drawn YOLO labels
2. **Add side-view printer bed backgrounds** to `data/clean_printerbed/` so synthetic composites can simulate the side-angle camera position
3. **Collect more perfect print frames** and run `prepare_negatives.py` — goal is 20–30% negative ratio in training set (currently ~6%)
4. **Retrain** after new data is added
5. **Benchmark on Raspberry Pi** — test inference speed with `best.onnx` (42.7 MB), compare YOLOv8s vs YOLOv8n speed/accuracy tradeoff
6. **Implement `src/datacollector.py`** — passive data collection during real successful prints
7. **Wire OctoPrint/Moonraker API** in `src/printer_interface.py` when hardware is available

---

## Key Implementation Notes

- Label files must be `<stem>.txt` (not `<stem>.jpg.txt`). `fix_labels.py` corrects this if needed.
- `split_dataset.py` must only be run **once** on a fresh dataset — it moves files in-place.
- `rembg` background removal requires the `u2netp` model to be downloaded on first run (~170 MB, cached in `~/.u2net/`).
- For edge deployment, export to ONNX after training: `model.export(format='onnx')`.
- `train.py` project name is `'3d_print_monitor'` — ultralytics saves to `runs/detect/3d_print_monitor/{name}/`. Do NOT prefix with `runs/detect/`.
