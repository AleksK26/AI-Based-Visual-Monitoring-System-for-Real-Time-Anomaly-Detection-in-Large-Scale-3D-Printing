# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thesis project for detecting 3D printing defects (spaghetti, warping, layer shifting, stringing, off-platform, cracking) using YOLOv8. The end goal is an edge-deployable system running on a Raspberry Pi that monitors a large-format printer bed via webcam and automatically pauses the print via OctoPrint/Moonraker API when a defect is detected.

## Common Commands

```bash
# Install dependencies (none locked in requirements.txt yet)
pip install ultralytics opencv-python rembg torch torchvision tqdm pillow

# Fix label filenames (one-time: renames image.jpg.txt → image.txt)
python fix_labels.py

# Crop printer bed backgrounds interactively from raw photos
python data/scripts/crop_background.py

# Generate synthetic training dataset (~1,000 composite images)
python data/scripts/syntetic_generated_data.py

# Validate dataset directory structure
python directory_check.py

# Train YOLOv8s on the defect dataset
python train.py

# Run inference on test images/video
python test_model.py

# Run the full monitoring pipeline (stub, uses mock camera)
python main.py
```

## Architecture

### Data Pipeline

The core challenge solved here: the Kaggle defect dataset contains macro close-up photos, but the real-world use case is global-view detection from a large printer bed. The synthetic data pipeline bridges this domain gap:

1. `data/3D-Printing-Defect-Dataset/` — Kaggle source images per class
2. `data/scripts/crop_background.py` — Interactive tool to crop clean printer bed photos
3. `data/clean_printerbed/` — Clean background images of the actual printer
4. `data/scripts/syntetic_generated_data.py` — **Core pipeline**: uses `rembg` to remove backgrounds from Kaggle defect images, resizes to 50–80% of frame, composites onto real printer bed backgrounds, writes YOLO-format labels
5. `data/processed/` — Final training dataset (~900 images: ~850 synthetic + ~52 real negative prints)

### Training

`train.py` — Trains YOLOv8s (`yolov8s.pt`) for 35 epochs with mosaic (0.2), mixup (0.1), rotation (5°), and horizontal flip augmentation. Config in `configs/defect_data.yaml`. Outputs saved to `runs/detect/3d_print_monitor/`.

**Known issue**: `configs/defect_data.yaml` currently uses the training set as validation set (`val: images/train`). Real-world accuracy is ~80–90% despite near-perfect training metrics.

### Inference

`test_model.py` — Loads `best.pt`, runs inference with `conf=0.30`, `iou=0.50`, ByteTrack tracking. Reads from `data/real_world_test/`, saves results to `runs/detect/real_world_test/`.

### Application Modules (`src/`)

These are stubs planned for the live monitoring system:
- `detector.py` — Will wrap YOLOv8 inference with a persistence/debounce filter to suppress false positives
- `camera.py` — Will use `cv2.VideoCapture(0)`; currently returns mock frames
- `printer_interface.py` — Will call OctoPrint or Moonraker HTTP API to pause the print
- `datacollector.py` — Will passively save frames during successful prints for future training
- `notifier.py` / `utils.py` — Empty, future alerting logic

`main.py` orchestrates these: camera → detector → notifier → printer interface.

### Trained Models

- `runs/detect/3d_print_monitor/yolov8s_centered_synthetic/weights/best.pt` — Primary trained model
- `runs/detect/3d_print_monitor/yolov8s_centered_synthetic2/weights/best.pt` — Later iteration
- ONNX export available for Raspberry Pi deployment (`best.onnx`, ~44.8 MB)
- `yolov8s.pt` / `yolo26n.pt` — Base pre-trained weights (Small and Nano variants)

### Dataset Classes (6)

`configs/defect_data.yaml` defines: `Spaghetti`, `Warping`, `Layer_shifting`, `Stringing`, `Offplatfrom` (sic), `Cracking`

## Key Implementation Notes

- The `prepare_negatives.py` script stages negative (perfect print) examples into the training set to reduce false positives.
- Label files must be named `<image_stem>.txt` (not `<image_stem>.jpg.txt`). `fix_labels.py` corrects this.
- Synthetic image placement forces defects to the center of the frame to teach spatial context.
- For edge deployment, export to ONNX: `model.export(format='onnx')` after training.
