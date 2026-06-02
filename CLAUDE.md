# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

Thesis project for detecting 3D printing defects (Spaghetti, Warping, Layer_shifting, Stringing, Cracking) using YOLOv8s. The end goal is an edge-deployable system running on a Raspberry Pi that monitors a large-format printer bed via webcam and automatically pauses the print via OctoPrint/Moonraker API when a defect is detected.

---

## Current State (v3 — mixed real + synthetic data)

### Dataset Status — ready to rebuild for v3
- **Processed data cleared** — empty, must run pipeline below
- **5 classes** (Offplatform permanently dropped — no valid source data)
- **v2 model deleted** — v2 had: oversized bounding boxes (scale 30-90%), Spaghetti FP/FN issues, no real defect images in training

### What changed for v3
1. **Mixed data pipeline** — 150 synthetic + 150 real Kaggle images per class (300 total/class)
   - Real images: `add_real_defects.py` copies from `3D-Printing-Defect-Dataset/data/`, labels as near-full-image bbox
   - Synthetics: scale reduced to **10–45%** (was 30–90%) — fixes oversized boxes in video output
2. **Dead code removed** — `prepare_negatives.py`, `data/scripts/fix_labels.py`, `data/scripts/videotoframes.py`
3. **test_model.py improved** — larger font (1.4), thicker boxes (4px), label clamped to never clip off top of frame
4. **Temporal smoother tightened** — TEMPORAL_WINDOW=7, TEMPORAL_MIN_HITS=5 (was 5/3) — reduces flickering
5. **Thresholds updated** — Spaghetti 0.65, Warping 0.50, Layer_shifting 0.50, Stringing 0.40, Cracking 0.55

### v2 results (for reference)
- mAP@50: 0.938 | Precision: 0.947 | Recall: 0.916
- Per-class AP: Cracking 0.995, Layer_shifting 0.994, Warping 0.954, Stringing 0.954, **Spaghetti 0.793** (weakest)
- Remaining issues: Spaghetti FP on clean prints, oversized detection boxes in video, flickering

---

## Known Issues & Open Problems

### 1. Spaghetti FP/FN (Priority: HIGH)
Model fires Spaghetti on clean prints and sometimes misses real ones. Root cause: trained only on synthetic composites — learned "complex texture on bed" rather than actual spaghetti appearance.
**Fix (v3):** Real Kaggle spaghetti images added directly to training via `add_real_defects.py`.

### 2. Bounding boxes too large in video (Priority: HIGH)
Synthetic scale was 30–90% of frame, so model predicted very large boxes.
**Fix (v3):** Synthetic scale reduced to 10–45%.

### 3. Stringing side-angle FN (Priority: MEDIUM)
Fixed in v2 (s3.jpg now detects at 0.65). Monitor in v3.

### 4. `src/detector.py` Persistence Logic Not Used by `test_model.py`
Intentional — `test_model.py` is for offline review. `Detector` class used by `main.py` (live loop).

---

## Full Pipeline — run in order for v3

```bash
# Install dependencies
pip install ultralytics opencv-python rembg torch torchvision tqdm pillow

# STEP 1: 150 synthetic composites per class (10-45% scale, clean bed backgrounds)
python data/scripts/syntetic_generated_data.py

# STEP 2: 150 real Kaggle defect images per class (actual defect texture)
python data/scripts/add_real_defects.py

# STEP 3: ~302 negative (no-defect) samples
python data/scripts/add_negatives.py

# STEP 4: ONE-TIME — 80/20 train/val split (moves files in-place)
python split_dataset.py

# STEP 5: Train
python train.py

# STEP 6: Test on real-world images/videos
python test_model.py
```

---

## Architecture

### Data Pipeline (v3)

```
data/3D-Printing-Defect-Dataset/data/{class}/   ← source defect images (5 classes)
    ├─ rembg → composite onto data/clean_printerbed/   (150 synthetics/class, 10-45% scale)
    └─ direct copy with bbox label 0.5 0.5 0.85 0.85   (150 real/class)
data/kaggle_new/archive (1)/no_defected/         ← 250 clean webcam frames → negatives
data/processed/newperfprint/                     ← 52 perfect prints → negatives
    ↓ split_dataset.py (80/20)
data/processed/images/train/ + val/             ← ~1200 train / ~300 val + negatives
    ↓ train.py
runs/detect/3d_print_monitor/yolov8s_improved_v3/weights/best.pt
```

**Dataset composition per class:**
- 150 synthetic (positional diversity, clean bed context)
- 150 real Kaggle (genuine defect texture/appearance)
- Total: 300/class × 5 = 1500 positives
- ~302 negatives = ~17% negative ratio

### Training

`train.py` — YOLOv8s, 50 epochs, patience=15. Output: `runs/detect/3d_print_monitor/yolov8s_improved_v3/`

### Inference (Offline Testing)

`test_model.py` — temporal smoother: 7-frame window, 5-hit minimum. Font scale 1.4, box thickness 4.
**Re-derive CLASS_THRESHOLDS from val F1 curves after every retrain.**

Current thresholds (v3 — derived from F1-optimal 0.683 + confusion matrix):
- Spaghetti: 0.72 | Warping: 0.65 | Layer_shifting: 0.55 | Stringing: 0.45 | Cracking: 0.60
- Spaghetti + Warping raised above F1-optimal — both show background FP in confusion matrix
- Stringing kept below optimal — FN-prone on side-angle shots, prioritise recall

### Dataset Classes (5)

```
0: Spaghetti
1: Warping
2: Layer_shifting
3: Stringing
4: Cracking
```

### Trained Models

| Model | mAP@50 | Notes |
|---|---|---|
| yolov8s_improved_v2 | 0.938 | Deleted — synthetic only, oversized boxes |
| **yolov8s_improved_v3** | TBD | **Target** — mixed real+synthetic, smaller scale |

---

## Data Sources

| Source | Location | Use |
|---|---|---|
| Kaggle defect dataset | `data/3D-Printing-Defect-Dataset/data/` | Synthetic source + direct real training |
| Clean printer beds | `data/clean_printerbed/` | Synthetic backgrounds (14 images) |
| Perfect prints | `data/processed/newperfprint/` | 52 negatives (keep, do not delete) |
| Kaggle no_defected | `data/kaggle_new/archive (1)/no_defected/` | 250 sampled negatives |
| Kaggle defected | `data/kaggle_new/archive (1)/defected/` | bed_not_stick/leg_broken/no_bottom/no_support — NOT our classes, skip |
| 12GB competition | `data/kaggle_new/train.csv` | 81K frames, binary `has_under_extrusion` label — no bbox annotations, future work |

---

## Key Implementation Notes

- `split_dataset.py` must only be run **once** on a fresh dataset — it moves files in-place
- `rembg` requires `u2netp` model on first run (~170 MB, cached in `~/.u2net/`)
- Export to ONNX after training: `model.export(format='onnx')`
- `train.py` project name is `'3d_print_monitor'` — ultralytics saves to `runs/detect/3d_print_monitor/{name}/`. Do NOT prefix with `runs/detect/`
- Real defect images use bbox `0.5 0.5 0.85 0.85` — defect fills ~85% of Kaggle close-up frame

---

## Next Steps (After v3)

1. Re-derive per-class thresholds from v3 val F1 curves
2. If Spaghetti still FP-prone: raise threshold further or add more negative variety
3. Benchmark on Raspberry Pi with `best.onnx`
4. Wire OctoPrint/Moonraker API in `src/printer_interface.py`
5. Implement `src/datacollector.py` for passive real-print data collection
6. Future: use 12GB competition dataset — sample label=0 frames as negatives, hand-label subset of label=1 frames for bounding boxes
