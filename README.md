# 3D Printing Defect Monitor

Real-time computer-vision monitoring for large-format FDM 3D printing. A webcam watches the
print bed; a YOLO model detects defects as they form; on a *confirmed* defect the system pauses
the print over the printer's network API. Built as a thesis project, targeting an edge
deployment (Raspberry Pi + Elegoo OrangeStorm Giga / Klipper).

> **Note on contents:** this repository holds **code, configuration, and documentation only.**
> Datasets and trained model weights are intentionally **not committed** (see
> [What's *not* here](#whats-not-in-this-repo)).

---

## Detected defect classes

| ID | Class | |
|----|-------|---|
| 0 | Spaghetti | extrusion detached from the print, tangled filament |
| 1 | Warping | corners/edges lifting off the bed |
| 2 | Layer_shifting | layers offset sideways (stair-step) |
| 3 | Stringing | fine threads between print regions |
| 4 | Cracking | layer-adhesion cracks/splits |
| 5 | Blob_of_death | molten plastic mass encasing the hotend |

## How it works

```
 Camera ──frames──▶ Detector ──per-class thresholds──▶ Temporal smoother ──▶ Printer
 (webcam/video)     (YOLO11s)   + merge-NMS              (N of last M frames)   (pause/stop)
```

- **`src/detector.py`** — the single source of truth for inference: a low base confidence is
  cast wide, then **per-class thresholds** filter, fragmented same-class boxes are merged
  (merge-NMS), and a **per-class temporal smoother** (a class must appear in *N* of the last
  *M* analysed frames) debounces single-frame false positives before any pause.
- **`src/printer_interface.py`** — Moonraker (Klipper, default) and OctoPrint back-ends for
  `is_printing` / `pause` / `resume` / `cancel`, configured by environment variables. Mock mode
  by default, so the whole pipeline runs with no hardware.
- **`app.py`** — Streamlit dashboard: live annotated feed, per-class detection status, and manual
  pause/resume/stop controls.
- **`main.py`** — headless loop for the edge device (same detector + printer).
- **`test_model.py`** — offline batch inference over images/videos for review (imports the same
  thresholds as the live path, so offline and live results agree).

## Repository layout

```
app.py                  Streamlit monitoring dashboard
main.py                 Headless edge monitoring loop
test_model.py           Offline image/video inference for review
train.py                Training entry point (YOLO)
build_dataset.py        Deterministic dataset assembly + 80/20 split
split_dataset.py        One-time train/val split
derive_thresholds.py    Per-class F1-optimal threshold finder
compare_models.py       Architecture comparison helper
src/                    detector, camera, printer_interface, (datacollector, notifier stubs)
configs/                dataset YAML
data/scripts/           data pipeline: synthetic compositing, real-image import,
                        negatives, autolabel+QA, and the image scraper (data/scripts/scraper/)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Dashboard (recommended):**
```bash
streamlit run app.py
# Live printer control (Klipper/Moonraker):
PRINTER_MODE=live PRINTER_URL=http://<printer-ip>:7125 streamlit run app.py
```

**Headless monitor (edge device):**
```bash
python main.py --source 0                 # webcam index 0
python main.py --source path/to/clip.mp4  # video file
```

**Offline review:**
```bash
python test_model.py     # processes data/real_world_test/ → runs/detect/real_world_test*/
```

**Rebuild the dataset & train** (requires the datasets — see below):
```bash
python build_dataset.py    # assemble + split
python train.py            # train YOLO; weights land in runs/detect/3d_print_monitor/<name>/
```

Printer control is configured via env vars: `PRINTER_MODE` (`mock`|`live`),
`PRINTER_FIRMWARE` (`moonraker`|`octoprint`), `PRINTER_URL`, `PRINTER_API_KEY`.

## Model status

Current shipped model: **YOLO11s**, 6 classes — **validation mAP@50 ≈ 0.887** *(in-distribution;
out-of-domain performance is not yet field-validated)*. Mixed real + synthetic training data with
per-class threshold tuning. See `comparison_v8s_vs_11s.md` and `data_adequacy_assessment.md` for the
architecture comparison and data audit.

## What's *not* in this repo

To keep the repository lean and avoid publishing datasets, these are git-ignored and live locally:

- **`data/`** — all datasets, scraped images, labels, synthetic composites (only `data/scripts/`
  pipeline code is tracked).
- **Model weights** — `*.pt`, `*.onnx` and the entire `runs/` directory (training artifacts +
  weights). Trained models are reproduced by running the data pipeline + `train.py`.

## Roadmap

- v8 retrain on the expanded diverse-real dataset (needs GPU).
- On-printer validation: capture real defects via the deployment camera, hold out a true test set.
- Raspberry Pi latency benchmark (ONNX / NCNN).
- Live OctoPrint/Moonraker field run; passive data collection (`src/datacollector.py`).

---

*Thesis project — Sapienza University of Rome.*
