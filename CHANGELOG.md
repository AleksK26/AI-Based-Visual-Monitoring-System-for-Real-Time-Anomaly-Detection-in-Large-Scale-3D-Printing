# Changelog

Engineering changes, with notes on what is relevant to the thesis write-up.

## 2026-06-26 — Live-path parity, thin-class data triage, dashboard, repo hygiene

### Inference parity (offline ↔ live)
- **Unified the inference config in `src/detector.py`** as the single source of truth:
  `BASE_CONF`, `IOU_THRESHOLD`, `TEMPORAL_WINDOW`, `TEMPORAL_MIN_HITS`, and the per-class
  `CLASS_THRESHOLDS`. `test_model.py` now imports these instead of re-declaring them.
- **Fixed a drift bug:** `main.py` had been running the old **v5** model at a single flat
  confidence (0.55) with a non-class-aware persistence counter, while the offline tool used
  **v7** + per-class thresholds + a temporal smoother. `main.py` now uses
  `Detector.DEFAULT_MODEL` (v7) and the shared per-class logic, with a class-aware HUD.
- `Detector` now: casts a wide net at `BASE_CONF`, filters per-class, merges fragmented
  same-class boxes (merge-NMS), and applies a **per-class temporal smoother** (a class must
  appear in *N* of the last *M* analysed frames before it can trigger a pause). Exposes
  `class_hits` and `triggered_classes` for the UI/HUD.
- *Thesis relevance:* the per-class thresholds and temporal debouncing described in the
  methodology are now actually exercised by the deployment loop, not just the offline script —
  closes a gap between the "system design" and "what the code runs".

### Data — diverse-real triage for the thin classes
Curated **existing** scraped raw candidates (no new scraping) into tight-labeled training images,
to grow the under-represented classes ahead of a future v8 retrain:

| Class | Diverse-real before → after |
|-------|------------------------------|
| Warping | 30 → **68** |
| Blob_of_death | 26 → **54** |
| Layer_shifting | 12 → **16** |

- Method: vision-assisted triage from indexed contact sheets, with an explicit
  Warping↔Layer_shifting anti-confusion guardrail, then tight per-image boxes written as YOLO
  sidecars and QA'd (`data/scripts/autolabel_qa.py`).
- **Finding worth citing:** the Layer_shifting raw pool is effectively *exhausted* — only 4 new
  non-duplicate, in-domain images existed; the rest were duplicates, diagrams/CAD/text overlays,
  or off-class. This class genuinely needs a fresh scrape with new search terms, whereas Warping
  and Blob had ample candidates. (Reinforces the data-adequacy / domain-gap discussion.)
- *Caveat:* 24 of the 28 new Blob boxes are best-effort central estimates pending a full-res
  refinement pass (Blob has no geometrically-confusable sibling class, so the risk is low).

### Monitoring dashboard (`app.py`, Streamlit)
- Live annotated feed, per-class hit progress bars, printer-state indicator, auto-pause banner,
  and manual **Pause / Resume / Stop** controls — all driven by the same `Detector` as the
  headless loop, so the dashboard and deployment agree.
- Added `resume_print()` / `cancel_print()` to `src/printer_interface.py` (Moonraker + OctoPrint)
  to back the new controls.
- *Thesis relevance:* gives a concrete "human-in-the-loop / product viability" artefact —
  a usable interface over the detection + actuation pipeline.

### Repository hygiene
- Rewrote `.gitignore`: **data and model weights are no longer tracked** (datasets, `runs/`,
  `*.pt`, `*.onnx`); only code, configs, and docs are committed. Untracked the previously
  committed weights and training artifacts.
- Added this `CHANGELOG.md` and a project `README.md`.
