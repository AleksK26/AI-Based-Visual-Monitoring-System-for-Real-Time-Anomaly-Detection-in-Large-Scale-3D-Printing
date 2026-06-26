# Defect-image scraping pipeline

Automated collection of **colour-diverse, real** 3D-printing defect images to fix the
v5 model's biases (gold→Cracking, lab-framing→Warping) and to gather the classes v5
never had data for (Off-Platform, Blob of Death) plus gold-but-clean hard negatives.

It is a 4-stage pipeline with **one** intelligent step (vision verification); the rest
is plumbing:

```
harvest.py   multi-engine image search  →  data/scraped_raw/<target>/
verify.py    Gemini vision keep/reject   →  structured judgement per image
run.py       deterministic gating        →  copies accepted into data/real_*/...
dedup.py     perceptual de-duplication   →  removes near-duplicate reposts
[ you ]      final review by eye         →  delete any obvious mistakes
```

## Why it's built this way

- **Multi-source harvest** (DuckDuckGo + Google + Bing + Baidu + optional Reddit) so we
  aren't limited by one engine's recall or quirks. Cross-engine duplicates collapse via
  content hashing. Reddit (r/3Dprinting, r/FixMyPrint, …) is the best source of in-situ
  "webcam-above-bed" failure photos — exactly the view our deployment sees.
- **Vision verification is the whole point.** A search for "warping" returns wrong-colour,
  wrong-defect, macro-lab, render, and meme images. Gemini vision classifies each against
  an explicit per-class definition + image spec and reports class / colour / view / a gold
  flag / confidence.
- **Diversity is enforced, not hoped for.** Queries fan out across a 13-colour palette, and
  `max_per_colour` caps each filament colour so we don't trade the gold bias for a new
  "white = warping" bias. The run prints the colour + view distribution so you can see it.
- **Image spec matches deployment.** Prefers real printer / whole-bed / webcam-style shots
  over studio macros and marketing renders (see `IMAGE_SPEC` in `config.py`). Each class
  also has a `view_pref` for the angle that actually shows that defect.

## Install

```bash
pip install google-genai ddgs icrawler imagehash requests pillow
# optional Reddit source:
pip install praw
```

## Configure

Verification uses **Gemini 2.5 Flash** on Google AI Studio's free API tier. Get a
key at https://aistudio.google.com (free; separate from any Gemini app subscription).

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY = "AIza..."
# optional, escalate to the stronger model for borderline classes:
$env:SCRAPER_VERIFY_MODEL = "gemini-2.5-pro"
# optional Reddit:
$env:REDDIT_CLIENT_ID = "..."; $env:REDDIT_CLIENT_SECRET = "..."
```

## ⚠️ Read first: scraping is the SUPPLEMENT, not the primary source

Labelled, YOLOv8-ready datasets exist (Roboflow Universe #2 syLucauc, #3 Failure
Detector, #4 HCMUT — bounding-boxed, side/top view, ~16k images covering spaghetti /
stringing / cracks / blobs). Use those as the **primary** source via
`data/scripts/import_roboflow.py` — no garbage, no NSFW, no manual labelling.

Use this scraper **only for gaps the Roboflow sets don't cover** — chiefly
**off-platform / detached-from-bed**, and colour-diverse warping or gold-clean hard
negatives if those are thin. Harvesting image search for everything wastes time on junk.

## Run

```bash
cd data/scripts/scraper

python run.py --only off_platform      # target a specific gap
python run.py --dry-run --only cracking # harvest+verify+report, don't stage anything
python run.py --skip-harvest           # re-verify already-harvested raw/ (saves time)
python run.py                          # all targets
```

Recall is now scoped to avoid the "1000 garbage jpgs" problem:
- **hard cap** of `MAX_RAW_PER_CLASS` (180) raw candidates per class — harvest stops there
- **safe-search on**, and **Baidu disabled** (it was the NSFW/irrelevant source)
- engines scoped to Reddit + DuckDuckGo + Google + Bing (`ENABLED_ENGINES` in config.py)

### Output is staged separately, then promoted

Accepted images go to **`data/scraped_review/<target>/`** — NOT into the training
folders. Review by eye, delete mistakes, then merge the keepers in:

```bash
python promote.py --only off_platform        # copy reviewed keepers -> data/real_off_platform/
python promote.py --only off_platform --move  # move instead (empties staging)
```

`promote.py` is the human gate — nothing reaches `data/real_*/` or `hard_negatives/`
until you've looked at it.

## Targets & gaps (from project memory)

| Target | Output | Maps to | v5 gap |
|---|---|---|---|
| cracking | `data/real_cracking/` | class 4 | ZERO non-gold real (synthetic-only) |
| warping | `data/real_warping/` | class 1 | ZERO non-gold real (synthetic-only) |
| off_platform | `data/real_off_platform/` | new class 5 (not wired yet) | NO data — dropped in v5 |
| blob_of_death | `data/real_blob_of_death/` | future class (not wired yet) | NO data — never collected |
| gold_clean | `data/hard_negatives/` | negative | only 1 gold negative (p.jpg) |
| diverse_clean | `data/hard_negatives/` | negative | broaden clean-colour coverage |

`off_platform` and `blob_of_death` are **collected but not yet wired into the model** as
classes 5/6 — that changes the class count (touches `configs/defect_data.yaml`,
`add_real_defects.py`, thresholds) and is a deliberate separate step once we see how much
data we actually get. `cracking`/`warping` map to existing classes and can be wired in now.

## After scraping

1. **Review the output folders by eye** — delete obvious mistakes. Bad labels are what
   biased v5; this human pass is not optional.
2. Wire `real_cracking/` and `real_warping/` into `add_real_defects.py` (class 4 / 1).
3. Run the v6 pipeline (`syntetic_generated_data.py` → `add_real_defects.py` →
   `add_negatives.py` → `split_dataset.py`) then train both YOLOv8s and YOLO11s
   (`python train.py --model yolov8s.pt` and `--model yolo11s.pt`) and compare.

## Legal / ethical note

Small-scale academic research collection (thesis). Every harvested image's origin is
logged in `data/scraped_raw/<target>/sources.jsonl`. The scraped set is **not**
redistributed. Where a properly-licensed dataset exists (Roboflow Universe, Kaggle), prefer
downloading that over scraping — cleaner provenance for the thesis. Treat the scrape as a
fallback / supplement, and document it as a methodology limitation.
