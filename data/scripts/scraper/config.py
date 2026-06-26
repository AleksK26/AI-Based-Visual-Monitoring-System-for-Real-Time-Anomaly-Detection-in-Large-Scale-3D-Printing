"""
Central configuration for the defect-image scraping pipeline.

This is the ONE file to edit when you want to change what the scraper collects.
It defines, per defect class:
  - search_queries : built across a COLOUR PALETTE for diversity (build_queries)
  - definition     : the exact keep/reject prompt the vision model uses (verify.py)
  - output_dir     : where accepted images land (plugs straight into the existing
                     add_real_defects.py / add_negatives.py pipeline folders)
  - min_confidence : reject below this verifier confidence
  - reject_gold    : drop gold/brass/metallic objects (root cause of the v5 bias)
  - max_per_colour : diversity cap — no single filament colour may exceed this many
                     accepted images, so we never trade the gold bias for, say, a
                     "white = warping" bias.

Why diversity matters (the whole point of this scrape): the v5 model learned
"gold texture = defect" because every Kaggle cracking/warping image was the same
gold-object lab setup. Fixing that means colour-DIVERSE real defects across MANY
colours, PLUS gold-but-CLEAN hard negatives. If we over-collect one new colour we
just recreate the same failure with a different hue — hence COLOURS + max_per_colour.

Why IMAGE_SPEC matters: the deployment target is a webcam looking down/across a
large-format printer bed (Elegoo OrangeStorm Giga). Macro studio close-ups and
marketing renders don't resemble that at all, so the spec below steers collection
toward realistic in-situ printer shots and is folded into every verification prompt.
"""

import os
from pathlib import Path

# Repo root = three levels up from this file (data/scripts/scraper/config.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"


def _load_dotenv():
    """Load KEY=VALUE pairs from a .env file into the environment (no dependency).

    Looks for .env at the project root first, then in this scraper folder. Existing
    environment variables win, so a real exported var always overrides the file.
    This is why you can just drop GEMINI_API_KEY into .env and run the scraper.
    """
    for candidate in (REPO_ROOT / ".env", Path(__file__).resolve().parent / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        break  # first .env found wins


_load_dotenv()

# Where raw, unverified candidates are dumped before verification/dedup.
RAW_DIR = DATA_ROOT / "scraped_raw"

# Accepted images are staged HERE for human review — kept SEPARATE from the real
# training folders (data/real_*/ , hard_negatives/). Nothing touches training data
# until you review a class and run promote.py to merge it into its final folder.
STAGING_DIR = DATA_ROOT / "scraped_review"

# --- Harvest scoping (fixes the "1000 garbage jpgs" problem) ---------------
# Hard cap on RAW candidates downloaded per target — harvest stops once hit, so we
# never pull 1000 images for one class. The verifier only needs a few hundred to
# find ~40 good ones.
MAX_RAW_PER_CLASS = 180

# Engines to use, in priority order. Reddit (domain-scoped subreddits) is the most
# PRECISE source for real in-situ failures; Google/Bing/DDG are well-scoped for
# English queries. Baidu is intentionally OFF — it was the source of the NSFW /
# irrelevant noise and adds little for English 3D-printing terms.
ENABLED_ENGINES = ["reddit", "ddg", "google", "bing"]

# Force safe-search on every engine that supports it.
SAFE_SEARCH = True

# ---------------------------------------------------------------------------
# Vision verification model (Google Gemini, via verify.py).
# Defaults to gemini-2.5-flash: strong vision accuracy on the FREE Google AI Studio
# API tier — get a key at aistudio.google.com and set GEMINI_API_KEY. The confidence
# threshold + deterministic gold/colour gates + final human review are the backstops.
# If a --dry-run review shows it confusing cracking<->warping or missing gold,
# escalate with  SCRAPER_VERIFY_MODEL=gemini-2.5-pro  in your environment.
# ---------------------------------------------------------------------------
VERIFY_MODEL = os.environ.get("SCRAPER_VERIFY_MODEL", "gemini-2.5-flash")

DEFAULT_TARGET_PER_CLASS = 40   # accepted images wanted per class before stopping
CANDIDATES_PER_QUERY = 25       # raw candidates harvested per query


# ---------------------------------------------------------------------------
# IMAGE TYPE / SPEC — what a good training image looks like for THIS project.
# Folded into every verification prompt so the vision model enforces it.
# ---------------------------------------------------------------------------
IMAGE_SPEC = (
    "IMAGE REQUIREMENTS (this is for training a webcam-based monitor that watches a "
    "large-format FDM printer bed from above/at an angle):\n"
    "  - Must be a REAL photograph of a real 3D printer or a real 3D-printed part — "
    "NOT a CAD model, NOT a marketing/product render, NOT a diagram or illustration.\n"
    "  - PREFER shots that look like a monitoring camera: the print seen on/above the "
    "build plate, in the context of the printer (bed, nozzle, gantry visible). "
    "Wide-angle or overhead 'whole-bed' views are ideal.\n"
    "  - Macro extreme close-ups of a part on a desk are LOWER value (accept only if the "
    "defect is unmistakable) — mark view='macro_closeup'.\n"
    "  - The defect (or, for negatives, the clean print) must be clearly visible and in "
    "reasonable focus. Reject blurry, tiny, watermarked-stock, collage, or text-overlay images.\n"
    "  - Reject filament-spool-only photos, printer box/marketing shots, and screenshots of UIs."
)


# ---------------------------------------------------------------------------
# COLOUR PALETTE — drives query diversity. Keep this broad on purpose.
# ---------------------------------------------------------------------------
COLOURS = [
    "white", "black", "grey", "silver", "red", "orange", "yellow",
    "green", "blue", "purple", "pink", "brown", "natural translucent",
]
# Colours that count as "gold/metallic" for the reject_gold rule / negatives.
GOLD_METALLIC = {"gold", "brass", "bronze", "metallic", "copper"}


def build_queries(templates, colours=COLOURS, context_queries=()):
    """Expand '{colour}' templates across the palette, then add colour-agnostic
    context queries. Returns a de-duplicated, order-preserving list."""
    out = []
    for tmpl in templates:
        for c in colours:
            out.append(tmpl.format(colour=c))
    out.extend(context_queries)
    seen, deduped = set(), []
    for q in out:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


# ---------------------------------------------------------------------------
# PER-SECTOR NEEDS — grounded in memory (project_state.md / project_next_steps.md).
#
# The deployment camera is a webcam looking down/across a large-format bed, so the
# memory's recurring lesson is: collect WIDE-ANGLE / webcam-above-bed shots, NOT
# macro lab close-ups (the existing real_stringing/ set was gathered this way; the
# excluded Kaggle sets were macro lab towers / gold objects). Memory does not give
# literal angle degrees — it gives this qualitative wide-vs-macro distinction plus
# the per-class data gaps below. Each class therefore carries a `view_pref` (the
# angle that actually shows that defect) and a `lacking` note (what v5 is missing).
#
#   cracking : v5 has 300 SYNTHETIC only, all gold-biased -> ZERO non-gold real.
#   warping  : v5 has 300 SYNTHETIC only, all gold-lab biased -> ZERO non-gold real.
#   off_plat : class was DROPPED in v5 -> NO data at all. Highest-gap new class.
#   blob     : never collected -> NO data at all (future class).
#   stringing: already has 24 wide-angle Reddit images -> NOT collected here.
#   negatives: only 1 gold hard-negative (p.jpg) -> gold-clean is the critical gap.
#
# off_platform and blob_of_death are collected into their own folders but NOT yet
# wired into the YOLO model as classes 5/6 — deliberate separate step (class count
# touches configs/defect_data.yaml, add_real_defects.py, thresholds). cracking and
# warping map to EXISTING classes 4/1 and can be wired in immediately.
# ---------------------------------------------------------------------------
CLASSES = {
    "cracking": {
        "output_dir": DATA_ROOT / "real_cracking",
        "min_confidence": 0.7,   # raised 0.6->0.7 for v7 scrape: cracking<->warping is the verifier's known confusion pair
        "reject_gold": True,
        "max_per_colour": 8,
        "target": 50,
        "lacking": "ZERO non-gold real images in v5 (synthetic-only, gold-biased).",
        "view_pref": (
            "Hairline cracks need to be visible, so ACCEPT medium and close-range shots "
            "that clearly show the crack; whole-bed webcam shots are fine ONLY if the crack "
            "is actually visible. Still reject gold lab macros and product renders."
        ),
        # Colour-agnostic queries (v7 decision): colour diversity enforced at accept
        # stage via max_per_colour, not baked into the searches.
        "search_queries": build_queries(
            templates=[],
            colours=[],
            context_queries=[
                "3d print cracking layer separation",
                "3d print crack splitting walls",
                "3d printed part layer delamination",
                "fdm print cracking large part on bed",
                "3d print layers separating mid print",
                "3d print crack failure printer bed",
                "why is my 3d print cracking",
                "tall 3d print split crack between layers",
            ],
        ),
        "definition": (
            "CRACKING.\n"
            "LOOK FOR: a distinct fracture/split line cutting across the printed layers; layers "
            "visibly pulled apart or delaminated; a gap opening in an otherwise solid wall, often "
            "following the horizontal layer lines.\n"
            "ACCEPT: any non-gold filament colour, any scale, as long as the crack is clearly visible "
            "and is the main subject.\n"
            "REJECT (commonly confused): stringing (thin wispy hairs between parts), spaghetti "
            "(tangled mess of loose extruded filament), warping (a corner lifting but NO fracture), "
            "intentional model gaps/seams/engraved text, post-processing sanding marks; also "
            "gold/brass/metallic test objects, marketing renders, and any image where no crack is "
            "actually visible."
        ),
    },
    "warping": {
        "output_dir": DATA_ROOT / "real_warping",
        "min_confidence": 0.7,   # raised 0.6->0.7 for v7 scrape: this class feeds a broken model class — accepts must be unambiguous
        "reject_gold": True,
        "max_per_colour": 8,
        "target": 50,
        "lacking": "ZERO non-gold real images in v5 (synthetic-only, gold-lab biased).",
        "view_pref": (
            "Warping reads clearly from a distance — PREFER wide-angle / angled / "
            "webcam-above-bed views showing the part's corners lifting off the plate. "
            "Avoid macro detail crops that lose the bed context."
        ),
        # Colour-agnostic queries (v7 decision): do NOT bake colours into searches —
        # colour diversity is enforced at the ACCEPT stage by max_per_colour instead.
        "search_queries": build_queries(
            templates=[],
            colours=[],
            context_queries=[
                "3d print warping corners lifting off bed",
                "3d print warped corner failed print",
                "PLA print curling up at the edges",
                "ABS 3d print warping bed adhesion problem",
                "3d print corner lifting off build plate",
                "3d print warping bed adhesion failure large print",
                "fdm warping whole bed view",
                "why is my 3d print warping",
                "3d print bottom edges not flat warped",
                "first layer warping 3d printer failure",
                # second-wave phrasings (2026-06-11) — first wave yielded 21 keepers from 194;
                # different wording reaches different result pools
                "3d print lifting off bed mid print",
                "print bed adhesion problem corners curling up",
                "petg print warped pulled away from bed",
                "3d printing brim corners peeling up",
                "large 3d print edges lifting bed",
                "3d print base curved not flat bed adhesion",
            ],
        ),
        "definition": (
            "WARPING.\n"
            "LOOK FOR: the print's base corners or edges curled UPWARD away from the build plate; a "
            "visible gap between the bottom of the print and the bed at the edges; or the whole part "
            "bowed so it no longer sits flat. The part is still attached to the bed.\n"
            "ACCEPT: any non-gold colour; wide/angled views that show the lift against the bed are best.\n"
            "REJECT (commonly confused): cracking (a fracture while the part stays flat), OFF-PLATFORM "
            "(part fully detached/toppled — different class), lifted SUPPORTS only, a normal flat "
            "well-adhered print, a part already removed and photographed off the printer; also "
            "gold/brass/metallic lab objects and renders."
        ),
    },
    "layer_shifting": {
        "output_dir": DATA_ROOT / "real_layer_shifting",
        "min_confidence": 0.65,
        "reject_gold": True,     # the whole point: colour-diverse real data, not the gold Kaggle set
        "max_per_colour": 8,
        "target": 50,
        "lacking": "ZERO diverse real images in v7; synthetic + gold-Kaggle only -- structurally the 'next warping'.",
        "view_pref": (
            "Layer shift reads as a horizontal step/offset in the side wall of a print. PREFER "
            "medium or wide views that show the displaced upper portion against the lower portion "
            "(and ideally the bed); the sideways step must be clearly visible. Avoid extreme macros "
            "that lose the part outline."
        ),
        "search_queries": build_queries(
            templates=[],
            colours=[],
            context_queries=[
                "3d print layer shift failure",
                "3d print layer shifting misaligned layers",
                "fdm print layers shifted sideways",
                "3d print layer shift staircase offset",
                "3d printer skipped steps layer shift",
                "3d print shifted layers mid print on bed",
                "why is my 3d print layer shifting",
                "3d print offset layers belt slipped",
                "large 3d print layer shift failed print",
                "3d print misalignment shifted top half",
                "3d print layer shift benchy",
                "3d printer lost steps shifted print",
            ],
        ),
        "definition": (
            "LAYER SHIFTING.\n"
            "LOOK FOR: a horizontal step or offset where part of the print is displaced sideways "
            "relative to the layers below it, breaking the vertical profile into a staircase; the "
            "upper section sits shifted in X or Y from the lower section. The part is otherwise intact.\n"
            "ACCEPT: any non-gold colour; medium/wide views where the sideways offset is clearly visible.\n"
            "REJECT (commonly confused): WARPING (edges curl UP off the bed, no sideways offset), "
            "CRACKING (a fracture/split, layers pulled apart but not offset sideways), normal stepped "
            "model geometry or intentional terraces, spaghetti, gold/brass lab objects, and renders."
        ),
    },
    "off_platform": {
        "output_dir": DATA_ROOT / "real_off_platform",
        "min_confidence": 0.55,
        "reject_gold": False,
        "max_per_colour": 8,
        "target": 40,
        "lacking": "NO data at all — class was dropped in v5. Highest-gap new class.",
        "view_pref": (
            "Detachment is a whole-bed event — PREFER wide-angle / overhead / webcam "
            "views showing the part loose, on its side, or off the plate in the printer "
            "context. Reject off-printer studio photos of the finished part."
        ),
        "search_queries": build_queries(
            templates=[
                "{colour} 3d print detached from bed failed",
                "{colour} 3d print came off build plate",
            ],
            context_queries=[
                "3d print knocked off bed failure",
                "3d printed part fell off plate mid print",
                "3d print detached dragging around bed nozzle",
            ],
        ),
        "definition": (
            "OFF-PLATFORM / DETACHED.\n"
            "LOOK FOR: the print no longer anchored to the build plate — toppled on its side, shifted/"
            "dragged out of its original footprint, being shoved around by the moving nozzle, or fully "
            "off the bed. The key signal is LOSS OF ATTACHMENT, seen inside the printer.\n"
            "ACCEPT: any colour; wide/overhead views of the loose part in the printer are best.\n"
            "REJECT (commonly confused): WARPING (edges lift but the part is STILL stuck to the bed — "
            "different class), a print still printing normally and attached, a finished part sitting on "
            "a desk or held in a hand off the printer, renders."
        ),
    },
    "blob_of_death": {
        "output_dir": DATA_ROOT / "real_blob_of_death",
        "min_confidence": 0.55,
        "reject_gold": False,
        "max_per_colour": 8,
        "target": 30,
        "lacking": "NO data at all — never collected (future class).",
        "view_pref": (
            "The blob is centred on the print head — PREFER medium shots of the "
            "hotend / nozzle / gantry where the blob engulfing the head is clearly "
            "visible. Bed-only wide shots usually won't show it; reject those."
        ),
        "search_queries": build_queries(
            templates=[
                "{colour} 3d printer blob of death hotend",
                "{colour} filament blob around nozzle 3d printer",
            ],
            context_queries=[
                "3d printer blob of death molten plastic",
                "3d printer hotend encased in filament blob",
                "blob of death 3d printing failure print head",
            ],
        ),
        "definition": (
            "BLOB OF DEATH.\n"
            "LOOK FOR: a large solidified glob of filament built up around and ENGULFING the hotend / "
            "nozzle / heater block / print head — the head is encased in a lump of plastic.\n"
            "ACCEPT: any colour; medium shots where the blob on the print head is the clear subject.\n"
            "REJECT (commonly confused): minor ooze or a few strings on an otherwise visible nozzle, "
            "a clean nozzle, SPAGHETTI lying on the BED rather than on the head, tangled filament on "
            "the spool, renders, non-printer objects."
        ),
    },
}


# ---------------------------------------------------------------------------
# Hard negatives — clean (defect-free) prints. TWO groups, both diversified:
#   gold_clean    : gold/metallic CLEAN prints — directly attacks "gold = defect".
#   diverse_clean : CLEAN prints across the FULL colour palette — so the model
#                   doesn't learn any *other* colour means "defect" either.
# Both land in the existing hard_negatives/ folder (empty labels via add_negatives.py).
# ---------------------------------------------------------------------------
HARD_NEGATIVES = {
    "gold_clean": {
        "output_dir": DATA_ROOT / "hard_negatives",
        "min_confidence": 0.55,
        "must_be_gold": True,            # keep ONLY gold/metallic
        "max_per_colour": 99,            # all "gold-ish", cap not meaningful here
        "target": 30,
        "view_pref": "Any clear view of the clean gold/metallic print is fine.",
        "search_queries": build_queries(
            templates=[],
            colours=[],
            context_queries=[
                "gold PLA 3d print finished clean successful",
                "brass coloured 3d print successful quality",
                "metallic gold filament 3d print good quality on bed",
                "bronze 3d printed model successful print",
                "shiny copper 3d print clean finish",
            ],
        ),
        "definition": (
            "A SUCCESSFUL, defect-free 3D print in GOLD, brass, bronze, copper, or metallic "
            "filament — clean and complete, no cracks, warping, spaghetti, or blobs. ACCEPT only "
            "gold/metallic CLEAN prints. REJECT: any visible defect, non-metallic colours, "
            "raw-spool photos, renders."
        ),
    },
    "diverse_clean": {
        "output_dir": DATA_ROOT / "hard_negatives",
        "min_confidence": 0.55,
        "must_be_gold": False,
        "max_per_colour": 5,             # spread across many colours
        "target": 40,
        "view_pref": "Prefer prints shown on the bed in the printer context.",
        "search_queries": build_queries(
            templates=[
                "{colour} 3d print successful clean finish on bed",
                "{colour} PLA 3d print good quality complete",
            ],
            context_queries=[
                "successful 3d print on printer bed webcam view",
                "clean finished 3d print large format printer",
            ],
        ),
        "definition": (
            "A SUCCESSFUL, defect-free 3D print of ANY colour — clean and complete, no cracks, "
            "warping, spaghetti, detachment, or blobs, sitting normally on the printer bed. "
            "ACCEPT clean prints of any colour. REJECT: any visible defect, raw-spool photos, "
            "marketing renders, non-print objects."
        ),
    },
}


def all_targets():
    """Yield (name, spec, is_hard_negative) for every collection target."""
    for name, spec in CLASSES.items():
        yield name, spec, False
    for name, spec in HARD_NEGATIVES.items():
        yield name, spec, True
