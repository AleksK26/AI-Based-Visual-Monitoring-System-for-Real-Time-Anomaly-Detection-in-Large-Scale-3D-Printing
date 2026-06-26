"""
Adds real defect images directly to the training set.

These are real photos of actual defects — they teach the model what each
defect really looks like, which pure synthetic composites lack.
Synthetics handle positional/scale diversity on a printer bed; real images
handle genuine defect texture and appearance.

Label format: near-full-image bounding box centered at (0.5, 0.5) with
width/height 0.85 — works for both macro close-ups and wide-angle shots
where the defect is distributed across the frame.

Run AFTER syntetic_generated_data.py and BEFORE add_negatives.py.
"""

import os
import random
import shutil
from pathlib import Path

# --- CONFIGURATION ---
OUTPUT_IMG_DIR = "data/processed/images/train"
OUTPUT_LBL_DIR = "data/processed/labels/train"
RAW_DATA_ROOT  = "data/3D-Printing-Defect-Dataset/data"

SAMPLES_PER_CLASS = 150
SEED = 42

# The Kaggle Warping/Cracking/Stringing folders are EXCLUDED from direct real-image
# training because they come from a homogeneous lab setup (same printer, same yellow
# warning label, same gold/brass objects, white background) or are macro lab towers.
# Adding them directly taught v5 "gold object = Cracking" / "yellow label = Warping"
# rather than actual defect appearance.
#
# v6 FIX: Warping and Cracking now get COLOUR-DIVERSE real images from the scraper
# (data/scripts/scraper/), which land in data/real_warping/ and data/real_cracking/.
# Those break the colour bias the Kaggle sets caused. The Kaggle classes still
# contribute via syntetic_generated_data.py (rembg strips the biased background).
#
# v6: Spaghetti is now covered abundantly by the Roboflow real datasets with TIGHT
# boxes (~12.5k boxes). The loose 0.5/0.5/0.85/0.85 Kaggle Spaghetti boxes would
# conflict with those tight boxes and soften localization, so Spaghetti is dropped
# here and sourced from Roboflow instead. Layer_shifting has NO Roboflow source, so
# its Kaggle close-ups stay.
# Must match configs/defect_data.yaml exactly for all included classes.
CLASS_MAP = {
    "Layer_shifting": 2,
}

# Real stringing images sourced from Reddit — wide-angle, diverse printers/beds.
# NOT from the Kaggle Stringing folder (wrong type — lab test towers).
# Take ALL available since the dataset is small (~24 images).
REAL_STRINGING_DIR = "data/real_stringing"
STRINGING_CLASS_ID = 3

# Scraped colour-diverse real defects (data/scripts/scraper/). These REPLACE the
# excluded gold-biased Kaggle sets and are the v6 fix for the "gold = defect" bias.
# (key, src_dir, class_id) — class ids must match configs/defect_data.yaml.
SCRAPED_REAL_DEFECTS = [
    ("Cracking",       "data/real_cracking",        4),
    ("Warping",        "data/real_warping",         1),
    ("Blob_of_death",  "data/real_blob_of_death",   5),
    ("Layer_shifting", "data/real_layer_shifting",  2),  # v8: colour-diverse real layer-shift (was gold-only)
]

# The defect fills ~85% of the close-up frame — approximate full-image bbox.
# Also reasonable for wide-angle stringing shots where threads span the print.
BBOX = "0.5 0.5 0.85 0.85"

ACCEPTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def add_class_images(src_dir, class_name, class_id, max_samples, output_img_dir, output_lbl_dir):
    """Copy up to max_samples images from src_dir into the training set.

    If an image carries a sidecar YOLO label (<stem>.txt next to it), that TIGHT
    auto/hand label is used verbatim — this is the v7 fix for Warping/Cracking/Blob,
    which were previously force-boxed at the loose 0.85 full frame (the warping box
    overlapped Layer_shifting geometry and broke the class). Images with no sidecar
    fall back to the legacy near-full-frame BBOX — fine for wide-angle Stringing and
    the Kaggle Layer_shifting close-ups where the defect spans the frame.
    """
    images = [f for f in Path(src_dir).iterdir()
              if f.suffix.lower() in ACCEPTED_EXTENSIONS]

    if not images:
        print(f"  WARNING: no images found in {src_dir}")
        return 0

    n = min(max_samples, len(images))
    sampled = random.sample(images, n)

    sidecar_used = 0
    for i, img_path in enumerate(sampled):
        dst_img = Path(output_img_dir) / f"real_{class_name}_{i}{img_path.suffix}"
        dst_lbl = Path(output_lbl_dir) / f"real_{class_name}_{i}.txt"
        shutil.copy2(img_path, dst_img)

        sidecar = img_path.with_suffix(".txt")
        if sidecar.exists() and sidecar.read_text().strip():
            dst_lbl.write_text(sidecar.read_text())  # tight per-image label
            sidecar_used += 1
        else:
            dst_lbl.write_text(f"{class_id} {BBOX}\n")  # legacy full-frame fallback

    tag = f"{sidecar_used} tight sidecar labels" if sidecar_used else "full-frame fallback"
    print(f"  {class_name}: {n}/{len(images)} images added (class {class_id}) [{tag}]")
    return n


def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

    random.seed(SEED)
    total = 0

    # Kaggle close-up classes
    for class_name, class_id in CLASS_MAP.items():
        src_dir = Path(RAW_DATA_ROOT) / class_name
        if not src_dir.exists():
            print(f"  WARNING: {src_dir} not found — skipping {class_name}")
            continue
        total += add_class_images(src_dir, class_name, class_id, SAMPLES_PER_CLASS,
                                   OUTPUT_IMG_DIR, OUTPUT_LBL_DIR)

    # Real stringing — wide-angle Reddit images (webp + jpg), take all available
    stringing_dir = Path(REAL_STRINGING_DIR)
    if stringing_dir.exists():
        total += add_class_images(stringing_dir, "Stringing", STRINGING_CLASS_ID,
                                   9999, OUTPUT_IMG_DIR, OUTPUT_LBL_DIR)
    else:
        print(f"  WARNING: {REAL_STRINGING_DIR} not found — Stringing gets no real images")

    # Scraped colour-diverse Cracking / Warping — take up to SAMPLES_PER_CLASS each.
    for cls_name, src_dir, class_id in SCRAPED_REAL_DEFECTS:
        d = Path(src_dir)
        if d.exists() and any(p.suffix.lower() in ACCEPTED_EXTENSIONS for p in d.iterdir()):
            total += add_class_images(d, cls_name, class_id, SAMPLES_PER_CLASS,
                                      OUTPUT_IMG_DIR, OUTPUT_LBL_DIR)
        else:
            print(f"  NOTE: {src_dir} empty/missing — {cls_name} stays synthetic-only "
                  f"(run data/scripts/scraper/run.py to populate it)")

    print(f"\nTotal real defect images added: {total}")
    print("Next step: python data/scripts/add_negatives.py")


if __name__ == "__main__":
    main()
