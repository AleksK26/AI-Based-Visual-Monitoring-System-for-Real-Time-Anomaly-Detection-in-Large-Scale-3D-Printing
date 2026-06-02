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

# Warping is EXCLUDED from direct real-image training.
# The Kaggle dataset comes from a homogeneous experimental setup (same printer,
# same yellow warning label, same gold objects, same white background). Adding
# it directly teaches the model "yellow label + fan = defect" rather than
# actual defect appearance. Warping still contributes through
# syntetic_generated_data.py where rembg strips the biased background away.
#
# Cracking is also EXCLUDED — same Kaggle lab setup, all images show the same
# gold/brass colored test object being printed. Model learned "gold object =
# Cracking" rather than actual crack appearance. Use synthetic only.
#
# Stringing Kaggle dataset is also EXCLUDED — all images are macro close-ups
# of small lab test towers, not webcam-above-bed shots. Use data/real_stringing/
# instead (wide-angle Reddit images collected specifically for this use case).
#
# Must match configs/defect_data.yaml exactly for all included classes.
CLASS_MAP = {
    "Spaghetti":      0,
    "Layer_shifting": 2,
}

# Real stringing images sourced from Reddit — wide-angle, diverse printers/beds.
# NOT from the Kaggle Stringing folder (wrong type — lab test towers).
# Take ALL available since the dataset is small (~24 images).
REAL_STRINGING_DIR = "data/real_stringing"
STRINGING_CLASS_ID = 3

# The defect fills ~85% of the close-up frame — approximate full-image bbox.
# Also reasonable for wide-angle stringing shots where threads span the print.
BBOX = "0.5 0.5 0.85 0.85"

ACCEPTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def add_class_images(src_dir, class_name, class_id, max_samples, output_img_dir, output_lbl_dir):
    """Copy up to max_samples images from src_dir into the training set."""
    images = [f for f in Path(src_dir).iterdir()
              if f.suffix.lower() in ACCEPTED_EXTENSIONS]

    if not images:
        print(f"  WARNING: no images found in {src_dir}")
        return 0

    n = min(max_samples, len(images))
    sampled = random.sample(images, n)

    for i, img_path in enumerate(sampled):
        dst_img = Path(output_img_dir) / f"real_{class_name}_{i}{img_path.suffix}"
        dst_lbl = Path(output_lbl_dir) / f"real_{class_name}_{i}.txt"
        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text(f"{class_id} {BBOX}\n")

    print(f"  {class_name}: {n}/{len(images)} images added (class {class_id})")
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

    print(f"\nTotal real defect images added: {total}")
    print("Next step: python data/scripts/add_negatives.py")


if __name__ == "__main__":
    main()
