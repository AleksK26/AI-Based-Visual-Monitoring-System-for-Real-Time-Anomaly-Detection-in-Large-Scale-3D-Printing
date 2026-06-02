"""
Adds negative (no-defect) training samples from three sources:
  1. data/processed/newperfprint/  — pre-labelled perfect prints (52 images)
  2. data/kaggle_new/archive (1)/no_defected/  — clean webcam frames (798 images)
  3. data/hard_negatives/  — images the model previously FP'd on (add more here
     any time the model fires on a clean print in real_world_test)

Negative samples have EMPTY label files, which tells YOLO there are no
defects in those images. This is critical for reducing false positives.

Target negative ratio: ~17% of total dataset (300 negatives vs ~1500 synthetics).

Run AFTER syntetic_generated_data.py and BEFORE split_dataset.py.
"""

import os
import random
import shutil
from pathlib import Path

# --- CONFIGURATION ---
OUTPUT_IMG_DIR = "data/processed/images/train"
OUTPUT_LBL_DIR = "data/processed/labels/train"

# Source 1: hand-prepared perfect prints
NEWPERFPRINT_DIR = "data/processed/newperfprint"

# Source 2: kaggle no-defect webcam frames
KAGGLE_NO_DEFECT_DIR = "data/kaggle_new/archive (1)/no_defected"

# Source 3: hard negatives — real images the model previously false-positived on.
# Add any new FP images here before retraining.
HARD_NEGATIVES_DIR = "data/hard_negatives"

# How many kaggle no-defect images to sample (they have 798 available)
KAGGLE_SAMPLE_COUNT = 250

SEED = 42


def copy_negatives(src_dir, dst_img_dir, dst_lbl_dir, prefix, sample_n=None):
    src = Path(src_dir)
    if not src.exists():
        print(f"  WARNING: source not found — {src_dir}")
        return 0

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in src.iterdir() if f.suffix.lower() in image_exts]

    if sample_n is not None and sample_n < len(images):
        random.seed(SEED)
        images = random.sample(images, sample_n)

    copied = 0
    for i, img_path in enumerate(images):
        dst_img = Path(dst_img_dir) / f"{prefix}_{i}{img_path.suffix}"
        dst_lbl = Path(dst_lbl_dir) / f"{prefix}_{i}.txt"

        shutil.copy2(img_path, dst_img)
        # Empty label file — no bounding boxes = clean image
        dst_lbl.write_text("")
        copied += 1

    return copied


def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

    print("Adding negative (no-defect) samples to training set ...")

    # Source 1: newperfprint (already YOLO-formatted, just copy)
    n1 = copy_negatives(
        NEWPERFPRINT_DIR,
        OUTPUT_IMG_DIR,
        OUTPUT_LBL_DIR,
        prefix="neg_perfprint",
        sample_n=None,  # use all 52
    )
    print(f"  Copied {n1} images from newperfprint/")

    # Source 2: kaggle no_defected
    n2 = copy_negatives(
        KAGGLE_NO_DEFECT_DIR,
        OUTPUT_IMG_DIR,
        OUTPUT_LBL_DIR,
        prefix="neg_kaggle",
        sample_n=KAGGLE_SAMPLE_COUNT,
    )
    print(f"  Copied {n2} images from kaggle no_defected/")

    # Source 3: hard negatives (real FP images — all used, no sampling)
    n3 = copy_negatives(
        HARD_NEGATIVES_DIR,
        OUTPUT_IMG_DIR,
        OUTPUT_LBL_DIR,
        prefix="neg_hard",
        sample_n=None,
    )
    print(f"  Copied {n3} images from hard_negatives/")

    total = n1 + n2 + n3
    print(f"\nTotal negatives added: {total}")
    print("Next step: python split_dataset.py")


if __name__ == "__main__":
    main()
