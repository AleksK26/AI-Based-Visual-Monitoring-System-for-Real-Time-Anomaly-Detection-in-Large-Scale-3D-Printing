"""
One-time utility: splits data/processed/images/train into 80% train / 20% val.
Run this ONCE before retraining. It moves files in-place, so originals are gone from train/.

Usage:
    python split_dataset.py
"""

import os
import shutil
import random

IMAGES_TRAIN = "data/processed/images/train"
LABELS_TRAIN = "data/processed/labels/train"
IMAGES_VAL   = "data/processed/images/val"
LABELS_VAL   = "data/processed/labels/val"

VAL_RATIO = 0.20
SEED = 42

def split():
    random.seed(SEED)

    os.makedirs(IMAGES_VAL, exist_ok=True)
    os.makedirs(LABELS_VAL, exist_ok=True)

    images = [
        f for f in os.listdir(IMAGES_TRAIN)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not images:
        print(f"No images found in {IMAGES_TRAIN}. Nothing to split.")
        return

    random.shuffle(images)
    n_val = max(1, int(len(images) * VAL_RATIO))
    val_images = images[:n_val]
    train_images = images[n_val:]

    moved = 0
    skipped = 0

    for fname in val_images:
        stem = os.path.splitext(fname)[0]
        label_fname = stem + ".txt"

        src_img = os.path.join(IMAGES_TRAIN, fname)
        dst_img = os.path.join(IMAGES_VAL, fname)
        src_lbl = os.path.join(LABELS_TRAIN, label_fname)
        dst_lbl = os.path.join(LABELS_VAL, label_fname)

        shutil.move(src_img, dst_img)

        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
        else:
            # Create empty label for negatives that had no label file
            open(dst_lbl, 'w').close()

        moved += 1

    print(f"Split complete.")
    print(f"  Train: {len(train_images)} images  ->  {IMAGES_TRAIN}")
    print(f"  Val  : {moved} images             ->  {IMAGES_VAL}")
    print(f"\nNow update configs/defect_data.yaml: val: images/val")
    print("Then retrain with: python train.py")

if __name__ == "__main__":
    split()
