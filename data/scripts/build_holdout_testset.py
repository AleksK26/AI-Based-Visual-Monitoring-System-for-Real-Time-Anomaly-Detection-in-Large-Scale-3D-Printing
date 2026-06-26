"""
Build a small, leakage-free held-out test set for qualitative real-world review.

Strategy (decided 2026-06-09):
  - Hold out a few SINGLE-CLASS images per class from data/roboflow_review/ and
    MOVE them out (image + label) so they are excluded from the v6 training merge.
  - Supplement Cracking from data/scraped_raw/cracking/ (raw scrape, never trained).
  - Warping / Layer_shifting: Roboflow has none -> keep existing Reddit .webp,
    optionally add a couple of originals (handled outside this script).

Images are copied into data/real_world_test/ with a class-prefixed name so the
test output is self-documenting (test_model.py iterates that folder flat).

Run with --apply to actually move/copy; default is a dry run.
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RF_IMG = ROOT / "data" / "roboflow_review" / "images"
RF_LBL = ROOT / "data" / "roboflow_review" / "labels"
HOLDOUT_LBL = ROOT / "data" / "roboflow_review" / "_heldout_labels"  # parked, not trained
SCRAPED_CRACK = ROOT / "data" / "scraped_raw" / "cracking"
KAGGLE_SRC = ROOT / "data" / "3D-Printing-Defect-Dataset" / "data"
KAGGLE_HOLDOUT = ROOT / "data" / "_kaggle_heldout"  # parked originals, removed from train pool
TEST_DIR = ROOT / "data" / "real_world_test"

ID_TO_NAME = {0: "Spaghetti", 1: "Warping", 2: "Layer_shifting", 3: "Stringing", 4: "Cracking"}

# How many to pull per class from Roboflow single-class images
PER_CLASS = {0: 4, 3: 4, 4: 3}   # Spaghetti, Stringing, Cracking (Cracking +scrape below)
SCRAPED_CRACK_N = 1
# Warping/Layer_shifting have NO Roboflow source. Pull a few Kaggle originals and
# MOVE them out of the source folder so they leak into neither direct-copy nor
# synthetic-compositing training. {class_name: n}
KAGGLE_PER_CLASS = {"Warping": 3, "Layer_shifting": 3}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

random.seed(42)


def find_image_for_label(stem):
    for ext in IMG_EXTS:
        p = RF_IMG / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def scan():
    """Return {class_id: [(label_path, image_path), ...]} for SINGLE-class images."""
    single = defaultdict(list)
    total = 0
    for lbl in RF_LBL.glob("*.txt"):
        total += 1
        ids = set()
        try:
            for line in lbl.read_text().splitlines():
                line = line.strip()
                if line:
                    ids.add(int(line.split()[0]))
        except Exception:
            continue
        if len(ids) == 1:
            cid = next(iter(ids))
            img = find_image_for_label(lbl.stem)
            if img:
                single[cid].append((lbl, img))
    return single, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually move/copy files")
    args = ap.parse_args()

    single, total = scan()
    print(f"Scanned {total} Roboflow labels.")
    for cid in sorted(single):
        print(f"  class {cid} {ID_TO_NAME.get(cid,'?'):14s}: {len(single[cid])} single-class images")

    picks = []  # (src_img, dest_name, label_to_park)
    for cid, n in PER_CLASS.items():
        pool = sorted(single.get(cid, []), key=lambda t: t[1].name)
        chosen = random.sample(pool, min(n, len(pool)))
        for i, (lbl, img) in enumerate(chosen, 1):
            dest = f"test_{ID_TO_NAME[cid]}_{i:02d}{img.suffix.lower()}"
            picks.append(("roboflow", img, dest, lbl))

    crack_pool = sorted(p for p in SCRAPED_CRACK.glob("*") if p.suffix.lower() in IMG_EXTS)
    for i, img in enumerate(random.sample(crack_pool, min(SCRAPED_CRACK_N, len(crack_pool))),
                            start=PER_CLASS.get(4, 0) + 1):
        picks.append(("scraped", img, f"test_Cracking_{i:02d}{img.suffix.lower()}", None))

    # Warping / Layer_shifting: pull Kaggle originals and move them out of train pool.
    for cls_name, n in KAGGLE_PER_CLASS.items():
        src = KAGGLE_SRC / cls_name
        pool = sorted(p for p in src.glob("*") if p.suffix.lower() in IMG_EXTS)
        chosen = random.sample(pool, min(n, len(pool)))
        for i, img in enumerate(chosen, 1):
            dest = f"test_{cls_name}_{i:02d}{img.suffix.lower()}"
            picks.append(("kaggle", img, dest, None))

    print(f"\n{'APPLYING' if args.apply else 'DRY RUN'} — {len(picks)} test images:")
    if args.apply:
        HOLDOUT_LBL.mkdir(exist_ok=True)
        KAGGLE_HOLDOUT.mkdir(exist_ok=True)
    for source, img, dest, lbl in picks:
        print(f"  [{source:8s}] {img.name}  ->  real_world_test/{dest}")
        if args.apply:
            if source == "kaggle":
                # copy to test, then move original out of Kaggle source -> not trained
                shutil.copy2(img, TEST_DIR / dest)
                shutil.move(str(img), str(KAGGLE_HOLDOUT / f"{img.parent.name}__{img.name}"))
                continue
            shutil.copy2(img, TEST_DIR / dest)
            if source == "roboflow":
                # remove from training pool: move image out, park the label
                shutil.move(str(lbl), str(HOLDOUT_LBL / lbl.name))
                img.unlink()  # image no longer in roboflow_review/images -> not merged to train
    if args.apply:
        print("\nDone. Roboflow hold-outs removed from training pool "
              f"(labels parked in {HOLDOUT_LBL.relative_to(ROOT)}).")
    else:
        print("\nDry run only. Re-run with --apply to execute.")


if __name__ == "__main__":
    main()
