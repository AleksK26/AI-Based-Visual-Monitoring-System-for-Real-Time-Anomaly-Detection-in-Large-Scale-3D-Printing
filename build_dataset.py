"""
Deterministic v7 dataset builder — one command, fully reproducible.

Replaces the old manual "copy roboflow_review into processed, then run the three
add_* scripts by hand" recipe (which CLAUDE.md never documented). Assembles the
6-class training pool from every source, in order, then splits 80/20:

  1. WIPE   data/processed/{images,labels}/{train,val}
  2. ROBOFLOW  data/roboflow_review/ -> processed/train  (tight boxes, classes 0/3/4)
               with a Spaghetti cap to fix the 69%-of-pool imbalance
  3. SYNTH   syntetic_generated_data.py  (composites incl. new Blob_of_death)
  4. REAL    add_real_defects.py         (tight sidecar labels for warp/crack/blob)
  5. NEG     add_negatives.py            (un-capped, holdout-leakage-guarded)
  6. SPLIT   split_dataset.py            (80/20, moves files in place)
  7. REPORT  per-class image/box counts + negative ratio

Usage:
  python build_dataset.py                  # full rebuild
  python build_dataset.py --spaghetti-cap 2000
  python build_dataset.py --report-only    # print current pool stats, no rebuild
"""

import argparse
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
IMG_TRAIN, LBL_TRAIN = PROC / "images" / "train", PROC / "labels" / "train"
IMG_VAL, LBL_VAL = PROC / "images" / "val", PROC / "labels" / "val"
ROBOFLOW = ROOT / "data" / "roboflow_review"

CLASS_NAMES = {0: "Spaghetti", 1: "Warping", 2: "Layer_shifting",
               3: "Stringing", 4: "Cracking", 5: "Blob_of_death"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SEED = 42


def wipe():
    for d in (IMG_TRAIN, LBL_TRAIN, IMG_VAL, LBL_VAL):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    print(f"[wipe] cleared {PROC}/images,labels/{{train,val}}")


def copy_roboflow(spaghetti_cap):
    """Copy roboflow_review into train, capping Spaghetti-only images."""
    label_dir, img_dir = ROBOFLOW / "labels", ROBOFLOW / "images"
    if not label_dir.is_dir():
        print(f"[roboflow] SKIP — {label_dir} not found")
        return
    spag_only, others = [], []
    for lf in label_dir.glob("*.txt"):
        classes = {ln.split()[0] for ln in lf.read_text().split("\n") if ln.strip()}
        if not classes:
            continue
        (spag_only if classes == {"0"} else others).append(lf)

    random.seed(SEED)
    random.shuffle(spag_only)
    kept_spag = spag_only[:spaghetti_cap]
    keep = others + kept_spag
    print(f"[roboflow] {len(others)} multi/other-class + {len(kept_spag)}/{len(spag_only)} "
          f"Spaghetti-only (capped at {spaghetti_cap}) = {len(keep)} images")

    copied = 0
    for lf in keep:
        stem = lf.stem
        img = next((img_dir / f"{stem}{e}" for e in IMAGE_EXTS if (img_dir / f"{stem}{e}").exists()), None)
        if img is None:
            continue
        shutil.copy2(img, IMG_TRAIN / img.name)
        shutil.copy2(lf, LBL_TRAIN / lf.name)
        copied += 1
    print(f"[roboflow] copied {copied} image+label pairs into train")


def run(script, *args):
    print(f"\n[run] {script} {' '.join(args)}")
    r = subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"FAILED: {script} (exit {r.returncode})")


def report():
    box_per_class, img_per_class = Counter(), Counter()
    negatives = 0
    splits = {"train": (IMG_TRAIN, LBL_TRAIN), "val": (IMG_VAL, LBL_VAL)}
    counts = {}
    for split, (idir, ldir) in splits.items():
        n_img = sum(1 for p in idir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        counts[split] = n_img
        for lf in ldir.glob("*.txt"):
            lines = [ln for ln in lf.read_text().split("\n") if ln.strip()]
            if not lines:
                negatives += 1
                continue
            present = set()
            for ln in lines:
                c = int(ln.split()[0])
                box_per_class[c] += 1
                present.add(c)
            for c in present:
                img_per_class[c] += 1
    total = counts["train"] + counts["val"]
    print("\n" + "=" * 56)
    print(f"  V7 DATASET BUILT  |  train {counts['train']}  val {counts['val']}  total {total}")
    print("=" * 56)
    print(f"  {'class':<15}{'images':>9}{'boxes':>9}")
    for c in sorted(CLASS_NAMES):
        print(f"  {CLASS_NAMES[c]:<15}{img_per_class[c]:>9}{box_per_class[c]:>9}")
    ratio = 100 * negatives / total if total else 0
    print(f"  {'negatives':<15}{negatives:>9}   ({ratio:.1f}% of dataset)")
    print("=" * 56)
    print("Next: python train.py --model yolo11s.pt --name yolo11s_improved_v7")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spaghetti-cap", type=int, default=2000)
    ap.add_argument("--report-only", action="store_true", help="just print current pool stats")
    args = ap.parse_args()

    if args.report_only:
        report()
        return

    wipe()
    copy_roboflow(args.spaghetti_cap)
    run("data/scripts/syntetic_generated_data.py")
    run("data/scripts/add_real_defects.py")
    run("data/scripts/add_negatives.py")
    run("split_dataset.py")
    report()


if __name__ == "__main__":
    main()
