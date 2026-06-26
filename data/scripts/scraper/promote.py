"""
Step 4 (manual gate): PROMOTE reviewed images from staging into the real dataset.

The scraper stages accepted images in data/scraped_review/<target>/ — kept SEPARATE
from training data on purpose. After you've eyeballed a folder and deleted any
mistakes, run this to copy the keepers into their real destination
(data/real_cracking/, data/real_warping/, ..., or hard_negatives/), where
add_real_defects.py / add_negatives.py will pick them up.

This is a deliberate human gate: nothing reaches the training set until you've
looked at it. Bad labels are what biased v5.

Usage:
  python promote.py --only cracking warping   # promote specific targets
  python promote.py                           # promote everything reviewed
  python promote.py --only off_platform --move  # move instead of copy (clears staging)
"""

import argparse
import shutil
from pathlib import Path

from config import STAGING_DIR, all_targets

ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def promote(name, spec, move=False):
    staging = STAGING_DIR / name
    final = Path(spec["output_dir"])
    if not staging.exists():
        print(f"  [{name}] nothing staged at {staging} — skip")
        return 0
    imgs = [p for p in staging.iterdir() if p.suffix.lower() in ACCEPTED_EXTS]
    if not imgs:
        print(f"  [{name}] no images left in staging (all reviewed out?) — skip")
        return 0

    final.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in imgs:
        dest = final / f"promoted_{name}_{n:03d}{p.suffix.lower()}"
        (shutil.move if move else shutil.copy2)(str(p), str(dest))
        n += 1
    verb = "moved" if move else "copied"
    print(f"  [{name}] {verb} {n} reviewed images -> {final}")
    return n


def main():
    ap = argparse.ArgumentParser(description="Promote reviewed scraped images into the dataset.")
    ap.add_argument("--only", nargs="*", help="promote only these targets")
    ap.add_argument("--move", action="store_true",
                    help="move instead of copy (empties the staging folder)")
    args = ap.parse_args()

    targets = list(all_targets())
    if args.only:
        wanted = set(args.only)
        targets = [t for t in targets if t[0] in wanted]

    total = sum(promote(name, spec, move=args.move) for name, spec, _ in targets)
    print(f"\nPromoted {total} images total. Re-run the dataset build "
          "(add_real_defects.py / add_negatives.py / split_dataset.py) to include them.")


if __name__ == "__main__":
    main()
