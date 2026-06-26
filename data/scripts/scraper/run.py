"""
Orchestrator for the defect-image scraping pipeline:  harvest -> verify -> gate -> dedup.

For each target (defect class or hard-negative group) it:
  1. HARVESTS raw candidates from all engines      (harvest.py)
  2. VERIFIES each candidate with Gemini vision      (verify.py)
  3. GATES on deterministic rules from config.py:
       - the verifier must see a real 3D print (not a render/marketing shot)
       - defect_class must match the target (for negatives: defect_class == none)
       - confidence >= min_confidence
       - reject_gold: drop gold/metallic (the bias fix); must_be_gold: keep only gold
       - per-colour cap (max_per_colour) so no single filament colour dominates
  4. COPIES accepted images into the target's output_dir (the existing
     data/real_* / hard_negatives folders the training pipeline already reads)
  5. PERCEPTUAL-DEDUPS the output folder                 (dedup.py)
  6. Prints a colour + view distribution so you can SEE the diversity you got.

A final human review of the output folders is still expected — auto-labels are
good but not infallible, and mislabelled data is exactly what biased v5.

Usage:
  python run.py                      # all targets
  python run.py --only cracking warping
  python run.py --skip-harvest       # re-verify already-harvested raw/ candidates
  python run.py --dry-run            # harvest + verify + report, but don't copy

Env: GEMINI_API_KEY (required for verify). Optional REDDIT_CLIENT_ID/SECRET.
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import harvest
import verify as verify_mod
from config import RAW_DIR, STAGING_DIR, all_targets
from dedup import dedupe_dir

ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _normalise_colour(raw: str) -> str:
    """Collapse the free-text colour into a coarse bucket for the diversity cap."""
    c = (raw or "").lower()
    buckets = [
        "white", "black", "grey", "gray", "silver", "red", "orange", "yellow",
        "green", "blue", "purple", "pink", "brown", "gold", "brass", "bronze",
        "copper", "metallic", "natural", "translucent", "clear",
    ]
    for b in buckets:
        if b in c:
            return "grey" if b == "gray" else b
    return "other"


def _accept(assessment: dict, name: str, spec: dict, is_neg: bool,
            colour_counts: Counter) -> tuple[bool, str]:
    """Apply the deterministic gating rules. Returns (accept, reason_if_rejected)."""
    a = assessment
    if not a.get("shows_3d_print"):
        return False, "not a 3D print"
    if a.get("is_render_or_marketing"):
        return False, "render/marketing"
    if a["confidence"] < spec["min_confidence"]:
        return False, f"low confidence {a['confidence']:.2f}"

    colour = _normalise_colour(a.get("filament_colour", ""))

    if is_neg:
        if a.get("defect_class") != "none":
            return False, f"not clean (saw {a.get('defect_class')})"
        if spec.get("must_be_gold") and not a.get("is_gold_metallic"):
            return False, "negative must be gold/metallic"
        if not spec.get("must_be_gold") and a.get("is_gold_metallic"):
            return False, "gold handled by gold_clean group"
    else:
        if a.get("defect_class") != name:
            return False, f"class mismatch ({a.get('defect_class')})"
        if spec.get("reject_gold") and a.get("is_gold_metallic"):
            return False, "gold/metallic (bias)"

    cap = spec.get("max_per_colour", 99)
    if colour_counts[colour] >= cap:
        return False, f"colour cap reached for {colour}"

    colour_counts[colour] += 1
    return True, ""


def process_target(name, spec, is_neg, args):
    print(f"\n=== {name} {'(hard negative)' if is_neg else ''} ===")
    if "lacking" in spec:
        print(f"  gap: {spec['lacking']}")

    if not args.skip_harvest:
        harvest.harvest_target(name, spec)

    raw_dir = RAW_DIR / name
    if not raw_dir.exists():
        print(f"  no raw candidates at {raw_dir} — run without --skip-harvest first")
        return

    # Accepted images go to a SEPARATE staging folder for review — NOT into the
    # real training folders. promote.py merges them in after you eyeball them.
    out_dir = STAGING_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    target_n = spec.get("target", 40)

    candidates = sorted(p for p in raw_dir.iterdir()
                        if p.suffix.lower() in ACCEPTED_EXTS)
    print(f"  verifying {len(candidates)} candidates (target {target_n} accepted)...")

    colour_counts, view_counts = Counter(), Counter()
    accepted = 0
    for i, img in enumerate(candidates, 1):
        if accepted >= target_n:
            print(f"  reached target of {target_n} — stopping verification")
            break
        try:
            a = verify_mod.verify_image(img, spec)
        except Exception as e:
            print(f"    [{i}] verify error, skipping: {e}")
            continue

        ok, why = _accept(a, name, spec, is_neg, colour_counts)
        if not ok:
            continue
        view_counts[a.get("view", "other")] += 1
        accepted += 1
        if not args.dry_run:
            dest = out_dir / f"scraped_{name}_{accepted:03d}.jpg"
            shutil.copy2(img, dest)
        if accepted % 5 == 0:
            print(f"    accepted {accepted}/{target_n} "
                  f"(colours: {dict(colour_counts)})")

    print(f"  ACCEPTED {accepted} images for '{name}'")
    print(f"    colour distribution: {dict(colour_counts)}")
    print(f"    view distribution:   {dict(view_counts)}")

    if not args.dry_run and accepted:
        removed = dedupe_dir(out_dir)
        if removed:
            print(f"    removed {removed} perceptual duplicates from {out_dir}")
    print(f"  -> staged in {out_dir}")
    print(f"     review by eye, then: python promote.py --only {name}")


def main():
    ap = argparse.ArgumentParser(description="Scrape + verify 3D-printing defect images.")
    ap.add_argument("--only", nargs="*", help="run only these targets by name")
    ap.add_argument("--skip-harvest", action="store_true",
                    help="reuse already-harvested raw candidates (verify only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="harvest + verify + report, but do not copy into output dirs")
    args = ap.parse_args()

    targets = list(all_targets())
    if args.only:
        wanted = set(args.only)
        targets = [t for t in targets if t[0] in wanted]
        if not targets:
            print(f"No matching targets. Available: "
                  f"{[t[0] for t in all_targets()]}")
            return

    for name, spec, is_neg in targets:
        process_target(name, spec, is_neg, args)

    print(f"\nDone. Accepted images are STAGED in {STAGING_DIR} (separate from training "
          "data).\nReview each folder by eye, delete mistakes, then run promote.py to merge "
          "the keepers into the real data/real_*/ and hard_negatives/ folders.")


if __name__ == "__main__":
    main()
