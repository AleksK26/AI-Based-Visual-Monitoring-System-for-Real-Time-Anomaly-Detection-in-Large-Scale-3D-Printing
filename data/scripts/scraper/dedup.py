"""
Step 3 helper: perceptual-hash DE-DUP.

Content-hash dedup (in harvest.py) only catches byte-identical files. The same
photo is often reposted at different resolutions / re-encodings across engines and
forums, which produces different bytes but the same image. Perceptual hashing
(pHash) collapses those near-duplicates so they don't all land in the training set
(duplicate training images waste labelling effort and skew the distribution).

Install: pip install imagehash pillow
"""

from pathlib import Path

from PIL import Image

try:
    import imagehash
except ImportError:
    imagehash = None

ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dedupe_dir(directory, max_distance: int = 5) -> int:
    """Remove perceptual near-duplicates in `directory` in place.

    Keeps the first image of each near-duplicate cluster (Hamming distance between
    pHashes <= max_distance). Returns the number of files removed. If imagehash is
    not installed, it's a no-op (returns 0) — content-hash dedup still applied.
    """
    if imagehash is None:
        print("    (imagehash not installed — skipping perceptual dedup)")
        return 0

    directory = Path(directory)
    images = sorted(p for p in directory.iterdir()
                    if p.suffix.lower() in ACCEPTED_EXTS)
    kept_hashes = []
    removed = 0
    for p in images:
        try:
            h = imagehash.phash(Image.open(p).convert("RGB"))
        except Exception:
            continue
        if any((h - kh) <= max_distance for kh in kept_hashes):
            p.unlink(missing_ok=True)
            removed += 1
        else:
            kept_hashes.append(h)
    return removed


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    n = dedupe_dir(target)
    print(f"Removed {n} perceptual duplicates from {target}")
