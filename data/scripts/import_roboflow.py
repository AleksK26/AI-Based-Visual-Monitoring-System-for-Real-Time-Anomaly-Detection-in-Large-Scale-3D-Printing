"""
Import a Roboflow (YOLOv8-format) dataset export and remap it to OUR class scheme.

This is the PRIMARY data path now: the Roboflow Universe datasets (#2 syLucauc,
#3 Failure Detector, #4 HCMUT) are already bounding-box labelled, side/top-view, and
YOLOv8-ready — far better than scraping image search. They use different class names
and ids, though, so this script:
  1. reads the export's data.yaml to learn its index -> name mapping
  2. remaps each source class NAME to our class id via NAME_TO_ID (below)
  3. drops boxes for classes we don't model (blobs/zits/under-over-extrusion/etc.)
  4. copies image + remapped label into a SEPARATE staging area for review
     (data/roboflow_review/), NOT straight into data/processed — same "keep it
     separate until we're ready" discipline as the scraper.

How to get an export:
  - Roboflow Universe -> open the dataset (or merge #2+#3+#4 in your workspace) ->
    Download Dataset -> format "YOLOv8" -> download the zip and unzip it.
  - Then:  python data/scripts/import_roboflow.py --src path/to/export --name syLucauc

Our class ids (configs/defect_data.yaml):
    0 Spaghetti | 1 Warping | 2 Layer_shifting | 3 Stringing | 4 Cracking

Install: pip install pyyaml
"""

import argparse
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGING = REPO_ROOT / "data" / "roboflow_review"

# Source-dataset class NAME (normalised: lowercased, spaces/hyphens -> underscore)
# -> our class id. Names not listed here are DROPPED (logged), e.g. blobs, zits,
# under_extrusion, over_extrusion, success, failure — none are in our 5 classes yet.
NAME_TO_ID = {
    "spaghetti": 0,
    "warping": 1, "warp": 1,
    "layer_shifting": 2, "layer_shift": 2, "layershift": 2, "layer_shifting_": 2,
    "stringing": 3, "string": 3, "strings": 3,
    "cracking": 4, "cracks": 4, "crack": 4, "cracked": 4,
}


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _load_names(data_yaml: Path) -> dict:
    """Return {source_index: source_name} from a Roboflow data.yaml."""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    raise ValueError(f"Could not parse class names from {data_yaml}")


def _find_splits(src: Path):
    """Yield (images_dir, labels_dir) for each split present in the export."""
    for split in ("train", "valid", "test", "val"):
        img = src / split / "images"
        lbl = src / split / "labels"
        if img.is_dir() and lbl.is_dir():
            yield img, lbl
    # flat layout fallback
    if (src / "images").is_dir() and (src / "labels").is_dir():
        yield src / "images", src / "labels"


def import_export(src: Path, dataset_name: str):
    data_yaml = next((p for p in [src / "data.yaml", src / "data.yml"] if p.exists()), None)
    if data_yaml is None:
        raise FileNotFoundError(f"No data.yaml in {src} — is this a YOLOv8 export?")

    src_names = _load_names(data_yaml)
    # Build source-index -> our-id remap, and report what maps where.
    remap, dropped_names = {}, set()
    for idx, raw in src_names.items():
        ours = NAME_TO_ID.get(_norm(raw))
        if ours is None:
            dropped_names.add(raw)
        else:
            remap[idx] = ours
    print(f"  source classes: {src_names}")
    print(f"  -> kept remap (src_idx->our_id): {remap}")
    if dropped_names:
        print(f"  -> dropping (not in our 5 classes): {sorted(dropped_names)}")

    out_img = STAGING / "images"
    out_lbl = STAGING / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    kept_imgs, kept_boxes, skipped_empty, per_class = 0, 0, 0, {}
    for img_dir, lbl_dir in _find_splits(src):
        for label_file in lbl_dir.glob("*.txt"):
            lines = [ln for ln in label_file.read_text().splitlines() if ln.strip()]
            new_lines = []
            for ln in lines:
                parts = ln.split()
                try:
                    cls = int(parts[0])
                except (ValueError, IndexError):
                    continue
                if cls in remap:
                    our_id = remap[cls]
                    new_lines.append(" ".join([str(our_id)] + parts[1:]))
                    per_class[our_id] = per_class.get(our_id, 0) + 1
                    kept_boxes += 1
            if not new_lines:
                skipped_empty += 1
                continue  # no boxes of our classes -> skip this image

            # find the matching image (same stem, any ext)
            stem = label_file.stem
            img_path = next((img_dir / f"{stem}{e}" for e in
                             (".jpg", ".jpeg", ".png", ".webp", ".bmp")
                             if (img_dir / f"{stem}{e}").exists()), None)
            if img_path is None:
                continue

            base = f"rf_{dataset_name}_{stem}"
            shutil.copy2(img_path, out_img / f"{base}{img_path.suffix.lower()}")
            (out_lbl / f"{base}.txt").write_text("\n".join(new_lines) + "\n")
            kept_imgs += 1

    our_names = ["Spaghetti", "Warping", "Layer_shifting", "Stringing", "Cracking"]
    by_name = {our_names[k]: v for k, v in sorted(per_class.items())}
    print(f"  imported {kept_imgs} images, {kept_boxes} boxes "
          f"(skipped {skipped_empty} with no relevant class)")
    print(f"  per-class box counts: {by_name}")
    print(f"  -> staged in {STAGING}/images + /labels")


def main():
    ap = argparse.ArgumentParser(description="Import a Roboflow YOLOv8 export, remapped to our classes.")
    ap.add_argument("--src", required=True, help="path to the unzipped Roboflow export")
    ap.add_argument("--name", required=True, help="short dataset label (filename prefix), e.g. syLucauc")
    args = ap.parse_args()
    import_export(Path(args.src), args.name)
    print("\nReview data/roboflow_review/ (use a YOLO label viewer to spot-check boxes), "
          "then merge images+labels into data/processed/images/{train} + labels/{train} "
          "before split_dataset.py.")


if __name__ == "__main__":
    main()
