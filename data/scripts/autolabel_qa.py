"""
Vision-assisted annotation + automated QA for the scraped real-defect images.

The scraped real defects (real_warping / real_cracking / real_blob_of_death) have no
bounding boxes. Drawing them by hand for ~75 images is slow, and the old fallback of a
single 0.5/0.5/0.85/0.85 full-frame box is exactly what broke Warping (its loose box
overlapped Layer_shifting geometry). This tool takes TIGHT per-image boxes estimated by
a vision model, writes them as YOLO sidecar labels, and runs an automated QA pass so the
boxes can be trusted/spot-checked without hand-drawing each one.

Pipeline:
  1. labels JSON: {"<filename>": [[cls, xc, yc, w, h], ...], ...}   (normalised 0..1)
  2. write <stem>.txt YOLO sidecars next to each image
  3. render each image with its boxes drawn -> _label_qa/<class>/<filename>
  4. build a contact sheet _label_qa/<class>_contactsheet.jpg for one-glance review
  5. print per-box area stats + flag suspicious boxes (area > 0.90 = near-full-frame,
     area < 0.015 = tiny, or any coord outside [0,1])

Usage:
  python data/scripts/autolabel_qa.py --img-dir data/real_warping \
      --labels data/scripts/_label_qa/warping_labels.json --tag warping
"""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = {
    0: "Spaghetti", 1: "Warping", 2: "Layer_shifting",
    3: "Stringing", 4: "Cracking", 5: "Blob_of_death",
}
CLASS_COLOR = {
    0: (255, 80, 80), 1: (80, 160, 255), 2: (80, 220, 120),
    3: (255, 200, 60), 4: (220, 100, 255), 5: (255, 130, 40),
}

QA_ROOT = Path("data/scripts/_label_qa")


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def write_sidecars(img_dir: Path, labels: dict):
    written = 0
    for fname, boxes in labels.items():
        stem = Path(fname).stem
        lines = [f"{int(c)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in boxes]
        (img_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        written += 1
    return written


def render_qa(img_dir: Path, labels: dict, tag: str):
    qa_dir = QA_ROOT / tag
    qa_dir.mkdir(parents=True, exist_ok=True)
    renders = []
    for fname, boxes in labels.items():
        path = img_dir / fname
        if not path.exists():
            print(f"  MISSING image: {fname}")
            continue
        im = Image.open(path).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        lw = max(3, round(min(W, H) * 0.006))
        fnt = _font(max(14, round(min(W, H) * 0.035)))
        for c, xc, yc, w, h in boxes:
            x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
            x2, y2 = (xc + w / 2) * W, (yc + h / 2) * H
            col = CLASS_COLOR.get(int(c), (255, 255, 255))
            d.rectangle([x1, y1, x2, y2], outline=col, width=lw)
            d.text((x1 + lw, y1 + lw), CLASS_NAMES.get(int(c), str(c)), fill=col, font=fnt)
        out = qa_dir / fname
        im.save(out)
        renders.append(out)
    return renders, qa_dir


def contact_sheet(renders, tag: str, cols=5, thumb=320):
    if not renders:
        return None
    rows = math.ceil(len(renders) / cols)
    cap = 22
    cell_w, cell_h = thumb, thumb + cap
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (28, 28, 28))
    d = ImageDraw.Draw(sheet)
    fnt = _font(13)
    for i, r in enumerate(renders):
        im = Image.open(r).convert("RGB")
        im.thumbnail((thumb, thumb))
        cx, cy = (i % cols) * cell_w, (i // cols) * cell_h
        sheet.paste(im, (cx + (thumb - im.width) // 2, cy + (thumb - im.height) // 2))
        d.text((cx + 3, cy + thumb + 3), r.stem.replace("promoted_", ""), fill=(230, 230, 230), font=fnt)
    out = QA_ROOT / f"{tag}_contactsheet.jpg"
    sheet.save(out, quality=88)
    return out


def stats_and_flags(labels: dict):
    print("\n  --- box-area QA (normalised area = w*h) ---")
    n_boxes = 0
    flags = []
    for fname, boxes in labels.items():
        for c, xc, yc, w, h in boxes:
            n_boxes += 1
            area = w * h
            bad_coord = any(not (0 <= v <= 1) for v in (xc, yc, w, h)) or \
                        (xc - w / 2 < -0.01) or (xc + w / 2 > 1.01) or \
                        (yc - h / 2 < -0.01) or (yc + h / 2 > 1.01)
            if bad_coord:
                flags.append(f"    OUT-OF-BOUNDS  {fname}  {CLASS_NAMES.get(int(c))} {xc,yc,w,h}")
            elif area > 0.90:
                flags.append(f"    NEAR-FULL-FRAME (area={area:.2f})  {fname}  {CLASS_NAMES.get(int(c))}")
            elif area < 0.015:
                flags.append(f"    TINY (area={area:.3f})  {fname}  {CLASS_NAMES.get(int(c))}")
    print(f"  images: {len(labels)} | boxes: {n_boxes}")
    if flags:
        print("  FLAGS:")
        print("\n".join(flags))
    else:
        print("  no suspicious boxes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--labels", required=True, help="path to labels JSON")
    ap.add_argument("--tag", required=True, help="class tag for QA folder/sheet")
    ap.add_argument("--no-sidecars", action="store_true", help="QA only, don't write .txt")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    labels = json.loads(Path(args.labels).read_text())

    if not args.no_sidecars:
        n = write_sidecars(img_dir, labels)
        print(f"  wrote {n} YOLO sidecar labels into {img_dir}")

    renders, qa_dir = render_qa(img_dir, labels, args.tag)
    sheet = contact_sheet(renders, args.tag)
    stats_and_flags(labels)
    print(f"\n  QA renders: {qa_dir}")
    if sheet:
        print(f"  contact sheet: {sheet}")


if __name__ == "__main__":
    main()
