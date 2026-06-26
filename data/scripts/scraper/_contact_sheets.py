"""One-off: tile raw scrape candidates into indexed contact sheets so a
human (or Claude) can triage 20 at a glance instead of opening files one by one.
Writes sheets + an index file mapping cell number -> original filename.
Usage: python _contact_sheets.py <target>   (e.g. warping, cracking)
Safe to delete after the v7 review."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TARGET = sys.argv[1] if len(sys.argv) > 1 else "warping"
RAW = Path(f"../../scraped_raw/{TARGET}")
OUT = Path(f"../../scraped_raw/{TARGET}_sheets")
OUT.mkdir(exist_ok=True)

COLS, ROWS = 5, 4
CELL_W, CELL_H = 340, 260
LABEL_H = 22

imgs = sorted(p for p in RAW.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
print(f"{len(imgs)} candidates")

try:
    font = ImageFont.truetype("arial.ttf", 18)
except OSError:
    font = ImageFont.load_default()

index_lines = []
per_sheet = COLS * ROWS
for s in range(0, len(imgs), per_sheet):
    batch = imgs[s:s + per_sheet]
    sheet_no = s // per_sheet + 1
    sheet = Image.new("RGB", (COLS * CELL_W, ROWS * (CELL_H + LABEL_H)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(batch):
        n = s + i + 1  # global 1-based cell number
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((CELL_W - 4, CELL_H - 4))
        except Exception as e:
            print(f"  skip unreadable {p.name}: {e}")
            continue
        cx = (i % COLS) * CELL_W
        cy = (i // COLS) * (CELL_H + LABEL_H)
        sheet.paste(im, (cx + (CELL_W - im.width) // 2, cy + (CELL_H - im.height) // 2))
        draw.rectangle([cx, cy + CELL_H, cx + CELL_W, cy + CELL_H + LABEL_H], fill="black")
        draw.text((cx + 6, cy + CELL_H + 2), f"#{n}", fill="yellow", font=font)
        index_lines.append(f"{n}\t{p.name}")
    sheet.save(OUT / f"sheet_{sheet_no:02d}.jpg", quality=85)
    print(f"  sheet_{sheet_no:02d}.jpg: cells {s+1}-{s+len(batch)}")

(OUT / "index.tsv").write_text("\n".join(index_lines), encoding="utf-8")
print(f"Done -> {OUT}")
