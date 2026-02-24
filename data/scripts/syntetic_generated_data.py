import os
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
from rembg import remove, new_session
from tqdm import tqdm

# --- CONFIGURATION ---
BACKGROUNDS_DIR = "data\clean_printerbed"
RAW_DATA_ROOT = r"data/3D-Printing-Defect-Dataset/data"
OUTPUT_IMG_DIR = "data/processed/images/train"
OUTPUT_LBL_DIR = "data/processed/labels/train"
# How many images to generate total?
TOTAL_IMAGES = 1000
TARGET_SIZE = (640, 640)

# Exact folder names -> Class IDs
CLASS_MAP = {
    "Spaghetti": 0,
    "Warping": 1,
    "Layer_shifting": 2,
    "Stringing": 3,
    "Offplatfrom": 4,
    "Cracking": 5
}

def build_defect_cache(session):
    """
    Pre-removes backgrounds from every defect image once at startup.
    Cached results are reused in the generation loop, so rembg is not
    called 1000 times — only once per unique source image.
    Returns: dict mapping defect_name -> list of PIL Images (RGBA, bg removed)
    """
    cache = {name: [] for name in CLASS_MAP}
    total_files = 0

    for defect_name in CLASS_MAP:
        folder = os.path.join(RAW_DATA_ROOT, defect_name)
        if not os.path.exists(folder):
            print(f"  WARNING: folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
        print(f"  {defect_name}: {len(files)} images")
        total_files += len(files)

        for fname in tqdm(files, desc=f"  Removing bg — {defect_name}", leave=False):
            try:
                path = os.path.join(folder, fname)
                img = Image.open(path).convert("RGBA")
                img_nobg = remove(img, session=session)
                cache[defect_name].append(img_nobg)
            except Exception:
                continue

    print(f"Cache built: {sum(len(v) for v in cache.values())} / {total_files} images processed.\n")
    return cache


def generate_centered_dataset():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

    # 1. Load Backgrounds
    background_paths = [
        os.path.join(BACKGROUNDS_DIR, f)
        for f in os.listdir(BACKGROUNDS_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not background_paths:
        print("ERROR: No background images found in data/backgrounds!")
        return

    # 2. Create rembg session once — lighter u2netp model, CPU only.
    #    This suppresses the 'cublasLt64_12.dll missing' CUDA error and
    #    is significantly faster on systems without a CUDA 12 toolkit.
    print("Loading rembg session (u2netp, CPU) ...")
    session = new_session("u2netp", providers=["CPUExecutionProvider"])

    # 3. Pre-cache all background-removed defect images.
    #    rembg runs ONCE per source image, not once per generated image.
    print("Pre-caching defect images (runs once) ...")
    defect_cache = build_defect_cache(session)

    valid_classes = [name for name, imgs in defect_cache.items() if imgs]
    if not valid_classes:
        print("ERROR: No defect images were cached. Check RAW_DATA_ROOT path.")
        return

    print("Generating Dataset with Randomized Placement...")

    for i in tqdm(range(TOTAL_IMAGES)):
        try:
            # 4. Pick Random Background & Resize
            bg_path = random.choice(background_paths)
            bg = Image.open(bg_path).convert("RGBA")
            bg = bg.resize(TARGET_SIZE, Image.LANCZOS)
            bg_w, bg_h = bg.size

            # 5. Pick Random Defect from cache (instant — no rembg call here)
            defect_name = random.choice(valid_classes)
            class_id = CLASS_MAP[defect_name]
            defect = random.choice(defect_cache[defect_name]).copy()

            # 6. Smart Resize — randomized scale 30–90% of bed (was 50–80%)
            # Wider range teaches the model to detect defects at various distances/sizes.
            scale = random.uniform(0.3, 0.9)

            max_defect_dim = max(defect.width, defect.height)
            scaling_factor = (bg_w * scale) / max_defect_dim

            new_w = int(defect.width * scaling_factor)
            new_h = int(defect.height * scaling_factor)

            # Clamp so defect never exceeds the canvas
            new_w = min(new_w, bg_w - 2)
            new_h = min(new_h, bg_h - 2)

            defect = defect.resize((new_w, new_h), Image.LANCZOS)

            # 7. RANDOMIZED POSITION — defects occur anywhere on the bed in reality.
            # Keep defect fully inside the canvas by limiting placement range.
            max_x = bg_w - new_w
            max_y = bg_h - new_h

            if max_x <= 0 or max_y <= 0:
                # Fallback to center if defect is near full canvas size
                x_pos = (bg_w - new_w) // 2
                y_pos = (bg_h - new_h) // 2
            else:
                x_pos = random.randint(0, max_x)
                y_pos = random.randint(0, max_y)

            # 8. Paste
            final_img = bg.copy()
            final_img.paste(defect, (x_pos, y_pos), defect)

            # 9. Save Image
            filename = f"synth_{defect_name}_{i}.jpg"
            final_img.convert("RGB").save(f"{OUTPUT_IMG_DIR}/{filename}")

            # 10. Save Label — compute actual bounding box from paste position
            x_center = (x_pos + new_w / 2) / bg_w
            y_center = (y_pos + new_h / 2) / bg_h
            w_norm = new_w / bg_w
            h_norm = new_h / bg_h

            label_filename = os.path.splitext(filename)[0] + ".txt"
            with open(f"{OUTPUT_LBL_DIR}/{label_filename}", "w") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        except Exception as e:
            continue

    print(f"Done! Generated {TOTAL_IMAGES} images in {OUTPUT_IMG_DIR}")

if __name__ == "__main__":
    generate_centered_dataset()
