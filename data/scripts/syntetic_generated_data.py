import os
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
from rembg import remove
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

    print("Generating Dataset with Randomized Placement...")

    for i in tqdm(range(TOTAL_IMAGES)):
        try:
            # 2. Pick Random Background & Resize
            bg_path = random.choice(background_paths)
            bg = Image.open(bg_path).convert("RGBA")
            bg = bg.resize(TARGET_SIZE, Image.LANCZOS)
            bg_w, bg_h = bg.size

            # 3. Pick Random Defect
            defect_name = random.choice(list(CLASS_MAP.keys()))
            class_id = CLASS_MAP[defect_name]
            defect_folder = os.path.join(RAW_DATA_ROOT, defect_name)

            if not os.path.exists(defect_folder): continue

            defect_files = [f for f in os.listdir(defect_folder) if f.lower().endswith(('.jpg', '.png'))]
            if not defect_files: continue

            defect_path = os.path.join(defect_folder, random.choice(defect_files))
            defect = Image.open(defect_path).convert("RGBA")

            # 4. Remove Background (Clean Cut)
            defect = remove(defect)

            # 5. Smart Resize — randomized scale 30–90% of bed (was 50–80%)
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

            # 6. RANDOMIZED POSITION — defects occur anywhere on the bed in reality.
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

            # 7. Paste
            final_img = bg.copy()
            final_img.paste(defect, (x_pos, y_pos), defect)

            # 8. Save Image
            filename = f"synth_{defect_name}_{i}.jpg"
            final_img.convert("RGB").save(f"{OUTPUT_IMG_DIR}/{filename}")

            # 9. Save Label — compute actual bounding box from paste position
            x_center = (x_pos + new_w / 2) / bg_w
            y_center = (y_pos + new_h / 2) / bg_h
            w_norm = new_w / bg_w
            h_norm = new_h / bg_h

            label_filename = os.path.splitext(filename)[0] + ".txt"
            with open(f"{OUTPUT_LBL_DIR}/{label_filename}", "w") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        except Exception as e:
            # Skip bad images silently
            continue

    print(f"Done! Generated {TOTAL_IMAGES} images in {OUTPUT_IMG_DIR}")

if __name__ == "__main__":
    generate_centered_dataset()
