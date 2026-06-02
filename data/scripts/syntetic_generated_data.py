import os
import random
import numpy as np
from PIL import Image
from rembg import remove, new_session
from tqdm import tqdm

# --- CONFIGURATION ---
BACKGROUNDS_DIR = "data/clean_printerbed"
RAW_DATA_ROOT = "data/3D-Printing-Defect-Dataset/data"
OUTPUT_IMG_DIR = "data/processed/images/train"
OUTPUT_LBL_DIR = "data/processed/labels/train"

# Default images to generate per class.
# Stringing overridden to 300 below — its Kaggle images are too homogeneous
# to use directly (all from one experimental setup), so synthetics carry the
# full load for that class. All other classes: 150 synthetic + 150 real = 300.
IMAGES_PER_CLASS = 150

# Per-class override — classes whose Kaggle real images are too homogeneous
# to use directly get 300 synthetics to carry the full load.
# All other classes: 150 synthetic + 150 real = 300 total.
CLASS_COUNT_OVERRIDE = {
    "Stringing": 300,
    "Warping":   300,
}

TARGET_SIZE = (640, 640)

# Class IDs must match configs/defect_data.yaml exactly.
# Offplatform removed — no valid source data exists for it.
CLASS_MAP = {
    "Spaghetti":     0,
    "Warping":       1,
    "Layer_shifting": 2,
    "Stringing":     3,
    "Cracking":      4,
}


def build_defect_cache(session):
    """
    Pre-removes backgrounds from every defect source image once at startup.
    Cached results are reused across the generation loop so rembg is not
    called thousands of times — only once per unique source image.
    Returns: dict mapping class_name -> list of PIL Images (RGBA, bg removed).
    """
    cache = {name: [] for name in CLASS_MAP}
    total_files = 0

    for class_name in CLASS_MAP:
        folder = os.path.join(RAW_DATA_ROOT, class_name)
        if not os.path.exists(folder):
            print(f"  WARNING: source folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"  {class_name}: {len(files)} source images")
        total_files += len(files)

        for fname in tqdm(files, desc=f"  Removing bg — {class_name}", leave=False):
            try:
                path = os.path.join(folder, fname)
                img = Image.open(path).convert("RGBA")
                img_nobg = remove(img, session=session)
                cache[class_name].append(img_nobg)
            except Exception:
                continue

    cached_total = sum(len(v) for v in cache.values())
    print(f"Cache built: {cached_total} / {total_files} images processed.\n")
    return cache


def generate_dataset():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

    # 1. Load background images — ONLY clean printer bed images.
    #    These must be in data/clean_printerbed/ and represent empty/successful
    #    print beds. Never use defect images as backgrounds.
    background_paths = [
        os.path.join(BACKGROUNDS_DIR, f)
        for f in os.listdir(BACKGROUNDS_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not background_paths:
        print(f"ERROR: No background images found in {BACKGROUNDS_DIR}")
        return

    print(f"Loaded {len(background_paths)} background images from {BACKGROUNDS_DIR}")

    # 2. Create rembg session once — lighter u2netp model, CPU only.
    #    Forces CPUExecutionProvider to suppress missing cublasLt64_12.dll errors.
    print("Loading rembg session (u2netp, CPU) ...")
    session = new_session("u2netp", providers=["CPUExecutionProvider"])

    # 3. Pre-cache all background-removed defect images.
    #    rembg runs ONCE per source image, not once per generated composite.
    print("Pre-caching defect images (runs once) ...")
    defect_cache = build_defect_cache(session)

    valid_classes = [name for name in CLASS_MAP if defect_cache.get(name)]
    if not valid_classes:
        print("ERROR: No defect images cached. Check RAW_DATA_ROOT path.")
        return

    total_to_generate = sum(CLASS_COUNT_OVERRIDE.get(c, IMAGES_PER_CLASS) for c in valid_classes)
    print(f"Generating {total_to_generate} total images across {len(valid_classes)} classes ...\n")

    global_idx = 0

    for class_name in valid_classes:
        class_id = CLASS_MAP[class_name]
        class_images = defect_cache[class_name]
        n_generate = CLASS_COUNT_OVERRIDE.get(class_name, IMAGES_PER_CLASS)

        for i in tqdm(range(n_generate), desc=f"  {class_name} (class {class_id})"):
            try:
                # 4. Pick random background and resize to target.
                bg_path = random.choice(background_paths)
                bg = Image.open(bg_path).convert("RGBA")
                bg = bg.resize(TARGET_SIZE, Image.LANCZOS)
                bg_w, bg_h = bg.size

                # 5. Pick random defect from the pre-cached pool (instant — no rembg here).
                defect = random.choice(class_images).copy()

                # 6. Randomised scale: 10–45% of bed width.
                #    Matches real webcam deployment — defects are a fraction of the
                #    full bed view, not most of the frame. Fixes the oversized-box
                #    problem observed in v2 video output.
                scale = random.uniform(0.10, 0.45)
                max_defect_dim = max(defect.width, defect.height)
                scaling_factor = (bg_w * scale) / max_defect_dim
                new_w = min(int(defect.width * scaling_factor), bg_w - 2)
                new_h = min(int(defect.height * scaling_factor), bg_h - 2)
                defect = defect.resize((new_w, new_h), Image.LANCZOS)

                # 7. Constrained placement — keep defects within the inner 70% of
                #    the frame (15% margin on each side). Every background in
                #    clean_printerbed/ has the actual bed surface in this region.
                #    This prevents defects landing on the printer frame edges,
                #    gantry arms, or empty space visible in wider-angle shots.
                margin_x = int(bg_w * 0.15)
                margin_y = int(bg_h * 0.15)
                x_min = margin_x
                y_min = margin_y
                x_max = bg_w - margin_x - new_w
                y_max = bg_h - margin_y - new_h
                # Fall back to centre if defect is too large for the safe zone
                x_pos = random.randint(x_min, max(x_min, x_max))
                y_pos = random.randint(y_min, max(y_min, y_max))

                # 8. Composite defect onto background.
                final_img = bg.copy()
                final_img.paste(defect, (x_pos, y_pos), defect)

                # 9. Save image.
                filename = f"synth_{class_name}_{global_idx}.jpg"
                final_img.convert("RGB").save(os.path.join(OUTPUT_IMG_DIR, filename))

                # 10. Save YOLO label — bounding box from paste position.
                x_center = (x_pos + new_w / 2) / bg_w
                y_center = (y_pos + new_h / 2) / bg_h
                w_norm = new_w / bg_w
                h_norm = new_h / bg_h

                label_filename = os.path.splitext(filename)[0] + ".txt"
                with open(os.path.join(OUTPUT_LBL_DIR, label_filename), "w") as lf:
                    lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                global_idx += 1

            except Exception as e:
                print(f"  Skipped {class_name} image {i}: {e}")
                continue

    print(f"\nDone. Generated {global_idx} synthetic images in {OUTPUT_IMG_DIR}")
    print("Next steps:")
    print("  1. python data/scripts/add_real_defects.py  <- adds 150 real Kaggle images/class")
    print("  2. python data/scripts/add_negatives.py     <- adds clean-print negatives")
    print("  3. python split_dataset.py                  <- creates 80/20 train/val split")
    print("  4. python train.py                          <- train YOLOv8s")


if __name__ == "__main__":
    generate_dataset()
