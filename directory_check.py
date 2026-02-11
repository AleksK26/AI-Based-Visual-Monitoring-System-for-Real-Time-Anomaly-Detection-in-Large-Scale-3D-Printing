import os
import glob

# CONFIG (Match your defect_data.yaml)
BASE_DIR = "data/processed" 
IMG_DIR = os.path.join(BASE_DIR, "images/train")
LBL_DIR = os.path.join(BASE_DIR, "labels/train")

def check_structure():
    print(f"Checking: {os.path.abspath(IMG_DIR)}")
    images = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    print(f"Found {len(images)} images.")

    print(f"Checking: {os.path.abspath(LBL_DIR)}")
    labels = glob.glob(os.path.join(LBL_DIR, "*.txt"))
    print(f"Found {len(labels)} label files.")

    if len(images) > 0 and len(labels) == 0:
        print("\nCRITICAL ERROR: Images found but NO labels found.")
        print("YOLO requires a folder named 'labels' next to 'images'.")
    elif len(images) == 0:
        print("\nCRITICAL ERROR: No images found. Check your path.")
    else:
        # Check a pair
        test_img = os.path.basename(images[0])
        test_lbl = test_img.replace(".jpg", ".txt")
        if os.path.exists(os.path.join(LBL_DIR, test_lbl)):
            print(f"\nSUCCESS: Matched {test_img} with {test_lbl}")
        else:
            print(f"\nERROR: Could not find label for {test_img}")

if __name__ == "__main__":
    check_structure()