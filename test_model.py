import os
from ultralytics import YOLO
import cv2

# --- CONFIGURATION ---
MODEL_PATH = r"C:\Users\hp pc\Desktop\3-D Prining Anomalies\runs\detect\3d_print_monitor\yolov8s_centered_synthetic\weights\best.pt"  # Your trained brain
TEST_FOLDER = "data/real_world_test" # Where your downloaded videos/images are

def run_test():
    # 1. Load the model
    if not os.path.exists(MODEL_PATH):
        print("Error: weights/best.pt not found! Did you move it?")
        return
    
    print(f"Loading model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # 2. Find all files in the test folder
    files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.jpg', '.png', '.mp4', '.avi', '.mov'))]
    
    if not files:
        print(f"No images or videos found in {TEST_FOLDER}. Please add some!")
        return

    print(f"Found {len(files)} files to test.")

    # 3. Run Prediction
    # 'conf=0.5' means the AI must be 50% sure to draw a box.
    # 'save=True' saves the video/image with boxes drawn on it.
    results = model.track(
        source=TEST_FOLDER, 
        conf=0.30,       # Try 0.60 or 0.65.
        iou=0.50,
        agnostic_nms=True,
        persist=True,      # <--- CRITICAL: Remembers objects between frames, used only with model.track specifically better for videos
        tracker="bytetrack.yaml", # Standard tracking algorithm
        save=True,
        project="runs/detect",
        name="real_world_test"
    )

    print("-" * 30)
    print(f"Done! Go check the folder: runs/detect/real_world_test")
    print("-" * 30)

if __name__ == "__main__":
    run_test()