import os
from ultralytics import YOLO
import cv2

# --- CONFIGURATION ---
# Updated to use the latest improved model weights.
# Switch to 'yolov8s_improved_v1' weights after the new training run completes.
MODEL_PATH = r"runs\detect\3d_print_monitor\yolov8s_centered_synthetic2\weights\best.pt"
TEST_FOLDER = "data/real_world_test"   # Where your downloaded videos/images are

# Confidence threshold: raised from 0.30 to 0.55 to reduce false positives.
# If you miss real defects, lower to 0.45. If you still get false alarms, raise to 0.65.
CONF_THRESHOLD = 0.55

def run_test():
    # 1. Load the model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model not found at {MODEL_PATH}")
        print("Did you train yet? Run: python train.py")
        return

    print(f"Loading model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # 2. Find all files in the test folder
    files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.jpg', '.png', '.mp4', '.avi', '.mov'))]

    if not files:
        print(f"No images or videos found in {TEST_FOLDER}. Please add some!")
        return

    print(f"Found {len(files)} files to test.")
    print(f"Confidence threshold: {CONF_THRESHOLD}")

    # 3. Run Prediction with ByteTrack for temporal consistency across video frames
    results = model.track(
        source=TEST_FOLDER,
        conf=CONF_THRESHOLD,
        iou=0.50,
        agnostic_nms=True,
        persist=True,           # Remembers objects between frames (video only)
        tracker="bytetrack.yaml",
        save=True,
        project="runs/detect",
        name="real_world_test"
    )

    print("-" * 30)
    print(f"Done! Results saved to: runs/detect/real_world_test")
    print("-" * 30)

if __name__ == "__main__":
    run_test()
