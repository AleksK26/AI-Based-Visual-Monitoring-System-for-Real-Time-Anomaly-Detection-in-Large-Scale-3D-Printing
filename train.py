"""
The model training pipeline.
CURRENT ROLE / Idea: Trains a base YOLOv8 model on the synthetic/web-scraped dataset to learn general defect features (Spaghetti, Warping).
FUTURE ROLE: Will be used for 'Transfer Learning'—taking the base model and fine-tuning it on the specific images captured from your large-scale printer for maximum accuracy.
"""

from ultralytics import YOLO

def train_model():
    # 1. Load the Model
    # Start from base ImageNet pre-trained weights for a clean, unbiased training run.
    # Switch to 'yolov8n.pt' for a faster Raspberry Pi-friendly variant.
    model = YOLO("yolov8s.pt")

    # 2. Start Training
    print("Starting Training on Large-Scale Defect Dataset...")

    results = model.train(
        # DATASET
        data='configs/defect_data.yaml',

        # TRAINING DURATION
        epochs=50,              # Increased from 35 — gives the model more time to converge
        patience=15,            # Stop early if no improvement for 15 epochs

        # IMAGE SETTINGS
        imgsz=640,              # Standard YOLO resolution
        batch=16,               # Reduce to 8 if you run out of GPU memory

        # AUGMENTATION — Real-world deployment conditions
        mosaic=0.5,             # Stitches 4 images together (simulates global bed view)
        mixup=0.1,              # Blends images slightly (prevents overfitting)
        degrees=5.0,            # Rotates images +/- 5 degrees (simulates camera tilt)
        fliplr=0.5,             # Flips left/right (defects look same flipped)

        # LIGHTING & CAMERA QUALITY — Critical for webcam/Pi deployment
        hsv_h=0.015,            # Hue shift (subtle colour variation)
        hsv_s=0.4,              # Saturation shift (handles different lighting)
        hsv_v=0.4,              # Value/brightness shift (day vs night lighting)
        blur=0.1,               # Random blur (simulates webcam focus drift and Pi cam quality)

        # HARDWARE
        device=0,               # Use GPU (change to 'cpu' if no NVIDIA GPU)
        workers=4,              # Data loading threads

        # NAMING — new name so results don't overwrite previous runs
        project='runs/detect/3d_print_monitor',
        name='yolov8s_improved_v1'
    )

    # 3. Validation against the real val split
    print("Training Complete. Validating on held-out val set...")
    metrics = model.val()
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # 4. Export for Deployment (Raspberry Pi / edge format)
    print("Exporting to ONNX for edge deployment...")
    model.export(format='onnx')
    print("Done. Check runs/detect/3d_print_monitor/yolov8s_improved_v1/")

if __name__ == '__main__':
    train_model()
