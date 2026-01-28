"""
The model training pipeline.
CURRENT ROLE / Idea: Trains a base YOLOv8 model on the synthetic/web-scraped dataset to learn general defect features (Spaghetti, Warping).
FUTURE ROLE: Will be used for 'Transfer Learning'—taking the base model and fine-tuning it on the specific images captured from your large-scale printer for maximum accuracy.
"""

from ultralytics import YOLO

def train_model():
    # 1. Load the Model
    # 'yolov8s.pt' downloads automatically. It's a pre-trained model (Transfer Learning).
    model = YOLO('yolov8s.pt') 

    # 2. Start Training
    print("Starting Training on Large-Scale Defect Dataset...")
    
    results = model.train(
        # DATASET
        data='configs/defect_data.yaml',
        
        # TRAINING DURATION
        epochs=100,             # 100 loops over the data
        patience=20,            # Stop early if no improvement for 20 epochs
        
        # IMAGE SETTINGS
        imgsz=640,              # Standard YOLO resolution
        batch=16,               # Reduce to 8 if you run out of GPU memory
        
        # AUGMENTATION (The Secret Sauce for your Thesis)
        mosaic=1.0,             # Stitches 4 images together (Simulates Global View)
        mixup=0.1,              # Blends images slightly (Prevents overfitting)
        degrees=10.0,           # Rotates images +/- 10 degrees (Simulates camera tilt)
        fliplr=0.5,             # Flips left/right (Spaghetti looks same flipped)
        
        # HARDWARE
        device=0,               # Use GPU (Change to 'cpu' if you don't have NVIDIA)
        workers=4,              # Data loading threads
        
        # NAMING
        project='3d_print_monitor',
        name='yolov8s_centered_synthetic'
    )
    
    # 3. Validation
    print("Training Complete. Validating...")
    metrics = model.val()
    print(f"Map@50: {metrics.box.map50}")

    # 4. Export for Deployment (Raspberry Pi format)
    print("Exporting to ONNX...")
    model.export(format='onnx')

if __name__ == '__main__':
    train_model()