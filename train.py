"""
Model training pipeline — now architecture-parametrised for the v8s vs 11s comparison.

CURRENT ROLE: trains a YOLO detector on the synthetic + real (scraped) defect dataset.
Run it twice with different --model to produce the two-model comparison written up in
the thesis (YOLOv8s vs YOLO11s):

    python train.py --model yolov8s.pt --name yolov8s_improved_v6
    python train.py --model yolo11s.pt  --name yolo11s_improved_v6

Then `python compare_models.py` evaluates both runs side by side (mAP + Pi-relevant
latency) so you can justify which one ships.

Key v6 change vs v5: aggressive HSV-hue augmentation (hsv_h 0.015 -> 0.30) to break the
colour<->class correlation that made v5 fire "gold = Cracking". Pair this with the
colour-diverse scraped data for the actual fix.

FUTURE ROLE: transfer-learn the winning architecture on real captures from the printer.
"""

import argparse

from ultralytics import YOLO


def train_model(model_weights="yolov8s.pt", run_name="yolov8s_improved_v6",
                epochs=50, batch=16, imgsz=640, device=0, export_ncnn=True):
    # 1. Load the chosen architecture from its ImageNet-pretrained weights.
    #    yolov8s.pt and yolo11s.pt are drop-in compatible through the same API.
    print(f"Loading {model_weights} ...")
    model = YOLO(model_weights)

    # 2. Train.
    print(f"Starting training: {run_name}")
    model.train(
        data="configs/defect_data.yaml",

        # Duration
        epochs=epochs,
        patience=15,

        # Image settings
        imgsz=imgsz,
        batch=batch,

        # Augmentation — real-world deployment conditions
        mosaic=0.5,
        mixup=0.1,
        degrees=5.0,
        fliplr=0.5,

        # COLOUR / LIGHTING — v6: aggressively randomise HUE so the model becomes
        # invariant to filament colour (breaks the v5 "gold = defect" correlation).
        hsv_h=0.30,             # was 0.015 — the key debiasing knob
        hsv_s=0.70,
        hsv_v=0.40,

        # Hardware
        device=device,
        workers=4,

        # Naming — ultralytics auto-prepends runs/detect/, so pass just the project.
        project="3d_print_monitor",
        name=run_name,
    )

    # 3. Validate on the held-out val split.
    print("Training complete. Validating on held-out val set...")
    metrics = model.val()
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # 4. Export for edge deployment.
    #    ONNX is portable; NCNN is markedly faster on the Raspberry Pi's ARM CPU and
    #    is the format we actually benchmark for the Pi target.
    print("Exporting to ONNX...")
    model.export(format="onnx")
    if export_ncnn:
        print("Exporting to NCNN (Raspberry Pi target)...")
        try:
            model.export(format="ncnn")
        except Exception as e:
            print(f"  NCNN export failed (continue anyway): {e}")

    print(f"Done. See runs/detect/3d_print_monitor/{run_name}/")
    return model


def main():
    ap = argparse.ArgumentParser(description="Train a YOLO defect detector.")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="pretrained weights / architecture, e.g. yolov8s.pt or yolo11s.pt")
    ap.add_argument("--name", default=None,
                    help="run name (default derived from --model)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16, help="reduce to 8 if low on GPU memory")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0, help="GPU index, or 'cpu'")
    ap.add_argument("--no-ncnn", action="store_true", help="skip the NCNN export")
    args = ap.parse_args()

    if args.name is None:
        stem = args.model.replace(".pt", "")
        args.name = f"{stem}_improved_v6"

    device = args.device if args.device == "cpu" else int(args.device)
    train_model(
        model_weights=args.model,
        run_name=args.name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        export_ncnn=not args.no_ncnn,
    )


if __name__ == "__main__":
    main()
