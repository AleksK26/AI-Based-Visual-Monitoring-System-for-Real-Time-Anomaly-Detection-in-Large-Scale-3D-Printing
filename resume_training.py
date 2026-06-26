"""
Resume / restart the yolo11s_improved_v6 training after an interruption (e.g. travel).

Background: yolov8s_improved_v6 already FINISHED (best.pt saved, mAP50 0.911).
yolo11s was interrupted partway. It ran very slowly (~27 min/epoch) because at
batch=16 it spilled past the 4 GB GPU into shared system memory.

TWO options — pick with the flag:

  # FAST (recommended): quit Ollama first, then restart yolo11s fresh at batch=8.
  # Fits under 4 GB -> no spillover -> ~4-5x faster. Loses the few early epochs.
  python resume_training.py --restart

  # EXACT RESUME: continue last.pt from the last completed epoch, keeping batch=16.
  # No epochs lost, but stays slow (~20h) unless you also freed VRAM.
  python resume_training.py --resume
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

RUN = Path("runs/detect/3d_print_monitor/yolo11s_improved_v6")
LAST = RUN / "weights" / "last.pt"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--resume", action="store_true",
                   help="continue last.pt from last completed epoch (batch=16, slow)")
    g.add_argument("--restart", action="store_true",
                   help="fresh yolo11s run at batch=8 (recommended, much faster)")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if args.resume:
        if not LAST.exists():
            print(f"No checkpoint at {LAST} — use --restart instead.")
            return
        print(f"Resuming from {LAST} (keeps original batch/args) ...")
        YOLO(str(LAST)).train(resume=True)
    else:
        print(f"Fresh yolo11s run at batch={args.batch} (close Ollama first for max speed) ...")
        YOLO("yolo11s.pt").train(
            data="configs/defect_data.yaml",
            epochs=50, batch=args.batch, imgsz=640, device=0, patience=15,
            hsv_h=0.30, hsv_s=0.70, hsv_v=0.40,           # v6 colour debias
            degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5, mosaic=0.5, mixup=0.1,
            project="3d_print_monitor", name="yolo11s_improved_v6_b8",
        )


if __name__ == "__main__":
    main()
