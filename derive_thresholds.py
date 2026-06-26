"""
Re-derive per-class confidence thresholds from a trained run's val F1 curve.

ultralytics computes, per class, an F1-vs-confidence curve over 1000 points. The
confidence that MAXIMISES F1 is the natural per-class operating threshold. This script
prints those, plus the precision/recall there, so CLASS_THRESHOLDS in test_model.py can
be set from data instead of guesswork. (We still hand-tune from there: raise FP-prone
classes above the F1 knee, keep recall-first classes a touch below — see test_model.py.)

Usage:
  python derive_thresholds.py --run yolo11s_improved_v7
"""

import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = ["Spaghetti", "Warping", "Layer_shifting", "Stringing", "Cracking", "Blob_of_death"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_improved_v7")
    ap.add_argument("--device", default=0)
    args = ap.parse_args()
    device = args.device if args.device == "cpu" else int(args.device)

    weights = Path("runs/detect/3d_print_monitor") / args.run / "weights" / "best.pt"
    print(f"Validating {weights} ...")
    m = YOLO(str(weights)).val(device=device, verbose=False)

    f1 = np.asarray(m.box.f1_curve)      # (n_present_classes, 1000)
    p = np.asarray(m.box.p_curve)
    r = np.asarray(m.box.r_curve)
    conf = np.linspace(0, 1, f1.shape[1])
    idx = list(m.box.ap_class_index)     # maps curve row -> class id

    print(f"\n{'class':<16}{'f1_opt_conf':>12}{'F1':>8}{'P':>8}{'R':>8}")
    derived = {}
    for row, cls_id in enumerate(idx):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        j = int(np.argmax(f1[row]))
        derived[name] = round(float(conf[j]), 2)
        print(f"{name:<16}{conf[j]:>12.3f}{f1[row][j]:>8.3f}{p[row][j]:>8.3f}{r[row][j]:>8.3f}")

    print("\nF1-optimal CLASS_THRESHOLDS (starting point — hand-tune in test_model.py):")
    print("{")
    for name in CLASS_NAMES:
        if name in derived:
            print(f'    "{name}": {derived[name]:.2f},')
    print("}")


if __name__ == "__main__":
    main()
