"""
Side-by-side comparison of trained models — the YOLOv8s vs YOLO11s evidence for the thesis.

For each run it loads best.pt, validates on the held-out val split (overall + per-class
mAP), and times inference on CPU as a Raspberry-Pi-relevant latency proxy. Results are
printed and written to comparison_v8s_vs_11s.md so the numbers drop straight into Ch6/Ch8.

Usage:
  python compare_models.py
  python compare_models.py --runs yolov8s_improved_v6 yolo11s_improved_v6
  python compare_models.py --device cpu      # time on CPU (closest to the Pi)
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

RUNS_ROOT = Path("runs/detect/3d_print_monitor")
VAL_IMAGES = Path("data/processed/images/val")
CLASS_NAMES = ["Spaghetti", "Warping", "Layer_shifting", "Stringing", "Cracking", "Blob_of_death"]


def find_weights(run_name: str) -> Path | None:
    w = RUNS_ROOT / run_name / "weights" / "best.pt"
    return w if w.exists() else None


def time_inference(weights, device, n=30, imgsz=640) -> float:
    """Return mean ms/image over up to n val images (after a warm-up).

    Loads a FRESH model from `weights` rather than reusing the one passed to
    model.val(): on torch 2.7 a val() pass leaves the object in an inference-mode
    state that makes the subsequent predict() raise "Inference tensors do not
    track version counter". A fresh instance sidesteps that entirely.
    """
    imgs = []
    if VAL_IMAGES.exists():
        imgs = [p for p in VAL_IMAGES.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}][:n]
    if not imgs:
        return float("nan")
    model = YOLO(str(weights))
    model.predict(str(imgs[0]), device=device, imgsz=imgsz, verbose=False)  # warm-up
    t0 = time.perf_counter()
    for p in imgs:
        model.predict(str(p), device=device, imgsz=imgsz, verbose=False)
    return (time.perf_counter() - t0) / len(imgs) * 1000.0


def evaluate(run_name: str, device) -> dict | None:
    weights = find_weights(run_name)
    if weights is None:
        print(f"  ! no best.pt for '{run_name}' (looked in {RUNS_ROOT / run_name})")
        return None
    print(f"  evaluating {run_name} ...")
    model = YOLO(str(weights))
    m = model.val(verbose=False)

    per_class = {}
    try:
        for i, ap in enumerate(m.box.maps):           # mAP50-95 per class
            label = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
            per_class[label] = float(ap)
    except Exception:
        pass

    return {
        "run": run_name,
        "weights": str(weights),
        "map50": float(m.box.map50),
        "map": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class": per_class,
        "latency_ms": time_inference(weights, device),
        "device": str(device),
    }


def write_markdown(results, path="comparison_v8s_vs_11s.md"):
    lines = ["# Model comparison — YOLOv8s vs YOLO11s\n",
             f"_Latency timed on device={results[0]['device']} as a Raspberry Pi proxy._\n",
             "| Model | mAP@50 | mAP@50-95 | Precision | Recall | ms/img |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['run']} | {r['map50']:.4f} | {r['map']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['latency_ms']:.1f} |")

    lines.append("\n## Per-class mAP@50-95\n")
    header = "| Model | " + " | ".join(CLASS_NAMES) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(CLASS_NAMES) + 1))
    for r in results:
        row = [r["run"]] + [f"{r['per_class'].get(c, float('nan')):.3f}" for c in CLASS_NAMES]
        lines.append("| " + " | ".join(row) + " |")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")


def main():
    ap = argparse.ArgumentParser(description="Compare trained YOLO runs.")
    ap.add_argument("--runs", nargs="*",
                    default=["yolov8s_improved_v7", "yolo11s_improved_v7"])
    ap.add_argument("--device", default="cpu",
                    help="'cpu' (Pi proxy) or a GPU index like 0")
    args = ap.parse_args()
    device = args.device if args.device == "cpu" else int(args.device)

    print(f"Comparing runs: {args.runs}")
    results = [r for r in (evaluate(name, device) for name in args.runs) if r]
    if not results:
        print("No runs evaluated — train at least one model first.")
        return

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['run']:28s} mAP@50={r['map50']:.4f}  mAP={r['map']:.4f}  "
              f"P={r['precision']:.4f}  R={r['recall']:.4f}  {r['latency_ms']:.1f} ms/img")
    write_markdown(results)


if __name__ == "__main__":
    main()
