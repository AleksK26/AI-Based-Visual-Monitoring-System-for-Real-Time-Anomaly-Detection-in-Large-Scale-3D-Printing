import os
import cv2
import torch
from collections import deque
from pathlib import Path
from ultralytics import YOLO

torch.backends.cudnn.benchmark = False

# --- CONFIGURATION ---
MODEL_PATH = r"runs\detect\3d_print_monitor\yolov8s_improved_v52\weights\best.pt"
TEST_FOLDER = "data/real_world_test"

# Cast a wide net — per-class thresholds do the real filtering below.
BASE_CONF = 0.25
IOU_THRESHOLD = 0.50

# Temporal smoother for video: a class must appear in at least TEMPORAL_MIN_HITS
# of the last TEMPORAL_WINDOW frames to be drawn. Suppresses single-frame FP flicker.
TEMPORAL_WINDOW   = 7
TEMPORAL_MIN_HITS = 5

# Per-class thresholds derived from v6 (yolov8s_improved_v52) F1-curve + confusion matrix.
#   Global F1-optimal: 0.542 | mAP@50: 0.964
#   Spaghetti:      raised well above optimal — background FP 0.47 (regressed vs v5)
#   Warping:        stable — background FP 0.19
#   Layer_shifting: near F1-optimal — near-perfect class (diagonal 0.99)
#   Stringing:      lowered — real images halved background FP (0.47→0.22), recover recall
#   Cracking:       near F1-optimal — minor FP (0.06)
CLASS_THRESHOLDS = {
    "Spaghetti":      0.75,   # background FP 0.47 — push high to suppress real-world FP
    "Warping":        0.55,   # background FP 0.19 — stable
    "Layer_shifting": 0.50,   # near-perfect, minor FP (0.06)
    "Stringing":      0.55,   # lowered too aggressively — real-world FP, raise back up
    "Cracking":       0.70,   # Kaggle bias: gold object → Cracking FP, raise until retrain fixes it
}

# Box colours per class (BGR)
COLORS = {
    "Spaghetti":      (0,   0,   255),
    "Warping":        (0,   165, 255),
    "Layer_shifting": (0,   255, 255),
    "Stringing":      (255, 0,   0  ),
    "Cracking":       (0,   255, 0  ),
}


def filter_detections(result, names):
    """
    Apply per-class confidence thresholds to a single YOLO result.
    Returns a list of dicts with keys: class_name, confidence, box (xyxy).
    """
    kept = []
    for box in result.boxes:
        cls_name = names[int(box.cls[0])]
        conf = float(box.conf[0])
        threshold = CLASS_THRESHOLDS.get(cls_name, BASE_CONF)
        if conf >= threshold:
            kept.append({
                "class_name": cls_name,
                "confidence": conf,
                "box": [int(v) for v in box.xyxy[0].tolist()],
            })
    return kept


def draw_detections(frame, detections):
    """Draw filtered detections onto a frame in-place."""
    fh, fw = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        color = COLORS.get(det["class_name"], (128, 128, 128))
        # Thicker box for visibility on video
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        font_scale = 1.4
        thickness = 3
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Clamp label so it never clips off the top of the frame
        label_y = max(y1, th + 14)
        cv2.rectangle(frame, (x1, label_y - th - 10), (x1 + tw + 8, label_y + baseline - 2), color, -1)
        cv2.putText(frame, label, (x1 + 4, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return frame


def process_image(model, path, out_dir):
    frame = cv2.imread(path)
    results = model.predict(
        source=frame,
        conf=BASE_CONF,
        iou=IOU_THRESHOLD,
        agnostic_nms=True,
        verbose=False,
    )
    detections = filter_detections(results[0], model.names)
    draw_detections(frame, detections)
    out_path = out_dir / Path(path).name
    cv2.imwrite(str(out_path), frame)
    return detections


def process_video(model, path, out_dir, total_frames=0):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = Path(path).stem
    out_path = out_dir / f"{stem}.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h),
    )
    total_detections = 0
    frame_idx = 0
    # Rolling window: each entry is the set of class names detected in that frame
    history = deque(maxlen=TEMPORAL_WINDOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(
            source=frame,
            conf=BASE_CONF,
            iou=IOU_THRESHOLD,
            agnostic_nms=True,
            verbose=False,
        )
        raw_dets = filter_detections(results[0], model.names)
        history.append({d["class_name"] for d in raw_dets})
        # Count how many of the last N frames each class appeared in
        class_hits = {}
        for frame_classes in history:
            for cls in frame_classes:
                class_hits[cls] = class_hits.get(cls, 0) + 1
        # Only draw detections for classes stable across the window
        stable_dets = [d for d in raw_dets if class_hits.get(d["class_name"], 0) >= TEMPORAL_MIN_HITS]
        draw_detections(frame, stable_dets)
        writer.write(frame)
        total_detections += len(stable_dets)
        frame_idx += 1
        if frame_idx % 50 == 0:
            pct = f"{100*frame_idx//total_frames}%" if total_frames else f"{frame_idx} frames"
            print(f"    ... {pct}", flush=True)
    cap.release()
    writer.release()
    return total_detections, frame_idx


def run_test():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model not found at {MODEL_PATH}")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (no GPU)"))

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    files = [
        f for f in os.listdir(TEST_FOLDER)
        if Path(f).suffix.lower() in image_exts | video_exts
    ]
    if not files:
        print(f"No images or videos found in {TEST_FOLDER}")
        return

    # Auto-increment output folder (real_world_test, real_world_test2, ...)
    base = Path("runs/detect/real_world_test")
    out_dir = base
    n = 2
    while out_dir.exists():
        out_dir = Path(f"runs/detect/real_world_test{n}")
        n += 1
    out_dir.mkdir(parents=True)

    print(f"Found {len(files)} file(s) — output → {out_dir}")
    print(f"Base conf: {BASE_CONF}  |  Per-class thresholds: {CLASS_THRESHOLDS}")
    print("-" * 60)

    for fname in sorted(files):
        fpath = os.path.join(TEST_FOLDER, fname)
        ext = Path(fname).suffix.lower()

        if ext in image_exts:
            dets = process_image(model, fpath, out_dir)
            if dets:
                summary = ", ".join(
                    f"{d['class_name']}({d['confidence']:.2f})" for d in dets
                )
                print(f"  {fname}: {summary}")
            else:
                print(f"  {fname}: no detections")

        elif ext in video_exts:
            cap_check = cv2.VideoCapture(os.path.join(TEST_FOLDER, fname))
            total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_w = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_check.release()
            print(f"  {fname}: {total_frames} frames @ {vid_w}x{vid_h}, processing...")
            total, frames = process_video(model, fpath, out_dir, total_frames)
            print(f"  {fname}: done — {frames} frames, {total} total detections")

    print("-" * 60)
    print(f"Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    run_test()
