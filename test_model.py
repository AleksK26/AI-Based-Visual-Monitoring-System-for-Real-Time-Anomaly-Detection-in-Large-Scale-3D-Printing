import os
import cv2
import torch
from collections import deque, defaultdict
from pathlib import Path
from ultralytics import YOLO

# Shared inference config — single source of truth (keeps offline test_model.py and the
# live src/detector.py / main.py path from drifting apart).
from src.detector import (
    BASE_CONF,
    IOU_THRESHOLD,
    TEMPORAL_WINDOW,
    TEMPORAL_MIN_HITS,
    CLASS_THRESHOLDS,
)

torch.backends.cudnn.benchmark = False

# --- CONFIGURATION ---
MODEL_PATH = r"runs\detect\3d_print_monitor\yolo11s_improved_v7\weights\best.pt"
TEST_FOLDER = "data/real_world_test"

# Box colours per class (BGR)
COLORS = {
    "Spaghetti":      (0,   0,   255),
    "Warping":        (0,   165, 255),
    "Layer_shifting": (0,   255, 255),
    "Stringing":      (255, 0,   0  ),
    "Cracking":       (0,   255, 0  ),
    "Blob_of_death":  (255, 0,   255),
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


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def merge_detections(detections, frame_w, frame_h, dilate=0.03):
    """Merge overlapping/adjacent same-class boxes into one.

    Fixes fragmented output (e.g. a single crack split into 16 boxes — the Roboflow
    cracking labels average ~7 boxes/image, so the model learned to over-emit). Same-class
    boxes whose boxes, dilated by `dilate` of the frame, intersect are unioned via union-find;
    each cluster becomes one box spanning its members with the max confidence.
    """
    mx, my = dilate * frame_w, dilate * frame_h
    by_cls = defaultdict(list)
    for d in detections:
        by_cls[d["class_name"]].append(d)

    out = []
    for cls, group in by_cls.items():
        n = len(group)
        parent = list(range(n))
        for i in range(n):
            ax1, ay1, ax2, ay2 = group[i]["box"]
            for j in range(i + 1, n):
                bx1, by1, bx2, by2 = group[j]["box"]
                if (ax1 - mx <= bx2 and bx1 <= ax2 + mx and
                        ay1 - my <= by2 and by1 <= ay2 + my):
                    parent[_find(parent, i)] = _find(parent, j)

        clusters = defaultdict(list)
        for i in range(n):
            clusters[_find(parent, i)].append(group[i])
        for members in clusters.values():
            out.append({
                "class_name": cls,
                "confidence": max(m["confidence"] for m in members),
                "box": [min(m["box"][0] for m in members), min(m["box"][1] for m in members),
                        max(m["box"][2] for m in members), max(m["box"][3] for m in members)],
            })
    return out


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
    detections = merge_detections(detections, frame.shape[1], frame.shape[0])
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
        stable_dets = merge_detections(stable_dets, frame.shape[1], frame.shape[0])
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
