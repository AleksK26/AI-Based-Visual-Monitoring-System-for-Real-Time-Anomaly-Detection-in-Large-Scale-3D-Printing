"""
The main entry point of the monitoring system.
CURRENT ROLE: Orchestrates Camera + Detector in mock mode (video file source)
to simulate the full detection pipeline without hardware.
FUTURE ROLE: Will run on the Edge Device (Raspberry Pi), managing the
real-time loop between the live camera feed, AI inference, and the
printer controller to trigger emergency pauses.

Usage:
    # Mock mode (test with a video file):
    python main.py --source data/real_world_test/my_print.mp4

    # Live mode (Raspberry Pi with USB webcam):
    python main.py --source 0
"""

import argparse
import time

import cv2

from src.camera import Camera
from src.detector import Detector, TEMPORAL_MIN_HITS
from src.printer_interface import PrinterInterface

# --- CONFIGURATION ---
# Model + per-class thresholds + temporal smoother live in src/detector.py (single source
# of truth, shared with test_model.py). main.py only owns the loop and display.
TARGET_FPS = 1              # How many frames per second to analyse (1 is enough)
DISPLAY    = True           # Show annotated frames in a window (set False on Pi)


def parse_args():
    parser = argparse.ArgumentParser(description="3D Print Defect Monitor")
    parser.add_argument(
        "--source", default="data/real_world_test",
        help="Camera index (0) or path to video/image file."
    )
    parser.add_argument(
        "--model", default=Detector.DEFAULT_MODEL,
        help="Path to trained YOLO .pt weights."
    )
    parser.add_argument(
        "--min-hits", type=int, default=TEMPORAL_MIN_HITS,
        help="Frames (of the rolling window) a class must appear in to trigger a pause."
    )
    return parser.parse_args()


def draw_detections(frame, detections, class_hits, min_hits):
    """Overlay bounding boxes and a class-aware status HUD on the frame."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Per-class hit counters (e.g. "Spaghetti 3/5"); turns orange as a class approaches
    # the pause threshold, red once it crosses it.
    if class_hits:
        y = 30
        for cls, hits in sorted(class_hits.items(), key=lambda kv: -kv[1]):
            if hits >= min_hits:
                color = (0, 0, 255)
            elif hits > 0:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            cv2.putText(frame, f"{cls}: {hits}/{min_hits}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y += 32
    else:
        cv2.putText(frame, "Monitoring - clean", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


def main():
    args = parse_args()

    # Try to convert source to int (live camera index)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    detector = Detector(
        model_path=args.model,
        min_hits=args.min_hits,
    )
    printer = PrinterInterface()
    frame_interval = 1.0 / TARGET_FPS

    print(f"[Main] Starting monitoring — source: {source}")
    print(f"[Main] Temporal smoother: {args.min_hits}/{detector.window} frames per class")
    print(f"[Main] Press 'q' to quit.\n")

    paused = False

    with Camera(source=source) as cam:
        while True:
            loop_start = time.time()

            frame = cam.grab_frame()
            if frame is None:
                print("[Main] Stream ended.")
                break

            should_pause, detections = detector.trigger(frame)

            if detections:
                names = [d["class_name"] for d in detections]
                print(f"[Detector] {names} — hits {detector.class_hits}")

            if should_pause and not paused and printer.is_printing():
                print(f"[Main] *** DEFECT CONFIRMED ({detector.triggered_classes}) — Pausing printer! ***")
                success = printer.pause_print()
                if success:
                    paused = True
                    detector.reset()
                    print("[Main] Printer paused. Monitoring continues.")

            if DISPLAY:
                annotated = draw_detections(
                    frame.copy(), detections,
                    detector.class_hits, args.min_hits
                )
                cv2.imshow("3D Print Monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Throttle to TARGET_FPS
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    if DISPLAY:
        cv2.destroyAllWindows()

    print("[Main] Monitoring stopped.")


if __name__ == "__main__":
    main()
