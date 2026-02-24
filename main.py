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
from src.detector import Detector
from src.printer_interface import PrinterInterface

# --- CONFIGURATION ---
DEFAULT_MODEL  = r"runs\detect\3d_print_monitor\yolov8s_centered_synthetic2\weights\best.pt"
CONF_THRESHOLD = 0.55       # Confidence to count a detection
PERSISTENCE    = 5          # Consecutive frames needed before pausing printer
TARGET_FPS     = 1          # How many frames per second to analyse (1 is enough)
DISPLAY        = True       # Show annotated frames in a window (set False on Pi)


def parse_args():
    parser = argparse.ArgumentParser(description="3D Print Defect Monitor")
    parser.add_argument(
        "--source", default="data/real_world_test",
        help="Camera index (0) or path to video/image file."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Path to trained YOLOv8 .pt weights."
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
        help="Confidence threshold (0–1)."
    )
    parser.add_argument(
        "--persistence", type=int, default=PERSISTENCE,
        help="Consecutive frames required to trigger a pause."
    )
    return parser.parse_args()


def draw_detections(frame, detections, consecutive_hits, persistence):
    """Overlay bounding boxes and status HUD on the frame."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    status = f"Hits: {consecutive_hits}/{persistence}"
    color = (0, 255, 0) if consecutive_hits == 0 else (0, 165, 255)
    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
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
        conf=args.conf,
        persistence_frames=args.persistence,
    )
    printer = PrinterInterface()
    frame_interval = 1.0 / TARGET_FPS

    print(f"[Main] Starting monitoring — source: {source}")
    print(f"[Main] Persistence filter: {args.persistence} consecutive frames")
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
                print(f"[Detector] Frame hit {detector.consecutive_hits}/{args.persistence} — {names}")

            if should_pause and not paused and printer.is_printing():
                print("[Main] *** DEFECT CONFIRMED — Pausing printer! ***")
                success = printer.pause_print()
                if success:
                    paused = True
                    detector.reset()
                    print("[Main] Printer paused. Monitoring continues.")

            if DISPLAY:
                annotated = draw_detections(
                    frame.copy(), detections,
                    detector.consecutive_hits, args.persistence
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
