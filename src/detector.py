"""
The AI Inference Engine wrapper.
CURRENT ROLE: Loads the YOLOv8 model and runs detection on incoming frames,
with a persistence/debounce filter to suppress single-frame false positives.
FUTURE ROLE: Fine-tune confidence threshold and persistence window based on
real-world false positive/negative rates observed during live monitoring.
"""

from ultralytics import YOLO
import os


class Detector:
    """
    Wraps YOLOv8 inference with a persistence filter.

    The persistence filter prevents spurious printer pauses by requiring
    a defect to appear in N consecutive frames before raising an alert.
    At 1 FPS, PERSISTENCE_FRAMES=5 means a defect must be visible for
    5 seconds continuously before the printer is paused.
    """

    DEFAULT_MODEL = r"runs\detect\3d_print_monitor\yolov8s_centered_synthetic2\weights\best.pt"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf: float = 0.55,
        iou: float = 0.50,
        persistence_frames: int = 5,
    ):
        """
        Args:
            model_path: Path to trained .pt weights file.
            conf: Minimum confidence to count a detection (0–1).
            iou: IOU threshold for non-maximum suppression (0–1).
            persistence_frames: Number of consecutive frames a defect must
                appear in before trigger() returns True.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found: {model_path}\n"
                "Run python train.py first."
            )

        print(f"[Detector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.persistence_frames = persistence_frames

        # Rolling counter — increments each frame a defect is detected,
        # resets to 0 on any clean frame.
        self._consecutive_hits = 0

    def detect(self, frame):
        """
        Run inference on a single BGR frame (numpy array from cv2).

        Returns:
            list[dict]: Each entry has keys:
                - 'class_id'   (int)
                - 'class_name' (str)
                - 'confidence' (float)
                - 'box'        (list[float]): [x1, y1, x2, y2] in pixels
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            agnostic_nms=True,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_id":   int(box.cls[0]),
                    "class_name": self.model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "box":        box.xyxy[0].tolist(),
                })

        return detections

    def trigger(self, frame) -> tuple[bool, list]:
        """
        Detect and apply the persistence filter.

        Returns:
            (should_pause, detections)
            - should_pause (bool): True only when defect seen for
              persistence_frames consecutive frames.
            - detections (list[dict]): Raw detections for this frame
              (useful for display even when not yet triggering).
        """
        detections = self.detect(frame)

        if detections:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        should_pause = self._consecutive_hits >= self.persistence_frames
        return should_pause, detections

    def reset(self):
        """Reset the persistence counter (call after printer is paused)."""
        self._consecutive_hits = 0

    @property
    def consecutive_hits(self) -> int:
        return self._consecutive_hits
