"""
The AI Inference Engine wrapper.
CURRENT ROLE: Loads the YOLOv8 model and runs detection on incoming frames,
with a persistence/debounce filter to suppress single-frame false positives.
FUTURE ROLE: Fine-tune confidence threshold and persistence window based on
real-world false positive/negative rates observed during live monitoring.
"""

from ultralytics import YOLO
from collections import defaultdict, deque
import os

# --- LIVE INFERENCE CONFIG (canonical — test_model.py imports these to stay in sync) ---
# Cast a wide net at BASE_CONF, then CLASS_THRESHOLDS does the real per-class filtering.
BASE_CONF = 0.25
IOU_THRESHOLD = 0.50

# Per-class temporal smoother: a class must appear in at least TEMPORAL_MIN_HITS of the
# last TEMPORAL_WINDOW frames before it can trigger a pause. Suppresses single-frame FP.
TEMPORAL_WINDOW = 7
TEMPORAL_MIN_HITS = 5

# Per-class confidence thresholds — v7 (yolo11s_improved_v7). F1-optimal from
# derive_thresholds.py, then hand-tuned: lift FP-prone classes above the F1 knee, keep
# FN-prone classes recall-first, keep Layer_shifting deployment-safe.
#   F1-opt: Spag 0.48 | Warp 0.63 | Layer 0.13 | String 0.31 | Crack 0.42 | Blob 0.51
CLASS_THRESHOLDS = {
    "Spaghetti":      0.55,
    "Warping":        0.60,
    "Layer_shifting": 0.50,
    "Stringing":      0.40,
    "Cracking":       0.45,
    "Blob_of_death":  0.50,
}


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def merge_detections(detections, frame_w, frame_h, dilate=0.03):
    """Merge overlapping/adjacent same-class boxes into one (mirrors test_model.py).

    Stops a single defect (e.g. one crack) being reported as many fragments, which would
    otherwise inflate the persistence counter and clutter the live overlay. Same-class boxes
    whose boxes — dilated by `dilate` of the frame — intersect are unioned; each cluster
    becomes one box spanning its members with the max confidence.
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
            best = max(members, key=lambda m: m["confidence"])
            out.append({
                "class_id": best["class_id"],
                "class_name": cls,
                "confidence": best["confidence"],
                "box": [min(m["box"][0] for m in members), min(m["box"][1] for m in members),
                        max(m["box"][2] for m in members), max(m["box"][3] for m in members)],
            })
    return out


class Detector:
    """
    Wraps YOLOv8 inference with a persistence filter.

    The persistence filter prevents spurious printer pauses by requiring
    a defect to appear in N consecutive frames before raising an alert.
    At 1 FPS, PERSISTENCE_FRAMES=5 means a defect must be visible for
    5 seconds continuously before the printer is paused.
    """

    DEFAULT_MODEL = r"runs\detect\3d_print_monitor\yolo11s_improved_v7\weights\best.pt"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        base_conf: float = BASE_CONF,
        iou: float = IOU_THRESHOLD,
        window: int = TEMPORAL_WINDOW,
        min_hits: int = TEMPORAL_MIN_HITS,
        class_thresholds: dict | None = None,
    ):
        """
        Args:
            model_path: Path to trained .pt weights file.
            base_conf: Wide-net confidence floor for the YOLO predict call. The real
                filtering is done per-class by `class_thresholds`, so keep this low.
            iou: IOU threshold for non-maximum suppression (0–1).
            window: Size of the rolling temporal window (frames).
            min_hits: A class must appear in at least this many of the last `window`
                frames before it can trigger a pause.
            class_thresholds: Per-class confidence thresholds. Defaults to the canonical
                CLASS_THRESHOLDS (shared with test_model.py).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found: {model_path}\n"
                "Run python train.py first."
            )

        print(f"[Detector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.base_conf = base_conf
        self.iou = iou
        self.window = window
        self.min_hits = min_hits
        self.class_thresholds = class_thresholds or dict(CLASS_THRESHOLDS)

        # Rolling window: each entry is the set of class names detected in that frame.
        self._history = deque(maxlen=window)
        # Exposed for HUD/UI: per-class hit counts in the current window, and the
        # classes that have crossed min_hits this frame (i.e. are triggering a pause).
        self.class_hits: dict[str, int] = {}
        self.triggered_classes: list[str] = []

    def detect(self, frame):
        """
        Run inference on a single BGR frame (numpy array from cv2).

        Casts a wide net at base_conf, then applies per-class thresholds and merges
        fragmented same-class boxes — mirrors test_model.py exactly.

        Returns:
            list[dict]: Each entry has keys:
                - 'class_id'   (int)
                - 'class_name' (str)
                - 'confidence' (float)
                - 'box'        (list[float]): [x1, y1, x2, y2] in pixels
        """
        results = self.model.predict(
            source=frame,
            conf=self.base_conf,
            iou=self.iou,
            agnostic_nms=True,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_name = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if conf >= self.class_thresholds.get(cls_name, self.base_conf):
                    detections.append({
                        "class_id":   int(box.cls[0]),
                        "class_name": cls_name,
                        "confidence": conf,
                        "box":        box.xyxy[0].tolist(),
                    })

        # Merge fragmented same-class boxes (one defect -> one box).
        h, w = frame.shape[:2]
        return merge_detections(detections, w, h)

    def trigger(self, frame) -> tuple[bool, list]:
        """
        Detect and apply the per-class temporal smoother.

        A class triggers a pause only when it is present this frame AND has appeared in
        at least `min_hits` of the last `window` frames. This is robust to single-frame
        dropouts (unlike a strict consecutive counter) and is class-aware, so a flickering
        Stringing FP can't be "topped up" by an unrelated Cracking hit.

        Returns:
            (should_pause, detections)
            - should_pause (bool): True when any class crosses min_hits this frame.
            - detections (list[dict]): This frame's filtered+merged detections (for display).
        """
        detections = self.detect(frame)

        self._history.append({d["class_name"] for d in detections})
        hits: dict[str, int] = {}
        for frame_classes in self._history:
            for cls in frame_classes:
                hits[cls] = hits.get(cls, 0) + 1
        self.class_hits = hits

        current = {d["class_name"] for d in detections}
        self.triggered_classes = [c for c in current if hits.get(c, 0) >= self.min_hits]
        should_pause = len(self.triggered_classes) > 0
        return should_pause, detections

    def reset(self):
        """Clear the temporal window (call after printer is paused)."""
        self._history.clear()
        self.class_hits = {}
        self.triggered_classes = []
