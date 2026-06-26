# Model comparison — YOLOv8s vs YOLO11s

_Latency timed on device=cpu as a Raspberry Pi proxy._

| Model | mAP@50 | mAP@50-95 | Precision | Recall | ms/img |
|---|---|---|---|---|---|
| yolov8s_improved_v7 | 0.8620 | 0.5904 | 0.8771 | 0.8164 | 95.0 |
| yolo11s_improved_v7 | 0.8866 | 0.6194 | 0.8815 | 0.8306 | 124.2 |

## Per-class mAP@50-95

| Model | Spaghetti | Warping | Layer_shifting | Stringing | Cracking | Blob_of_death |
|---|---|---|---|---|---|---|
| yolov8s_improved_v7 | 0.481 | 0.773 | 0.871 | 0.384 | 0.523 | 0.511 |
| yolo11s_improved_v7 | 0.517 | 0.782 | 0.926 | 0.403 | 0.542 | 0.546 |
