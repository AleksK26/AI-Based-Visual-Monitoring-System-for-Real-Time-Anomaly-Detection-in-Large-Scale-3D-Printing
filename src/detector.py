"""
The AI Inference Engine wrapper.
CURRENT ROLE: Loads the YOLOv8 model and runs detection on incoming frames.
FUTURE ROLE: Critical implementation of the 'Persistence Logic' (e.g., waiting for 10 consecutive detection frames) to filter out false positives before triggering a stop command to the printer.
"""