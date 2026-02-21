"""
Handles video stream acquisition.
CURRENT ROLE: Supports both Mock mode (reads from a video file) and
Live mode (reads from a USB/webcam device index).
FUTURE ROLE: Will interface with cv2.VideoCapture(0) to grab frames
from the USB camera mounted on the large-scale printer frame,
ensuring a global view of the print bed.
"""

import cv2


class Camera:
    """
    Abstracts frame acquisition from either a live webcam or a mock video file.

    Usage — mock mode (for testing without hardware):
        cam = Camera(source="data/real_world_test/my_print.mp4")

    Usage — live mode (Raspberry Pi with USB webcam):
        cam = Camera(source=0)
    """

    def __init__(self, source=0, target_fps: int = 1):
        """
        Args:
            source: Device index (int) for a live camera, or a file path (str)
                    for mock/test mode.
            target_fps: How many frames per second to sample. Frames between
                    samples are skipped to reduce CPU load on the Pi.
                    Default 1 FPS is enough to catch defects early.
        """
        self.source = source
        self.target_fps = target_fps
        self._cap = None

    def open(self):
        """Open the video capture. Call before grab_frame()."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source: {self.source}\n"
                "For a webcam, make sure it is connected and try index 0 or 1.\n"
                "For a file, check the path exists."
            )
        print(f"[Camera] Opened source: {self.source}")

    def grab_frame(self):
        """
        Grab one frame from the source.

        Returns:
            frame (numpy.ndarray | None): BGR frame, or None if the stream ended.
        """
        if self._cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")

        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def close(self):
        """Release the video capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            print("[Camera] Closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
