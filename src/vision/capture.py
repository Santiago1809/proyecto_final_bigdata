"""Camera capture wrapper using OpenCV.

Provides a context-managed VideoCapture that returns 640x480 BGR frames.
"""

import cv2


class CameraCapture:
    """Context manager wrapping OpenCV VideoCapture for frame acquisition.

    Captures frames at 640x480 resolution in BGR color space.

    Args:
        source: Camera device index (default 0) or video file path.

    Attributes:
        width: Configured frame width in pixels.
        height: Configured frame height in pixels.

    """

    def __init__(self, source: int | str = 0) -> None:
        self._source = source
        self._cap: cv2.VideoCapture | None = None
        self.width = 640
        self.height = 480

    def __enter__(self) -> "CameraCapture":
        """Open the camera device and configure resolution."""
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self._source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Read one warm-up frame to settle auto-exposure.
        self._cap.read()
        return self

    def __exit__(self, *exc) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> tuple[bool, cv2.Mat]:
        """Read the next frame from the camera.

        Returns:
            A tuple (success, frame) where success is True if the frame
            was read successfully, and frame is the BGR image (640x480).

        """
        if self._cap is None:
            raise RuntimeError("Camera not opened — use as context manager")
        return self._cap.read()
