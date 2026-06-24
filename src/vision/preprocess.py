"""Image preprocessing utilities for the vision pipeline.

Resizes frames to 300x300 and converts them to DNN-compatible blobs
for MobileNet SSD inference via OpenCV.
"""

import cv2
import numpy as np

_INPUT_SIZE = (300, 300)
_MEAN = (127.5, 127.5, 127.5)
_SCALE_FACTOR = 1.0 / 127.5


def resize(frame: np.ndarray, size: tuple[int, int] = _INPUT_SIZE) -> np.ndarray:
    """Resize a frame to the model's expected input dimensions.

    Args:
        frame: Input BGR image of any size.
        size: Target (width, height) in pixels. Defaults to (300, 300).

    Returns:
        Resized BGR image.

    """
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


def to_blob(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame into a 4D blob suitable for OpenCV DNN inference.

    The blob uses the MobileNet SSD normalization scheme:
    scale = 1/127.5, mean = (127.5, 127.5, 127.5).

    Args:
        frame: Resized BGR image (300x300).

    Returns:
        4D numpy array of shape (1, 3, 300, 300) ready for net.forward().

    """
    return cv2.dnn.blobFromImage(
        frame,
        scalefactor=_SCALE_FACTOR,
        size=_INPUT_SIZE,
        mean=_MEAN,
        swapRB=False,
        crop=False,
    )
