"""Image preprocessing utilities for the vision pipeline.

Resizes frames to 300x300 and converts them to DNN-compatible blobs
for MobileNet SSD inference via OpenCV (legacy).  Also provides
:func:`to_tf_input` for the TensorFlow-based classifier.
"""

import cv2
import numpy as np
import tensorflow as tf

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


def to_tf_input(frame: np.ndarray) -> np.ndarray:
    """Preprocess a BGR frame for MobileNetV2 inference.

    Resizes to 224×224 with bilinear interpolation, converts BGR→RGB,
    and applies ``mobilenet_v2.preprocess_input`` (scales pixels to
    ``[-1, 1]``).

    Args:
        frame: BGR image from camera (any size).

    Returns:
        ``float32`` array of shape ``(224, 224, 3)`` ready for model
        input.  The caller should add a batch dimension before inference.

    """
    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return tf.keras.applications.mobilenet_v2.preprocess_input(
        rgb.astype(np.float32),
    )
