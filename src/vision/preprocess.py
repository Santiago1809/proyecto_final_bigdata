"""Image preprocessing utilities for the vision pipeline.

Provides :func:`to_tf_input` for the TensorFlow-based classifier.
"""

import cv2
import numpy as np


def to_tf_input(frame: np.ndarray) -> np.ndarray:
    """Preprocess a BGR frame for the trained MobileNetV2 model.

    Resizes to 224×224 with bilinear interpolation, converts BGR→RGB,
    and casts to ``float32``.  **Does NOT** normalise — the model
    already has built-in preprocessing (``true_divide / 128`` →
    ``subtract 1``).

    Args:
        frame: BGR image from camera (any size).

    Returns:
        ``float32`` array of shape ``(224, 224, 3)`` with raw pixel
        values in ``[0, 255]``.  The caller should add a batch
        dimension before inference.

    """
    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)
