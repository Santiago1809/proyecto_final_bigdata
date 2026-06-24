"""Bottle classifier using MobileNet SSD via OpenCV DNN.

Detects bottles (COCO class 39) in camera frames with a configurable
confidence threshold.

Typical usage::

    model = BottleClassifier(threshold=0.7)
    model.load_model("models/MobileNetSSD_deploy.caffemodel",
                     "models/MobileNetSSD_deploy.prototxt")

    is_bottle, confidence = model.predict(frame)
"""

import cv2
import numpy as np

from src.vision.preprocess import resize, to_blob

# COCO class ID for "bottle" in MobileNet SSD
_COCO_BOTTLE_ID = 39


class BottleClassifier:
    """MobileNet SSD-based bottle detector with configurable threshold.

    Args:
        threshold: Minimum confidence score (0..1) to consider a
            detection valid. Default 0.7 per system spec.

    """

    def __init__(self, threshold: float = 0.7) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self._threshold = threshold
        self._net: cv2.dnn.Net | None = None

    @property
    def threshold(self) -> float:
        """Return the current confidence threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {value}")
        self._threshold = value

    def load_model(self, caffemodel_path: str, prototxt_path: str) -> None:
        """Load a pre-trained Caffe model into the OpenCV DNN backend.

        Args:
            caffemodel_path: Path to the .caffemodel weights file.
            prototxt_path: Path to the .prototxt architecture file.

        Raises:
            RuntimeError: If the model files cannot be loaded.

        """
        self._net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        if self._net.empty():
            raise RuntimeError(
                f"Failed to load model: {caffemodel_path!r}, {prototxt_path!r}"
            )

    def predict(self, frame: np.ndarray) -> tuple[bool, float]:
        """Run inference on a single frame and return the bottle prediction.

        Preprocessing (resize + blob conversion) is applied automatically.

        Args:
            frame: BGR image (any size; will be resized to 300x300).

        Returns:
            A tuple (is_bottle, confidence):
                - is_bottle: True if a bottle was detected above threshold.
                - confidence: Highest confidence score found for class 39,
                  or 0.0 if no bottle detected.

        """
        if self._net is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        resized = resize(frame)
        blob = to_blob(resized)
        self._net.setInput(blob)
        detections = self._net.forward()

        return self._parse_detections(detections)

    def _parse_detections(self, detections: np.ndarray) -> tuple[bool, float]:
        """Parse the raw DNN output and return (is_bottle, best_confidence).

        Args:
            detections: Shape (1, 1, N, 7) where each detection is
                [img_id, class_id, confidence, x1, y1, x2, y2].

        Returns:
            (is_bottle, confidence) tuple.

        """
        best_confidence = 0.0

        for i in range(detections.shape[2]):
            confidence = round(float(detections[0, 0, i, 2]), 6)
            class_id = int(detections[0, 0, i, 1])

            if class_id == _COCO_BOTTLE_ID and confidence > best_confidence:
                best_confidence = confidence

        is_bottle = best_confidence >= self._threshold
        return is_bottle, best_confidence
