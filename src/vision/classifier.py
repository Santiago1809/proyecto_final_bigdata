"""Bottle classifier using MobileNet SSD via OpenCV DNN.

Detects bottles (PASCAL VOC class 5) in camera frames with a configurable
confidence threshold. Returns the bounding box coordinates for display.

Typical usage::

    model = BottleClassifier(threshold=0.7)
    model.load_model("models/MobileNetSSD_deploy.caffemodel",
                     "models/MobileNetSSD_deploy.prototxt")

    is_bottle, confidence, box = model.predict(frame)
    if is_bottle:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
"""

import cv2
import numpy as np

from src.vision.preprocess import resize, to_blob

# PASCAL VOC class ID for "bottle" (class 5 out of 20 VOC classes)
_BOTTLE_CLASS_ID = 5

# Type alias: (x1, y1, x2, y2) in original frame pixel coordinates
_Box = tuple[int, int, int, int]


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

    def predict(self, frame: np.ndarray) -> tuple[bool, float, _Box | None]:
        """Run inference and return (is_bottle, confidence, bounding_box).

        Preprocessing (resize + blob conversion) is applied automatically.

        Args:
            frame: BGR image (any size; will be resized to 300x300).

        Returns:
            A tuple (is_bottle, confidence, box):
                - is_bottle: True if a bottle was detected above threshold.
                - confidence: Highest confidence score for PASCAL VOC
                  class 5 (bottle), or 0.0 if no bottle detected.
                - box: (x1, y1, x2, y2) pixel coordinates in the original
                  frame, or None if no bottle detected.

        """
        if self._net is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        resized = resize(frame)
        blob = to_blob(resized)
        self._net.setInput(blob)
        detections = self._net.forward()

        return self._parse_detections(detections, frame.shape)

    def _parse_detections(
        self, detections: np.ndarray, frame_shape: tuple[int, ...]
    ) -> tuple[bool, float, _Box | None]:
        """Parse raw DNN output into (is_bottle, confidence, box).

        DNN output coordinates are normalised to [0, 1] relative to the
        300×300 input.  This method scales them back to the original
        frame dimensions given by *frame_shape*.

        Args:
            detections: Shape (1, 1, N, 7) where each detection is
                [img_id, class_id, confidence, x1, y1, x2, y2].
            frame_shape: Original frame shape, e.g. ``(480, 640, 3)``.

        Returns:
            (is_bottle, best_confidence, box_or_None).

        """
        h, w = frame_shape[:2]
        best_confidence = 0.0
        best_box: _Box | None = None

        for i in range(detections.shape[2]):
            confidence = round(float(detections[0, 0, i, 2]), 6)
            class_id = int(detections[0, 0, i, 1])

            if class_id == _BOTTLE_CLASS_ID and confidence > best_confidence:
                best_confidence = confidence
                best_box = (
                    int(detections[0, 0, i, 3] * w),
                    int(detections[0, 0, i, 4] * h),
                    int(detections[0, 0, i, 5] * w),
                    int(detections[0, 0, i, 6] * h),
                )

        is_bottle = best_confidence >= self._threshold
        # Only return box when a bottle is actually detected
        return is_bottle, best_confidence, best_box if is_bottle else None
