"""Tests for BottleClassifier prediction, bounding box, and threshold.

Uses a mock OpenCV DNN net to produce controlled detection outputs
at various confidence levels, verifying the classification decision
flips at the configured threshold and bounding box coordinates are
returned correctly.

Test strategy:
  - Threshold 0.7: confidence 0.6 → NOT bottle, 0.7 → bottle, 0.8 → bottle
  - Threshold 0.6: confidence 0.59 → NOT bottle, 0.61 → bottle
  - Threshold 0.8: confidence 0.79 → NOT bottle, 0.81 → bottle
  - Bounding box is returned as a 4-tuple when detected, None otherwise
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.vision.classifier import BottleClassifier


def _make_detection(
    class_id: int, confidence: float,
    x1: float = 0.1, y1: float = 0.1, x2: float = 0.5, y2: float = 0.5,
) -> np.ndarray:
    """Build a single-detection array shaped (1, 1, 1, 7).

    Format: [img_id, class_id, confidence, x1, y1, x2, y2]
    Coordinates are normalised [0, 1] relative to the 300×300 DNN input.
    """
    detection = np.zeros((1, 1, 1, 7), dtype=np.float32)
    detection[0, 0, 0, 0] = 0          # img_id (unused)
    detection[0, 0, 0, 1] = float(class_id)
    detection[0, 0, 0, 2] = float(confidence)
    detection[0, 0, 0, 3] = float(x1)
    detection[0, 0, 0, 4] = float(y1)
    detection[0, 0, 0, 5] = float(x2)
    detection[0, 0, 0, 6] = float(y2)
    return detection


class TestBottleClassifierThreshold(unittest.TestCase):
    """Threshold boundary tests using a mock DNN net."""

    FRAME_H = 480
    FRAME_W = 640

    def setUp(self):
        self.classifier = BottleClassifier(threshold=0.7)
        self.classifier._net = MagicMock()
        self.frame = np.zeros((self.FRAME_H, self.FRAME_W, 3), dtype=np.uint8)

    def _inject_detection(self, class_id: int, confidence: float, **kw):
        """Replace net.forward() with a controlled detection."""
        self.classifier._net.forward.return_value = _make_detection(
            class_id, confidence, **kw
        )

    # --- Threshold 0.7 (default) ---

    def test_confidence_below_threshold_returns_not_bottle(self):
        """confidence 0.6 (< 0.7) → is_bottle=False, box=None."""
        self._inject_detection(class_id=5, confidence=0.6)
        is_bottle, conf, box = self.classifier.predict(self.frame)
        self.assertFalse(is_bottle)
        self.assertAlmostEqual(conf, 0.6)
        self.assertIsNone(box)

    def test_confidence_at_threshold_returns_bottle(self):
        """confidence 0.7 (== 0.7) → is_bottle=True, box is tuple."""
        self._inject_detection(class_id=5, confidence=0.7)
        is_bottle, conf, box = self.classifier.predict(self.frame)
        self.assertTrue(is_bottle)
        self.assertAlmostEqual(conf, 0.7)
        self.assertIsInstance(box, tuple)
        self.assertEqual(len(box), 4)

    def test_confidence_above_threshold_returns_bottle(self):
        """confidence 0.8 (> 0.7) → is_bottle=True."""
        self._inject_detection(class_id=5, confidence=0.8)
        is_bottle, conf, box = self.classifier.predict(self.frame)
        self.assertTrue(is_bottle)
        self.assertAlmostEqual(conf, 0.8)
        self.assertIsNotNone(box)

    # --- Threshold 0.6 ---

    def test_threshold_0_6_reject_below(self):
        """threshold=0.6, confidence=0.59 → is_bottle=False."""
        self.classifier.threshold = 0.6
        self._inject_detection(class_id=5, confidence=0.59)
        is_bottle, _, box = self.classifier.predict(self.frame)
        self.assertFalse(is_bottle)
        self.assertIsNone(box)

    def test_threshold_0_6_accept_above(self):
        """threshold=0.6, confidence=0.61 → is_bottle=True."""
        self.classifier.threshold = 0.6
        self._inject_detection(class_id=5, confidence=0.61)
        is_bottle, _, box = self.classifier.predict(self.frame)
        self.assertTrue(is_bottle)
        self.assertIsNotNone(box)

    # --- Threshold 0.8 ---

    def test_threshold_0_8_reject_below(self):
        """threshold=0.8, confidence=0.79 → is_bottle=False."""
        self.classifier.threshold = 0.8
        self._inject_detection(class_id=5, confidence=0.79)
        is_bottle, _, box = self.classifier.predict(self.frame)
        self.assertFalse(is_bottle)
        self.assertIsNone(box)

    def test_threshold_0_8_accept_above(self):
        """threshold=0.8, confidence=0.81 → is_bottle=True."""
        self.classifier.threshold = 0.8
        self._inject_detection(class_id=5, confidence=0.81)
        is_bottle, _, box = self.classifier.predict(self.frame)
        self.assertTrue(is_bottle)
        self.assertIsNotNone(box)

    # --- Non-bottle class ---

    def test_non_bottle_class_ignored(self):
        """class_id=1 (not bottle) high confidence → is_bottle=False, box=None."""
        self._inject_detection(class_id=1, confidence=0.95)
        is_bottle, conf, box = self.classifier.predict(self.frame)
        self.assertFalse(is_bottle)
        self.assertEqual(conf, 0.0)  # only bottle-class confidence is tracked
        self.assertIsNone(box)

    def test_no_detections_returns_not_bottle(self):
        """Empty detections array → is_bottle=False, confidence=0.0, box=None."""
        self.classifier._net.forward.return_value = np.zeros((1, 1, 0, 7), dtype=np.float32)
        is_bottle, conf, box = self.classifier.predict(self.frame)
        self.assertFalse(is_bottle)
        self.assertEqual(conf, 0.0)
        self.assertIsNone(box)

    # --- Bounding box coordinate mapping ---

    def test_box_coordinates_mapped_to_frame(self):
        """Normalised coords from DNN are mapped to original frame pixels."""
        # DNN coords (0.25, 0.25) → (0.75, 0.75) in 300×300 input
        # → frame pixels on 640×480: (160, 120) → (480, 360)
        self._inject_detection(
            class_id=5, confidence=0.9,
            x1=0.25, y1=0.25, x2=0.75, y2=0.75,
        )
        _, _, box = self.classifier.predict(self.frame)
        self.assertEqual(box, (160, 120, 480, 360))

    def test_box_none_when_no_bottle(self):
        """No detection → box is None."""
        self._inject_detection(class_id=1, confidence=0.9)
        _, _, box = self.classifier.predict(self.frame)
        self.assertIsNone(box)

    # --- Error handling ---

    def test_predict_without_model_raises(self):
        """Calling predict() without load_model() raises RuntimeError."""
        unloaded = BottleClassifier()
        with self.assertRaises(RuntimeError):
            unloaded.predict(self.frame)

    def test_invalid_threshold_raises(self):
        """threshold outside [0, 1] raises ValueError."""
        with self.assertRaises(ValueError):
            BottleClassifier(threshold=1.5)
        with self.assertRaises(ValueError):
            BottleClassifier(threshold=-0.1)


if __name__ == "__main__":
    unittest.main()
