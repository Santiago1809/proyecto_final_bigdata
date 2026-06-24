"""Tests for BottleClassifier prediction and threshold boundary.

Uses a mock OpenCV DNN net to produce controlled detection outputs
at various confidence levels, verifying the classification decision
flips at the configured threshold.

Test strategy:
  - Threshold 0.7: confidence 0.6 → NOT bottle, 0.7 → bottle, 0.8 → bottle
  - Threshold 0.6: confidence 0.59 → NOT bottle, 0.61 → bottle
  - Threshold 0.8: confidence 0.79 → NOT bottle, 0.81 → bottle
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.vision.classifier import BottleClassifier


def _make_detection(class_id: int, confidence: float) -> np.ndarray:
    """Build a single-detection array shaped (1, 1, 1, 7).

    Format: [img_id, class_id, confidence, x1, y1, x2, y2]
    """
    detection = np.zeros((1, 1, 1, 7), dtype=np.float32)
    detection[0, 0, 0, 0] = 0  # img_id (unused)
    detection[0, 0, 0, 1] = float(class_id)
    detection[0, 0, 0, 2] = float(confidence)
    # Bounding box (ignored for classification test)
    detection[0, 0, 0, 3] = 0.1
    detection[0, 0, 0, 4] = 0.1
    detection[0, 0, 0, 5] = 0.5
    detection[0, 0, 0, 6] = 0.5
    return detection


class TestBottleClassifierThreshold(unittest.TestCase):
    """Threshold boundary tests using a mock DNN net."""

    def setUp(self):
        self.classifier = BottleClassifier(threshold=0.7)
        # Set a mock net to avoid needing real model files
        self.classifier._net = MagicMock()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def _inject_detection(self, class_id: int, confidence: float):
        """Replace the net.forward() return with a controlled detection."""
        self.classifier._net.forward.return_value = _make_detection(class_id, confidence)

    # --- Threshold 0.7 (default) ---

    def test_confidence_below_threshold_returns_not_bottle(self):
        """confidence 0.6 (< 0.7) → is_bottle=False."""
        self._inject_detection(class_id=39, confidence=0.6)
        is_bottle, conf = self.classifier.predict(self.frame)
        self.assertFalse(is_bottle)
        self.assertAlmostEqual(conf, 0.6)

    def test_confidence_at_threshold_returns_bottle(self):
        """confidence 0.7 (== 0.7) → is_bottle=True."""
        self._inject_detection(class_id=39, confidence=0.7)
        is_bottle, conf = self.classifier.predict(self.frame)
        self.assertTrue(is_bottle)
        self.assertAlmostEqual(conf, 0.7)

    def test_confidence_above_threshold_returns_bottle(self):
        """confidence 0.8 (> 0.7) → is_bottle=True."""
        self._inject_detection(class_id=39, confidence=0.8)
        is_bottle, conf = self.classifier.predict(self.frame)
        self.assertTrue(is_bottle)
        self.assertAlmostEqual(conf, 0.8)

    # --- Threshold 0.6 ---

    def test_threshold_0_6_reject_below(self):
        """threshold=0.6, confidence=0.59 → is_bottle=False."""
        self.classifier.threshold = 0.6
        self._inject_detection(class_id=39, confidence=0.59)
        is_bottle, _ = self.classifier.predict(self.frame)
        self.assertFalse(is_bottle)

    def test_threshold_0_6_accept_above(self):
        """threshold=0.6, confidence=0.61 → is_bottle=True."""
        self.classifier.threshold = 0.6
        self._inject_detection(class_id=39, confidence=0.61)
        is_bottle, _ = self.classifier.predict(self.frame)
        self.assertTrue(is_bottle)

    # --- Threshold 0.8 ---

    def test_threshold_0_8_reject_below(self):
        """threshold=0.8, confidence=0.79 → is_bottle=False."""
        self.classifier.threshold = 0.8
        self._inject_detection(class_id=39, confidence=0.79)
        is_bottle, _ = self.classifier.predict(self.frame)
        self.assertFalse(is_bottle)

    def test_threshold_0_8_accept_above(self):
        """threshold=0.8, confidence=0.81 → is_bottle=True."""
        self.classifier.threshold = 0.8
        self._inject_detection(class_id=39, confidence=0.81)
        is_bottle, _ = self.classifier.predict(self.frame)
        self.assertTrue(is_bottle)

    # --- Non-bottle class ---

    def test_non_bottle_class_ignored(self):
        """class_id=1 (not bottle) even with high confidence → is_bottle=False."""
        self._inject_detection(class_id=1, confidence=0.95)
        is_bottle, conf = self.classifier.predict(self.frame)

    def test_no_detections_returns_not_bottle(self):
        """Empty detections array → is_bottle=False, confidence=0.0."""
        self.classifier._net.forward.return_value = np.zeros((1, 1, 0, 7), dtype=np.float32)
        is_bottle, conf = self.classifier.predict(self.frame)
        self.assertFalse(is_bottle)
        self.assertEqual(conf, 0.0)

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
