"""Tests for BottleTFClassifier prediction, threshold, and enum mapping.

Uses a mock ``tf.keras.Model`` to produce controlled softmax outputs
at various confidence levels, verifying:

  - Predicted class id, name, and confidence match expected values.
  - Threshold boundary: confidence below threshold is downgraded to
    class 0 (no bottle).
  - ``BottleType`` enum values correspond to class indices.
  - Invalid threshold values are rejected.

All tests mock ``tf.keras.models.load_model`` so no real model file
or GPU is required.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.vision.classifier_tf import BottlePrediction, BottleTFClassifier, BottleType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_softmax_probs(top_class: int, top_confidence: float, num_classes: int = 3) -> np.ndarray:
    """Build a (1, num_classes) softmax-like probability vector.

    The *top_class* receives *top_confidence* probability; the remainder
    is distributed evenly among the other classes.
    """
    probs = np.ones((1, num_classes), dtype=np.float32) * ((1.0 - top_confidence) / (num_classes - 1))
    probs[0, top_class] = top_confidence
    return probs


def _make_mock_model(return_probs: np.ndarray | None = None) -> MagicMock:
    """Create a mock Keras model with a controllable ``__call__`` return."""
    model = MagicMock()
    if return_probs is not None:
        import tensorflow as tf
        model.__call__.return_value = tf.constant(return_probs)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBottleType(unittest.TestCase):
    """BottleType enum values and mapping."""

    def test_enum_values(self):
        """BottleType has correct int values matching class indices."""
        self.assertEqual(BottleType.NONE, 0)
        self.assertEqual(BottleType.POOL_VERDE, 1)
        self.assertEqual(BottleType.HATSU_MORADO, 2)

    def test_enum_from_class_id(self):
        """BottleType can be constructed from class_id."""
        self.assertEqual(BottleType(0), BottleType.NONE)
        self.assertEqual(BottleType(1), BottleType.POOL_VERDE)
        self.assertEqual(BottleType(2), BottleType.HATSU_MORADO)


class TestBottlePrediction(unittest.TestCase):
    """BottlePrediction dataclass contract."""

    def test_construction(self):
        """BottlePrediction can be constructed with valid args."""
        pred = BottlePrediction(class_id=1, confidence=0.85, class_name="Pool Verde")
        self.assertEqual(pred.class_id, 1)
        self.assertEqual(pred.confidence, 0.85)
        self.assertEqual(pred.class_name, "Pool Verde")

    def test_frozen_dataclass(self):
        """BottlePrediction is frozen (immutable)."""
        pred = BottlePrediction(class_id=0, confidence=0.0, class_name="No bottle")
        with self.assertRaises(AttributeError):
            pred.class_id = 1  # pyright: ignore


class TestBottleTFClassifier(unittest.TestCase):
    """Classifier prediction and threshold behaviour."""

    # Shared dummy frame (any size; to_tf_input handles resize internally)
    FRAME = np.zeros((480, 640, 3), dtype=np.uint8)

    def setUp(self):
        """Create a BottleTFClassifier with mocked model.

        We patch ``tf.keras.models.load_model`` to avoid needing a real
        model file, then replace ``_forward`` with a controllable mock
        so that ``predict()`` returns controlled probabilities without
        invoking TensorFlow graph tracing.
        """
        patcher = patch("tensorflow.keras.models.load_model")
        self.addCleanup(patcher.stop)
        mock_load = patcher.start()
        mock_load.return_value = MagicMock()

        self.classifier = BottleTFClassifier("fake_path.h5", threshold=0.5)

        # Replace the compiled tf.function forward pass with a plain mock
        self.classifier._forward = MagicMock()

    def _set_probs(self, top_class: int, top_confidence: float):
        """Set the mock forward pass to return a specific probability vector."""
        import tensorflow as tf
        probs = _make_softmax_probs(top_class, top_confidence)
        self.classifier._forward.return_value = tf.constant(probs)

    # --- Class mapping ---

    def test_predict_class_0_no_bottle(self):
        """Top class 0 with high confidence → class_id=0, name='No bottle'."""
        self._set_probs(top_class=0, top_confidence=0.92)
        pred = self.classifier.predict(self.FRAME)
        self.assertEqual(pred.class_id, 0)
        self.assertEqual(pred.class_name, "No bottle")
        self.assertAlmostEqual(pred.confidence, 0.92)

    def test_predict_class_1_pool_verde(self):
        """Top class 1 with high confidence → class_id=1, name='Pool Verde'."""
        self._set_probs(top_class=1, top_confidence=0.88)
        pred = self.classifier.predict(self.FRAME)
        self.assertEqual(pred.class_id, 1)
        self.assertEqual(pred.class_name, "Pool Verde")
        self.assertAlmostEqual(pred.confidence, 0.88)

    def test_predict_class_2_hatsu_morado(self):
        """Top class 2 with high confidence → class_id=2, name='Hatsu Morado'."""
        self._set_probs(top_class=2, top_confidence=0.91)
        pred = self.classifier.predict(self.FRAME)
        self.assertEqual(pred.class_id, 2)
        self.assertEqual(pred.class_name, "Hatsu Morado")
        self.assertAlmostEqual(pred.confidence, 0.91)

    def test_class_id_maps_to_bottle_type(self):
        """class_id matches BottleType value."""
        self._set_probs(top_class=1, top_confidence=0.9)
        pred = self.classifier.predict(self.FRAME)
        self.assertEqual(pred.class_id, BottleType.POOL_VERDE)

        self._set_probs(top_class=2, top_confidence=0.9)
        pred = self.classifier.predict(self.FRAME)
        self.assertEqual(pred.class_id, BottleType.HATSU_MORADO)

    # --- Threshold behaviour ---

    def test_confidence_below_threshold_downgrades_to_class_0(self):
        """confidence 0.49 (< 0.5) → class_id=0 (no bottle)."""
        self._set_probs(top_class=1, top_confidence=0.49)
        pred = self.classifier.predict(self.FRAME)
        self.assertEqual(pred.class_id, 0)
        self.assertEqual(pred.class_name, "No bottle")

    def test_confidence_at_threshold_keeps_class(self):
        """confidence 0.5 (== 0.5) → keeps predicted class."""
        self._set_probs(top_class=1, top_confidence=0.5)
        pred = self.classifier.predict(self.FRAME)
        self.assertEqual(pred.class_id, 1)
        self.assertEqual(pred.class_name, "Pool Verde")

    def test_confidence_above_threshold_keeps_class(self):
        """confidence 0.51 (> 0.5) → keeps predicted class."""
        self._set_probs(top_class=1, top_confidence=0.51)
        pred = self.classifier.predict(self.FRAME)
        self.assertEqual(pred.class_id, 1)
        self.assertEqual(pred.class_name, "Pool Verde")

    # --- Threshold at different levels ---

    def test_custom_threshold_0_7_keeps_high_confidence(self):
        """threshold=0.7, confidence=0.85 → keeps class."""
        self.classifier.threshold = 0.7
        self._set_probs(top_class=2, top_confidence=0.85)
        pred = self.classifier.predict(self.FRAME)
        self.assertEqual(pred.class_id, 2)
        self.assertAlmostEqual(pred.confidence, 0.85)

    def test_custom_threshold_0_7_rejects_moderate(self):
        """threshold=0.7, confidence=0.65 → downgrades to class 0."""
        self.classifier.threshold = 0.7
        self._set_probs(top_class=2, top_confidence=0.65)
        pred = self.classifier.predict(self.FRAME)
        self.assertEqual(pred.class_id, 0)
        self.assertEqual(pred.class_name, "No bottle")

    # --- Dual bottle scenario (dominant class wins) ---

    def test_dual_bottle_dominant_wins(self):
        """Two classes both above threshold → highest confidence wins."""
        # Class 1 at 0.75, class 2 at 0.20
        import tensorflow as tf
        probs = np.array([[0.05, 0.75, 0.20]], dtype=np.float32)
        self.classifier._forward.return_value = tf.constant(probs)
        pred = self.classifier.predict(self.FRAME)
        self.assertEqual(pred.class_id, 1)
        self.assertEqual(pred.class_name, "Pool Verde")
        self.assertAlmostEqual(pred.confidence, 0.75)

    # --- Error handling ---

    def test_invalid_threshold_raises(self):
        """threshold outside [0, 1] raises ValueError."""
        with self.assertRaises(ValueError):
            BottleTFClassifier("fake.h5", threshold=1.5)
        with self.assertRaises(ValueError):
            BottleTFClassifier("fake.h5", threshold=-0.1)


if __name__ == "__main__":
    unittest.main()
