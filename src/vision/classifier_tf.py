"""TensorFlow-based bottle classifier using MobileNetV2 transfer learning.

Three-class classifier:
    0 — No bottle (no_bottle)
    1 — Pool Verde (pool_verde)
    2 — Hatsu Morado (hatsu_morado)

Inference runs via ``@tf.function(jit_compile=True)`` for XLA-accelerated
CPU inference.  Falls back gracefully if XLA is unavailable.

Usage::

    from src.vision.classifier_tf import BottleTFClassifier

    clf = BottleTFClassifier("models/my_model.h5")
    pred = clf.predict(frame)
    print(pred.class_name, pred.confidence)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import tensorflow as tf

from src.vision.preprocess import to_tf_input

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class BottleType(IntEnum):
    """Bottle type labels matching the 3-class model output."""

    NONE = 0
    POOL_VERDE = 1
    HATSU_MORADO = 2


_CLASS_NAMES: tuple[str, str, str] = (
    "No bottle",
    "Pool Verde",
    "Hatsu Morado",
)


@dataclass(frozen=True)
class BottlePrediction:
    """Prediction result from :class:`BottleTFClassifier`.

    Attributes:
        class_id: Predicted class index (0, 1, or 2).
        confidence: Softmax probability of the predicted class (0..1).
        class_name: Human-readable label corresponding to *class_id*.
    """

    class_id: int
    confidence: float
    class_name: str


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class BottleTFClassifier:
    """MobileNetV2-based 3-class bottle classifier with XLA JIT.

    Args:
        model_path: Path to a ``.h5`` or SavedModel directory produced by
            ``training/train.py``.
        threshold: Confidence threshold in ``[0, 1]``.  Predictions whose
            top probability is below this value are downgraded to
            ``BottleType.NONE``.  Defaults to ``0.5``.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
        ValueError: If *threshold* is outside ``[0, 1]``.
    """

    def __init__(self, model_path: str, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

        self._threshold = threshold
        self._model = tf.keras.models.load_model(model_path)
        self._forward = self._build_forward()

        logger.info(
            "Loaded TF model from %s (threshold=%.2f, XLA=%s)",
            model_path,
            threshold,
            self._xla_enabled,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Current confidence threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {value}")
        self._threshold = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray) -> BottlePrediction:
        """Run inference on a single BGR frame.

        Args:
            frame: BGR image from camera (any size; will be resized to
                224×224 internally).

        Returns:
            A :class:`BottlePrediction` with the top class, its softmax
            probability, and a human-readable label.
        """
        tensor = tf.constant(to_tf_input(frame))  # (224, 224, 3) float32
        probs = self._forward(tensor[None, ...])  # (1, 3) float32
        return self._parse_prediction(probs[0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_raw(self, x: tf.Tensor) -> tf.Tensor:
        """Unwrapped forward pass: model(x, training=False)."""
        return self._model(x, training=False)

    def _build_forward(self):
        """Wrap forward pass with ``@tf.function(jit_compile=True)``.

        Falls back to a standard ``tf.function`` (or plain eager) if XLA
        is not available on this platform.
        """
        try:
            fn = tf.function(self._forward_raw, jit_compile=True)
            self._xla_enabled = True
            return fn
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "XLA JIT unavailable (%s) — falling back to non-JIT inference",
                exc,
            )
            self._xla_enabled = False
            return tf.function(self._forward_raw)

    def _parse_prediction(self, probs: tf.Tensor) -> BottlePrediction:
        """Convert softmax output to a :class:`BottlePrediction`.

        If the top confidence is below ``self._threshold``, the prediction
        is downgraded to class 0 (No bottle).
        """
        probs_np: np.ndarray = probs.numpy()  # shape (3,)
        class_id = int(np.argmax(probs_np))
        confidence = float(probs_np[class_id])

        if confidence < self._threshold:
            class_id = 0
            confidence = float(probs_np[0])

        return BottlePrediction(
            class_id=class_id,
            confidence=confidence,
            class_name=_CLASS_NAMES[class_id],
        )
