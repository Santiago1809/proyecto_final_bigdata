"""TensorFlow-based bottle classifier using MobileNetV2 transfer learning.

Three-class classifier with optional feature-space rejection to reject
out-of-distribution inputs (e.g. unseen bottle brands).

    0 — No bottle (no_bottle)
    1 — Pool Verde (pool_verde)
    2 — Hatsu Morado (hatsu_morado)

Feature-space rejection compares the 128-dimensional feature vector from the
penultimate layer against per-class centroids.  If the Euclidean distance
exceeds the class threshold, the prediction is downgraded to class 0.

Usage::

    from src.vision.classifier_tf import BottleTFClassifier

    clf = BottleTFClassifier("models/my_model.keras")
    pred = clf.predict(frame)
    print(pred.class_name, pred.confidence)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

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

# Default centroids path: same stem as the model + ``_centroids.json``
_CENTROIDS_SUFFIX = "_centroids.json"


@dataclass(frozen=True)
class BottlePrediction:
    """Prediction result from :class:`BottleTFClassifier`.

    Attributes:
        class_id: Predicted class index (0, 1, or 2).
        confidence: Softmax probability of the predicted class (0..1).
        class_name: Human-readable label corresponding to *class_id*.
        rejected: Whether the prediction was rejected by feature-space
            distance check.  ``True`` means the input looked like the
            predicted class but was too far from its centroid.
        feature_distance: Euclidean distance from the predicted class
            centroid (0 if rejection is disabled).
    """

    class_id: int
    confidence: float
    class_name: str
    rejected: bool = False
    feature_distance: float = 0.0


# ---------------------------------------------------------------------------
# Feature-space rejection helpers
# ---------------------------------------------------------------------------


def _auto_detect_centroids_path(model_path: str) -> str | None:
    """Derive the centroids JSON path from the model path.

    ``models/bottle_classifier_latest.keras`` →
    ``models/bottle_classifier_latest_centroids.json``

    Returns ``None`` if the file does not exist.
    """
    p = Path(model_path)
    candidate = p.parent / f"{p.stem}{_CENTROIDS_SUFFIX}"
    return str(candidate) if candidate.exists() else None


def _load_centroids(path: str) -> dict:
    """Load per-class centroids from a JSON file.

    Returns a dict mapping class index ``str`` → centroid info with keys:
    ``mean``, ``threshold``, ``class_name``.
    """
    with open(path) as f:
        return json.load(f)


def _find_feature_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Find the penultimate Dense layer (feature layer before softmax).

    Tries ``dense_1`` first (standard name from ``model.py``), then
    auto-detects by finding the second-to-last Dense layer among top-level
    layers.
    """
    # Fast path: standard name
    try:
        return model.get_layer("dense_1")
    except ValueError:
        pass

    # Fallback: find Dense layers at the top level, exclude the last one
    dense_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]
    if len(dense_layers) >= 2:
        logger.info("Auto-detected feature layer: %s", dense_layers[-2].name)
        return dense_layers[-2]

    raise ValueError(
        "Could not find feature layer in the model. "
        "Expected a Dense layer before the final softmax. "
        f"Found Dense layers: {[l.name for l in dense_layers]}"
    )


def _build_combined_model(
    model: tf.keras.Model,
) -> tf.keras.Model:
    """Build a combined model that outputs both (features, softmax).

    The feature output comes from the penultimate Dense layer (128-d
    embedding).  The softmax output is the original classification.

    This allows a single forward pass to compute both values.
    """
    feature_layer = _find_feature_layer(model)
    combined = tf.keras.Model(
        inputs=model.inputs,
        outputs=[feature_layer.output, model.output],
    )
    return combined


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class BottleTFClassifier:
    """MobileNetV2-based 3-class bottle classifier with feature-space rejection.

    Args:
        model_path: Path to a ``.keras`` model produced by ``training/train.py``.
        threshold: Confidence threshold in ``[0, 1]``.  Predictions whose
            top softmax probability is below this value are downgraded to
            ``BottleType.NONE``.  Defaults to ``0.5``.
        centroids_path: Path to a centroids JSON file.  If ``None``,
            auto-detects ``<model_stem>_centroids.json`` alongside the model
            file.  Set to ``""`` (empty string) to disable feature-space
            rejection entirely.
        rejection_sigma: Number of standard deviations for the rejection
            threshold when centroids are loaded.  Only used for display;
            the actual threshold is computed by ``extract_centroids.py``.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
        ValueError: If *threshold* is outside ``[0, 1]``.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        centroids_path: str | None = None,
        rejection_sigma: float = 3.0,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

        self._threshold = threshold
        self._rejection_sigma = rejection_sigma
        self._model = self._load_model_safe(model_path)
        self._forward = self._build_forward()

        # ---- Feature-space rejection setup ----
        centroids_path = centroids_path if centroids_path is not None else _auto_detect_centroids_path(model_path)
        self._centroids: dict | None = None
        self._feature_forward = None
        self._rejection_active = False

        if centroids_path:
            try:
                self._centroids = _load_centroids(centroids_path)
                # Build combined model (features + softmax) in one forward pass
                combined = _build_combined_model(self._model)
                self._feature_forward = self._build_forward(combined)
                self._rejection_active = True
                logger.info(
                    "Feature-space rejection ACTIVE (%d classes, centroids: %s)",
                    len(self._centroids),
                    centroids_path,
                )
                # Log thresholds for reference
                for cid, info in sorted(self._centroids.items(), key=lambda x: int(x[0])):
                    logger.info(
                        "  centroid[%s] %s: threshold=%.4f, mean_dist=%.4f, samples=%d",
                        cid, info.get("class_name", "?"),
                        info["threshold"], info["mean_dist"], info.get("num_samples", 0),
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to load centroids from %s: %s — rejection disabled",
                    centroids_path,
                    exc,
                )

        logger.info(
            "Loaded TF model from %s (threshold=%.2f, XLA=%s, rejection=%s)",
            model_path,
            threshold,
            self._xla_enabled,
            "active" if self._rejection_active else "disabled",
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model_safe(model_path: str) -> tf.keras.Model:
        """Load a Keras model from a ``.keras`` file."""
        return tf.keras.models.load_model(model_path)

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

    @property
    def rejection_active(self) -> bool:
        """Whether feature-space rejection is currently active."""
        return self._rejection_active

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
            probability, and a human-readable label.  If feature-space
            rejection is active and the input is too far from the
            predicted class centroid, the result is downgraded to
            ``NONE``.
        """
        tensor = tf.constant(to_tf_input(frame))  # (224, 224, 3) float32

        if self._rejection_active:
            features, probs = self._feature_forward(tensor[None, ...])
            return self._parse_prediction(probs[0], features=features[0])
        else:
            probs = self._forward(tensor[None, ...])
            return self._parse_prediction(probs[0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_raw(self, x: tf.Tensor) -> tf.Tensor:
        """Unwrapped forward pass: model(x, training=False)."""
        return self._model(x, training=False)

    def _build_forward(self, model: tf.keras.Model | None = None):
        """Wrap forward pass with ``@tf.function(jit_compile=True)``.

        Falls back to a standard ``tf.function`` (or plain eager) if XLA
        is not available on this platform.

        Args:
            model: Optional model to wrap.  Defaults to ``self._model``.
        """
        target = model or self._model

        def _fn(x: tf.Tensor):
            return target(x, training=False)

        try:
            fn = tf.function(_fn, jit_compile=True)
            self._xla_enabled = True
            return fn
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "XLA JIT unavailable (%s) — falling back to non-JIT inference",
                exc,
            )
            self._xla_enabled = False
            return tf.function(_fn)

    def _reject_by_features(
        self,
        class_id: int,
        features: np.ndarray,
    ) -> tuple[bool, float, float]:
        """Check whether *features* are too far from the *class_id* centroid.

        Uses the runtime *rejection_sigma* to compute the threshold as
        ``mean_dist + sigma * std_dist``, so the user can adjust
        permissiveness without regenerating centroids.

        Returns:
            ``(rejected, distance, threshold)`` tuple where *rejected* is
            ``True`` if the Euclidean distance exceeds the threshold.
        """
        if not self._centroids:
            return False, 0.0, 0.0

        centroid_info = self._centroids.get(str(class_id))
        if centroid_info is None:
            return False, 0.0, 0.0

        centroid = np.array(centroid_info["mean"], dtype=np.float32)
        mean_dist = centroid_info["mean_dist"]
        std_dist = centroid_info["std_dist"]

        # Compute threshold from runtime sigma (not baked-in value)
        threshold = mean_dist + self._rejection_sigma * std_dist
        distance = float(np.linalg.norm(features - centroid))

        return distance > threshold, distance, threshold

    def _parse_prediction(
        self,
        probs: tf.Tensor,
        features: tf.Tensor | None = None,
    ) -> BottlePrediction:
        """Convert softmax output to a :class:`BottlePrediction`.

        If *features* are provided and feature-space rejection is active,
        the prediction is downgraded to class 0 (No bottle) when the
        feature vector is too far from the predicted class centroid.
        """
        probs_np: np.ndarray = probs.numpy()  # shape (3,)
        class_id = int(np.argmax(probs_np))
        confidence = float(probs_np[class_id])
        rejected = False
        feature_distance = 0.0

        # Feature-space rejection
        if features is not None and self._rejection_active:
            should_reject, feature_distance, threshold = self._reject_by_features(
                class_id, features.numpy(),
            )
            if should_reject:
                logger.info(
                    "REJECTED class %d (%s) — "
                    "dist=%.4f > threshold=%.4f (sigma=%.1f)",
                    class_id, _CLASS_NAMES[class_id],
                    feature_distance, threshold, self._rejection_sigma,
                )
                rejected = True
                # Keep the softmax confidence even after rejection — the
                # model IS confident, we're just overriding it because the
                # input is OOD.  Display can show "Pool Verde (95%) [OOD]".
                class_id = 0

        # Confidence threshold
        if confidence < self._threshold:
            class_id = 0
            # Only lower confidence if we didn't already reject
            if not rejected:
                confidence = float(probs_np[0])

        return BottlePrediction(
            class_id=class_id,
            confidence=confidence,
            class_name=_CLASS_NAMES[class_id],
            rejected=rejected,
            feature_distance=feature_distance,
        )
