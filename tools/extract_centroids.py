#!/usr/bin/env python3
"""Extract feature centroids from a trained model for feature-space rejection.

Usage::

    python -m tools.extract_centroids                        \\
        --model models/bottle_classifier_latest.keras         \\
        --data-dir training/data                              \\
        --sigma 3.0

Produces ``models/bottle_classifier_latest_centroids.json`` with per-class:
- ``mean``: centroid (128-d feature vector)
- ``mean_dist``: mean Euclidean distance from centroid
- ``threshold``: ``mean_dist + sigma * std_dist`` — prediction is rejected
  as unknown when its feature vector exceeds this distance from the centroid.

The centroids file is loaded by :class:`src.vision.classifier_tf.BottleTFClassifier`
to reject out-of-distribution inputs (e.g. unseen bottle brands).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.vision.preprocess import to_tf_input
from training.config import Config

_LOG = logging.getLogger("extract-centroids")
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_IMG_SIZE = 224
_BATCH_SIZE = 32  # default batch for feature extraction

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _discover_classes(data_dir: str) -> list[tuple[str, list[str], int]]:
    """Walk *data_dir* and return ``[(class_name, [image_paths], class_idx)]``.

    Class indices follow :class:`training.config.Config.CLASS_NAMES` order
    (``no_bottle=0, pool_verde=1, hatsu_morado=2``) to stay aligned with
    the trained model's output.
    """
    data_path = Path(data_dir)
    available = {d.name for d in data_path.iterdir() if d.is_dir()}
    class_names = Config().CLASS_NAMES

    classes: list[tuple[str, list[str], int]] = []
    for idx, cls_name in enumerate(class_names):
        if cls_name not in available:
            continue
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images.extend(str(p) for p in sorted((data_path / cls_name).glob(ext)))
        if images:
            classes.append((cls_name, images, idx))

    if not classes:
        raise ValueError(
            f"No matching class subdirectories found in {data_dir!r}. "
            f"Expected at least one of: {class_names}"
        )
    return classes


def _build_feature_dataset(
    image_paths: list[str],
    batch_size: int,
) -> tf.data.Dataset:
    """Build a batched ``tf.data.Dataset`` that yields preprocessed images.

    The pipeline:
    file path → read file → decode JPEG/PNG → resize → BGR→RGB → batch
    """
    def _decode_and_preprocess(path: str) -> tf.Tensor:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [_IMG_SIZE, _IMG_SIZE])
        # Match the classifier's preprocessing: resize + BGR→RGB
        # but to_tf_input expects BGR. Since decode_image gives RGB,
        # we keep RGB as-is for the TF pipeline.
        return tf.cast(image, tf.float32)

    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Centroids computation
# ---------------------------------------------------------------------------


def _find_feature_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Find the penultimate Dense layer (the feature layer before softmax).

    Looks for the last :class:`tf.keras.layers.Dense` layer whose name
    is **not** ``dense_2`` (the softmax output).  Falls back to
    ``dense_1`` by name.
    """
    # First, try the standard name
    try:
        return model.get_layer("dense_1")
    except ValueError:
        pass

    # Auto-detect: find Dense layers, exclude the last one
    dense_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]
    if len(dense_layers) >= 2:
        return dense_layers[-2]  # second-to-last Dense

    raise ValueError(
        "Could not find feature layer. Expected a Dense layer before the "
        "final softmax layer. Available Dense layers: "
        + ", ".join(l.name for l in dense_layers)
    )


def extract_centroids(
    model_path: str,
    data_dir: str,
    sigma: float = 3.0,
    batch_size: int = _BATCH_SIZE,
) -> dict:
    """Extract features and compute centroids for each class.

    Args:
        model_path: Path to the trained ``.keras`` model file.
        data_dir: Root directory with class subdirectories.
        sigma: Number of standard deviations for the rejection threshold.
        batch_size: Batch size for TF feature extraction (default 32).

    Returns:
        A dict mapping class index ``str`` → centroids info::

            {"0": {"mean": [...], "mean_dist": float, "threshold": float,
                    "class_name": str, "num_samples": int},
             "1": {...}}
    """
    # ---- Load model ----
    _LOG.info("Loading model from %s ...", model_path)
    model: tf.keras.Model = tf.keras.models.load_model(model_path)

    # Build feature extractor
    feature_layer = _find_feature_layer(model)
    _LOG.info("Feature layer: %s (output shape: %s)", feature_layer.name, feature_layer.output.shape)
    feature_model = tf.keras.Model(inputs=model.input, outputs=feature_layer.output)

    # ---- Load data ----
    classes = _discover_classes(data_dir)
    _LOG.info("Found %d classes:", len(classes))
    for cls_name, imgs, cls_idx in classes:
        _LOG.info("  [%d] %s: %d images", cls_idx, cls_name, len(imgs))

    # ---- Extract features (batched) ----
    all_features: dict[int, list[np.ndarray]] = {}

    for cls_name, image_paths, cls_idx in classes:
        all_features[cls_idx] = []
        _LOG.info("Extracting features for [%d] %s (%d images, batch=%d) ...",
                  cls_idx, cls_name, len(image_paths), batch_size)
        ds = _build_feature_dataset(image_paths, batch_size)
        for batch in ds:
            # The model expects RGB input with preprocess_input normalization
            # (true_divide / 128 → subtract 1). The model has this built in.
            feats = feature_model(batch, training=False).numpy()  # (B, 128)
            for feat in feats:
                all_features[cls_idx].append(feat)

    # ---- Compute centroids ----
    result: dict[str, dict] = {}
    for cls_idx, feats in all_features.items():
        if not feats:
            _LOG.warning("No features for class index %d, skipping", cls_idx)
            continue
        feats_arr = np.array(feats)  # (N, 128)
        centroid = feats_arr.mean(axis=0)  # (128,)

        # Distances from centroid
        dists = np.linalg.norm(feats_arr - centroid[None, :], axis=1)
        mean_dist = float(dists.mean())
        std_dist = float(dists.std())
        threshold = mean_dist + sigma * std_dist

        # Find the class name for this index
        matching = [c for c in classes if c[2] == cls_idx]
        cls_name = matching[0][0] if matching else f"class_{cls_idx}"
        result[str(cls_idx)] = {
            "mean": centroid.tolist(),
            "mean_dist": round(mean_dist, 4),
            "std_dist": round(std_dist, 4),
            "threshold": round(threshold, 4),
            "class_name": cls_name,
            "num_samples": len(feats),
        }
        _LOG.info(
            "  %s (idx=%d): mean_dist=%.3f, threshold=%.3f, samples=%d",
            cls_name, cls_idx, mean_dist, threshold, len(feats),
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract feature centroids for feature-space rejection",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained .keras model",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the training data root with class subdirectories",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Number of standard deviations for rejection threshold (default: 3.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_BATCH_SIZE,
        help=f"Batch size for feature extraction (default: {_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <model_dir>/<model_stem>_centroids.json)",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)

    args = _build_parser().parse_args()

    centroids = extract_centroids(
        model_path=args.model,
        data_dir=args.data_dir,
        sigma=args.sigma,
        batch_size=args.batch_size,
    )

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        model_path = Path(args.model)
        out_path = model_path.parent / f"{model_path.stem}_centroids.json"

    with open(out_path, "w") as f:
        json.dump(centroids, f, indent=2)

    _LOG.info("Centroids saved to %s", out_path)
    _LOG.info("Done. %d classes processed.", len(centroids))


if __name__ == "__main__":
    main()
