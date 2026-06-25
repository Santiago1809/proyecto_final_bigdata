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

_LOG = logging.getLogger("extract-centroids")
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _discover_classes(data_dir: str) -> list[tuple[str, list[str]]]:
    """Walk *data_dir* and return ``[(class_name, [image_paths])]``."""
    data_path = Path(data_dir)
    classes: list[tuple[str, list[str]]] = []
    for subdir in sorted(data_path.iterdir()):
        if not subdir.is_dir():
            continue
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images.extend(str(p) for p in sorted(subdir.glob(ext)))
        if images:
            classes.append((subdir.name, images))
    return classes


def _load_and_preprocess(path: str) -> np.ndarray:
    """Read an image from disk and preprocess it like the classifier does."""
    import cv2

    frame = cv2.imread(path)
    if frame is None:
        raise ValueError(f"Failed to read {path}")
    return to_tf_input(frame)  # (224, 224, 3) float32 [0, 255]


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
) -> dict:
    """Extract features and compute centroids for each class.

    Args:
        model_path: Path to the trained ``.keras`` model file.
        data_dir: Root directory with class subdirectories.
        sigma: Number of standard deviations for the rejection threshold.

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
    for cls_name, imgs in classes:
        _LOG.info("  %s: %d images", cls_name, len(imgs))

    class_to_idx = {name: i for i, (name, _) in enumerate(classes)}

    # ---- Extract features ----
    all_features: dict[int, list[np.ndarray]] = {i: [] for i in range(len(classes))}

    for cls_name, image_paths in classes:
        cls_idx = class_to_idx[cls_name]
        _LOG.info("Extracting features for %s (%d images) ...", cls_name, len(image_paths))
        for path in image_paths:
            try:
                tensor = _load_and_preprocess(path)
                feat = feature_model(tensor[None, ...], training=False).numpy()[0]
                all_features[cls_idx].append(feat)
            except Exception as exc:
                _LOG.warning("Skipping %s: %s", path, exc)

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

        cls_name = classes[cls_idx][0]
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
