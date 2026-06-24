"""Dataset loading, stratified splitting, and augmentation pipeline.

Expects a directory structure where each subdirectory name is a class::

    data/
        no_bottle/      ← class index 0
        pool_verde/     ← class index 1
        hatsu_morado/   ← class index 2

Images are resized to ``(IMG_SIZE, IMG_SIZE)`` and normalised to ``[0, 1]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from training.config import Config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _discover_classes(data_dir: str) -> tuple[list[str], dict[str, int]]:
    """Scan *data_dir* for subdirectories and return sorted class info."""
    data_path = Path(data_dir)
    class_names = sorted(d.name for d in data_path.iterdir() if d.is_dir())
    if not class_names:
        raise ValueError(
            f"No subdirectories found in {data_dir!r}. "
            "Expected one folder per class."
        )
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    return class_names, class_to_idx


def _collect_files(
    data_dir: str,
    class_names: list[str],
    class_to_idx: dict[str, int],
) -> tuple[list[str], list[int]]:
    """Walk class subdirectories and collect image file paths + labels."""
    data_path = Path(data_dir)
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    file_paths: list[str] = []
    labels: list[int] = []

    for class_name in class_names:
        class_dir = data_path / class_name
        for ext in extensions:
            for f in sorted(class_dir.glob(ext)):
                file_paths.append(str(f))
                labels.append(class_to_idx[class_name])

    if not file_paths:
        raise ValueError(
            f"No images found under {data_dir!r}. "
            f"Supported extensions: {', '.join(ext.replace('*', '') for ext in extensions)}"
        )
    return file_paths, labels


def _decode_resize(image_path: str, img_size: int) -> tf.Tensor:
    """Read, decode (JPEG / PNG), and resize a single image to [0, 1]."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    return tf.cast(image, tf.float32) / 255.0


def _build_augmentation() -> tf.keras.Sequential:
    """On-device augmentation pipeline applied only to the training set."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(
    data_dir: str,
    config: Config = Config(),
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load images from class subdirectories with a stratified train/val split.

    Args:
        data_dir: Path to the dataset root containing class subdirectories.
        config: Configuration object (or defaults).

    Returns:
        ``(train_ds, val_ds, class_names)`` where both datasets yield
        ``(image, label)`` tuples.  ``class_names`` preserves the index
        order for mapping predictions back to labels.

    Raises:
        ValueError: If *data_dir* has no subdirectories or no images.
    """
    class_names, class_to_idx = _discover_classes(data_dir)
    file_paths, labels = _collect_files(data_dir, class_names, class_to_idx)

    # Stratified train / validation split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths,
        labels,
        test_size=config.VALIDATION_SPLIT,
        stratify=labels,
        random_state=config.RANDOM_SEED,
    )

    # Free variables for dataset map closures
    img_size = config.IMG_SIZE
    batch_size = config.BATCH_SIZE
    seed = config.RANDOM_SEED
    AUTOTUNE = tf.data.AUTOTUNE

    def _make_tf_ds(
        paths: list[str],
        lbls: list[int],
        *,
        augment: bool,
    ) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((paths, lbls))

        if augment:
            ds = ds.shuffle(len(paths), seed=seed)

        # Decode and resize
        def _decode(path: str, label: int) -> tuple[tf.Tensor, tf.Tensor]:
            return _decode_resize(path, img_size), label

        ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)

        # Augment only the training set
        if augment:
            aug_pipeline = _build_augmentation()

            def _augment(image: tf.Tensor, label: int) -> tuple[tf.Tensor, int]:
                augmented = aug_pipeline(image, training=True)
                # RandomBrightness can push values outside [0, 1]; clip back
                augmented = tf.clip_by_value(augmented, 0.0, 1.0)
                return augmented, label

            ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

        return ds.batch(batch_size).prefetch(AUTOTUNE)

    train_ds = _make_tf_ds(train_paths, train_labels, augment=True)
    val_ds = _make_tf_ds(val_paths, val_labels, augment=False)

    return train_ds, val_ds, class_names
