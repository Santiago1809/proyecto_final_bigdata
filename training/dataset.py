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
    """Scan *data_dir* for subdirectories and return class info.

    The class order follows :attr:`Config.CLASS_NAMES` to stay aligned
    with :class:`src.vision.classifier_tf.BottleType` (NONE=0, POOL_VERDE=1,
    HATSU_MORADO=2).
    """
    data_path = Path(data_dir)
    available = {d.name for d in data_path.iterdir() if d.is_dir()}

    # Use Config.CLASS_NAMES order so class indices match BottleType
    expected = Config().CLASS_NAMES
    class_names = [name for name in expected if name in available]

    if not class_names:
        raise ValueError(
            f"No matching class subdirectories found in {data_dir!r}. "
            f"Expected at least one of: {Config.CLASS_NAMES}"
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
    """Read, decode (JPEG / PNG), and resize.

    Returns raw float32 pixels in [0, 255] — MobileNetV2 has its own
    normalisation built in (``true_divide / 128 → subtract 1``).
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    return tf.cast(image, tf.float32)


def _color_jitter(image: tf.Tensor, label: int) -> tuple[tf.Tensor, int]:
    """Randomly adjust hue and saturation to reduce colour dependency.

    Operates on the raw ``[0, 255]`` range — normalises to ``[0, 1]``,
    applies the jitter, and scales back.  Only applied during training.
    """
    norm = image / 255.0
    hue = tf.image.random_hue(norm, max_delta=0.15)
    sat = tf.image.random_saturation(hue, lower=0.5, upper=1.5)
    return sat * 255.0, label


def _build_augmentation() -> tf.keras.Sequential:
    """On-device augmentation pipeline applied only to the training set.

    Includes geometric transforms (flip, rotation, zoom, shear,
    translation) plus colour perturbation (brightness, contrast, hue,
    saturation) to make the model invariant to lighting conditions and
    bottle tint variations.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomShear(0.15),
        tf.keras.layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomGrayscale(factor=0.2),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(
    data_dir: str,
    config: Config = Config(),
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], list[int]]:
    """Load images from class subdirectories with a stratified train/val split.

    Args:
        data_dir: Path to the dataset root containing class subdirectories.
        config: Configuration object (or defaults).

    Returns:
        ``(train_ds, val_ds, class_names, class_counts)`` where both
        datasets yield ``(image, label)`` tuples.  ``class_names``
        preserves the index order for mapping predictions back to labels.
        ``class_counts`` has the number of images per class.

    Raises:
        ValueError: If *data_dir* has no subdirectories or no images.
    """
    class_names, class_to_idx = _discover_classes(data_dir)
    file_paths, labels = _collect_files(data_dir, class_names, class_to_idx)
    class_counts = [labels.count(i) for i in range(len(class_names))]

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

        # Cache decoded/resized images in RAM (training only — multiple epochs)
        if augment:
            ds = ds.cache()

        # Augment only the training set
        if augment:
            aug_pipeline = _build_augmentation()

            def _augment(image: tf.Tensor, label: int) -> tuple[tf.Tensor, int]:
                augmented = aug_pipeline(image, training=True)
                # Brightness/contrast can push values outside [0, 255]; clip back
                augmented = tf.clip_by_value(augmented, 0.0, 255.0)
                return augmented, label

            ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
            ds = ds.map(_color_jitter, num_parallel_calls=AUTOTUNE)

        return ds.batch(batch_size).prefetch(AUTOTUNE)

    train_ds = _make_tf_ds(train_paths, train_labels, augment=True)
    val_ds = _make_tf_ds(val_paths, val_labels, augment=False)

    return train_ds, val_ds, class_names, class_counts
