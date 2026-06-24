"""MobileNetV2-based classifier builder with custom head and two-phase training.

Provides:

- ``build_mobilenetv2()`` — constructs the full model with a frozen backbone
- ``unfreeze_top_layers()`` — enables fine-tuning of the last *N* backbone layers
"""

from __future__ import annotations

import tensorflow as tf

BACKBONE_LAYER_NAME = "mobilenetv2_1.00_224"


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_mobilenetv2(
    img_size: int = 224,
    num_classes: int = 3,
    dropout_rate: float = 0.2,
    dense_units: int = 128,
    backbone_trainable: bool = False,
) -> tf.keras.Model:
    """Build a MobileNetV2 backbone with a custom classification head.

    The backbone is loaded with ImageNet pretrained weights and frozen
    by default.  Use :func:`unfreeze_top_layers` for the fine-tuning phase.

    Args:
        img_size: Input image size (square).
        num_classes: Number of output classes (softmax units).
        dropout_rate: Dropout rate before the final dense layer.
        dense_units: Units in the penultimate ``ReLU`` dense layer.
        backbone_trainable: Whether the entire backbone is trainable
            from the start (default ``False``).

    Returns:
        An uncompiled Keras model ready for ``.compile()``.
    """
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = backbone_trainable

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    # Normalise from [0, 1] (as produced by dataset.py) to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Backbone runs in inference mode so that BatchNorm stats are not
    # updated during the frozen phase.
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# ---------------------------------------------------------------------------
# Fine-tuning helper
# ---------------------------------------------------------------------------


def unfreeze_top_layers(model: tf.keras.Model, num_layers: int = 30) -> None:
    """Unfreeze the last *num_layers* of the MobileNetV2 backbone.

    Early backbone layers learn generic features (edges, textures) and
    are best left frozen.  This helper keeps the early layers frozen and
    makes only the *top* (closest to the head) layers trainable.

    Args:
        model: A model built with :func:`build_mobilenetv2`.
        num_layers: Number of layers from the end of the backbone to
            unfreeze.  Defaults to 30, which is roughly the top 1/3 of
            MobileNetV2.

    Raises:
        ValueError: If the model does not contain a MobileNetV2 backbone
            with the expected layer name.
    """
    try:
        backbone = model.get_layer(BACKBONE_LAYER_NAME)
    except ValueError as exc:
        raise ValueError(
            f"Model does not contain a '{BACKBONE_LAYER_NAME}' layer. "
            "Was this model built with build_mobilenetv2()?"
        ) from exc

    # The backbone must be trainable as a whole first, then individual
    # early layers are re-frozen.
    backbone.trainable = True

    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False

    for layer in backbone.layers[-num_layers:]:
        layer.trainable = True
