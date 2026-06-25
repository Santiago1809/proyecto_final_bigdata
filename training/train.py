#!/usr/bin/env python3
"""Two-phase training entry point for the bottle classifier.

Usage::

    python training/train.py --data-dir ./data

Optional arguments::

    python training/train.py --data-dir ./data            \\
        --epochs-frozen 15 --epochs-finetune 10           \\
        --batch-size 16 --lr-frozen 1e-3 --lr-finetune 1e-5 \\
        --model-dir ./models
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

# Ensure the project root is on sys.path so that "from training.xxx" works
# when running as ``python training/train.py``.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tensorflow as tf

from training.config import Config
from training.dataset import load_dataset
from training.model import build_mobilenetv2, unfreeze_top_layers

_LOG = logging.getLogger("bottle-train")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _build_callbacks(patience: int, model_dir: str) -> list[tf.keras.callbacks.Callback]:
    """Return the standard set of training callbacks."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_best_only=True,
            filepath=os.path.join(model_dir, "best_model.keras"),
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a MobileNetV2 bottle classifier with two-phase "
            "(frozen backbone → fine-tune) training."
        ),
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the dataset root with class subdirectories",
    )
    parser.add_argument(
        "--model-dir",
        default=Config.MODEL_SAVE_DIR,
        help=f"Directory to save trained models (default: {Config.MODEL_SAVE_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BATCH_SIZE,
        help=f"Batch size (default: {Config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--epochs-frozen",
        type=int,
        default=Config.FROZEN_EPOCHS,
        help=f"Maximum epochs, frozen backbone (default: {Config.FROZEN_EPOCHS})",
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=Config.FINETUNE_EPOCHS,
        help=f"Maximum epochs, fine-tuning (default: {Config.FINETUNE_EPOCHS})",
    )
    parser.add_argument(
        "--lr-frozen",
        type=float,
        default=Config.FROZEN_LR,
        help=f"Learning rate, frozen phase (default: {Config.FROZEN_LR})",
    )
    parser.add_argument(
        "--lr-finetune",
        type=float,
        default=Config.FINETUNE_LR,
        help=f"Learning rate, fine-tune phase (default: {Config.FINETUNE_LR})",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=Config.IMG_SIZE,
        help=f"Input image size in pixels (default: {Config.IMG_SIZE})",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return _build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> str:
    """Run the full two-phase training pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Path to the saved ``.h5`` model file.

    Raises:
        Exception: Propagated if training fails.
    """
    _setup_logging()
    _LOG.info("Starting bottle classifier training")
    _LOG.info("Data directory: %s", args.data_dir)
    _LOG.info("Model directory: %s", args.model_dir)

    # ---- Phase 0: load data ----
    _LOG.info("Loading dataset ...")
    train_ds, val_ds, class_names, class_counts = load_dataset(
        data_dir=args.data_dir,
        config=Config(
            IMG_SIZE=args.img_size,
            BATCH_SIZE=args.batch_size,
        ),
    )
    _LOG.info("Classes found: %s", class_names)
    for name, count in zip(class_names, class_counts):
        _LOG.info("  %s: %d images", name, count)

    # Compute class weights to balance the loss
    total = sum(class_counts)
    n_classes = len(class_counts)
    class_weight = {
        i: total / (n_classes * count)
        for i, count in enumerate(class_counts)
    }
    _LOG.info("Class weights: %s", class_weight)
    _LOG.info("  (minority classes get higher weight to balance training)")

    # ---- Build model (frozen backbone) ----
    _LOG.info("Building MobileNetV2 with frozen backbone ...")
    model = build_mobilenetv2(
        img_size=args.img_size,
        num_classes=len(class_names),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_frozen),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ---- Phase 1: train head with frozen backbone ----
    _LOG.info("=" * 60)
    _LOG.info("PHASE 1 — Training head (frozen backbone)")
    _LOG.info("=" * 60)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_frozen,
        callbacks=_build_callbacks(Config.PATIENCE, args.model_dir),
        class_weight=class_weight,
        verbose=1,
    )

    # ---- Phase 2: fine-tune top layers ----
    _LOG.info("=" * 60)
    _LOG.info("PHASE 2 — Fine-tuning top %d layers", Config.FINETUNE_TOP_LAYERS)
    _LOG.info("=" * 60)

    unfreeze_top_layers(model, num_layers=Config.FINETUNE_TOP_LAYERS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_finetune),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        callbacks=_build_callbacks(Config.PATIENCE, args.model_dir),
        class_weight=class_weight,
        verbose=1,
    )

    # ---- Save ----
    os.makedirs(args.model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.model_dir, f"bottle_classifier_{timestamp}.keras")
    model.save(save_path)
    _LOG.info("Model saved to: %s", save_path)

    # Also save a "latest" copy (always overwrites)
    latest_path = os.path.join(args.model_dir, "bottle_classifier_latest.keras")
    model.save(latest_path)
    _LOG.info("Latest model saved to: %s", latest_path)

    return save_path


def main() -> None:
    """Entry point for ``python training/train.py``."""
    args = parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        _LOG.info("Training interrupted by user")
        sys.exit(1)
    except Exception as exc:
        _LOG.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
