"""Training configuration and hyperparameters.

All values use sensible defaults for the bottle detection use case.
Override via CLI arguments in ``train.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def auto_batch_size(max_batch: int = 32, safety_mb: int = 512) -> int:
    """Determine a safe batch size from available RAM."""
    try:
        import psutil  # noqa: PLC0415

        free_mb = psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        free_mb = int(line.split()[1]) / 1024.0
                        break
                else:
                    free_mb = 4096
        except OSError:
            free_mb = 4096

    inferred = int(free_mb / safety_mb)
    return max(1, min(inferred, max_batch))


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    IMG_SIZE: int = 224
    """Input image size (square)."""

    BACKBONE: str = "mobilenetv2"
    """Backbone CNN architecture name."""

    NUM_CLASSES: int = 3
    """Number of output classes."""

    CLASS_NAMES: tuple[str, ...] = field(
        default_factory=lambda: ("no_bottle", "pool_verde", "hatsu_morado"),
    )
    """Class label order, matching subdirectory names in the dataset."""

    # ------------------------------------------------------------------
    # Classification head
    # ------------------------------------------------------------------
    DENSE_UNITS: int = 256
    """Number of units in the first dense layer of the classification head."""

    DENSE_UNITS_2: int = 128
    """Number of units in the second dense layer of the classification head."""

    DROPOUT_RATE: float = 0.3
    """Dropout rate before the final softmax layer."""

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------
    BATCH_SIZE: int = field(default_factory=auto_batch_size)
    """Training and validation batch size (auto-calculated from available RAM)."""

    VALIDATION_SPLIT: float = 0.2
    """Fraction of data held out for validation (stratified)."""

    RANDOM_SEED: int = 42
    """Random seed for reproducibility."""

    # ------------------------------------------------------------------
    # Phase 1 — frozen backbone
    # ------------------------------------------------------------------
    FROZEN_EPOCHS: int = 20
    """Maximum epochs training with a frozen backbone."""

    FROZEN_LR: float = 5e-4
    """Learning rate for the frozen-backbone phase."""

    # ------------------------------------------------------------------
    # Phase 2 — fine-tuning
    # ------------------------------------------------------------------
    FINETUNE_EPOCHS: int = 15
    """Maximum epochs for fine-tuning the top layers."""

    FINETUNE_LR: float = 1e-5
    """Learning rate for the fine-tuning phase (much lower)."""

    FINETUNE_TOP_LAYERS: int = 30
    """Number of backbone layers to unfreeze for fine-tuning."""

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    PATIENCE: int = 5
    """Early stopping patience (monitors val_loss)."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    MODEL_SAVE_DIR: str = "models"
    """Directory where trained models are saved."""
