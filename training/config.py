"""Training configuration and hyperparameters.

All values use sensible defaults for the bottle detection use case.
Override via CLI arguments in ``train.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    BATCH_SIZE: int = 32
    """Training and validation batch size."""

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
