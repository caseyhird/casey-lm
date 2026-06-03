"""Framework-agnostic learning rate schedule math."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from model.config import TrainConfig


def resolve_warmup_steps(
    total_steps: int,
    *,
    warmup_steps: int | None,
    warmup_fraction: float,
) -> int:
    if total_steps <= 0:
        return 0
    if warmup_steps is not None:
        return min(max(warmup_steps, 0), total_steps)
    return min(max(int(total_steps * warmup_fraction), 1), total_steps)


def lr_at_step(step: int, train_config: TrainConfig, *, total_steps: int) -> float:
    """
    Learning rate for batch `step` (1-indexed global step after increment).

    Supports constant and cosine_warmup. Plateau is handled separately at epoch
    boundaries via PlateauState.
    """
    schedule = train_config.lr_schedule
    base_lr = train_config.lr

    if schedule == "constant":
        return base_lr

    if schedule == "cosine_warmup":
        if total_steps <= 0:
            return base_lr
        warmup_steps = resolve_warmup_steps(
            total_steps,
            warmup_steps=train_config.warmup_steps,
            warmup_fraction=train_config.warmup_fraction,
        )
        min_lr = base_lr * train_config.min_lr_ratio

        # Match torch LinearLR(start_factor=1e-8, end_factor=1.0) during warmup.
        if warmup_steps > 0 and step <= warmup_steps:
            progress = step / warmup_steps
            factor = 1e-8 + (1.0 - 1e-8) * progress
            return base_lr * factor

        cosine_steps = max(total_steps - warmup_steps, 1)
        t = min(max(step - warmup_steps, 0), cosine_steps)
        # Falls along curve from base_lr to min_lr
        # see https://scorrea92.medium.com/cosine-learning-rate-decay-e8b50aa455b
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t / cosine_steps))

    if schedule == "plateau":
        return base_lr

    raise ValueError(f"Unknown lr schedule: {schedule}")


@dataclass
class PlateauState:
    """Mutable state for ReduceLROnPlateau-style epoch scheduling."""

    current_lr: float
    best_val_loss: float = field(default_factory=lambda: float("inf"))
    epochs_without_improvement: int = 0

    def step_epoch(self, val_loss: float, train_config: TrainConfig) -> tuple[float, bool]:
        """
        Update plateau state after an epoch.

        Returns (new_lr, lr_was_reduced).
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return self.current_lr, False

        self.epochs_without_improvement += 1
        if self.epochs_without_improvement > train_config.plateau_patience:
            new_lr = max(self.current_lr * train_config.plateau_factor, 0.0)
            if new_lr < self.current_lr:
                self.current_lr = new_lr
                self.epochs_without_improvement = 0
                return self.current_lr, True
        return self.current_lr, False


def describe_lr_schedule(
    train_config: TrainConfig,
    *,
    total_steps: int,
    batches_per_epoch_count: int,
) -> str:
    schedule = train_config.lr_schedule
    if schedule == "constant":
        return f"constant lr={train_config.lr:g}"

    if schedule == "cosine_warmup":
        warmup_steps = resolve_warmup_steps(
            total_steps,
            warmup_steps=train_config.warmup_steps,
            warmup_fraction=train_config.warmup_fraction,
        )
        min_lr = train_config.lr * train_config.min_lr_ratio
        return (
            f"cosine_warmup base_lr={train_config.lr:g} warmup_steps={warmup_steps} "
            f"total_steps={total_steps} min_lr={min_lr:g} "
            f"({batches_per_epoch_count} batches/epoch)"
        )

    if schedule == "plateau": 
        (
            f"plateau base_lr={train_config.lr:g} factor={train_config.plateau_factor} "
            f"patience={train_config.plateau_patience}"
        )
    
    return f"Unknown lr schedule: {schedule}"
