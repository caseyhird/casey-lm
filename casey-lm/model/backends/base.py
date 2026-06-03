"""Backend protocol for multi-implementation train/run."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.lm_dataloader import DataConfig
from model.config import ImplName, LanguageModelConfig, TrainConfig


@dataclass
class CheckpointState:
    """Framework-agnostic checkpoint metadata returned by load_checkpoint."""

    impl: ImplName
    epoch: int
    global_step: int
    model_config: LanguageModelConfig
    train_config: TrainConfig
    data_config: DataConfig
    val_loss: float
    val_accuracy: float
    raw: dict[str, Any]


@dataclass
class SaveCheckpointArgs:
    path: Path
    model: Any
    optimizer: Any
    epoch: int
    global_step: int
    model_config: LanguageModelConfig
    train_config: TrainConfig
    data_config: DataConfig
    val_loss: float
    val_accuracy: float
    impl: ImplName


class Backend(ABC):
    """Interface each LM implementation must satisfy for train.py / run.py."""

    name: ImplName

    @abstractmethod
    def resolve_device(self, requested: str | None) -> Any:
        """Return a backend-native device identifier."""

    @abstractmethod
    def build_model(self, config: LanguageModelConfig, device: Any) -> Any:
        """Construct an untrained model on the given device."""

    @abstractmethod
    def count_parameters(self, model: Any) -> int:
        """Return the number of trainable parameters."""

    @abstractmethod
    def set_training_mode(self, model: Any, training: bool) -> None:
        """Switch between train and eval modes (including dropout)."""

    @abstractmethod
    def prepare_batch(self, batch: torch.Tensor, device: Any) -> Any:
        """Convert a dataloader batch to backend-native tensors."""

    @abstractmethod
    def train_step(
        self,
        model: Any,
        optimizer: Any,
        batch: Any,
        *,
        vocab_size: int,
    ) -> float:
        """Run one training step; return scalar loss."""

    @abstractmethod
    def eval_batch(
        self,
        model: Any,
        batch: Any,
        *,
        vocab_size: int,
    ) -> tuple[float, int, int]:
        """Return (batch_mean_loss, num_correct, num_tokens)."""

    @abstractmethod
    def create_optimizer(self, model: Any, lr: float) -> Any:
        """Create a backend-native optimizer."""

    @abstractmethod
    def get_lr(self, optimizer: Any) -> float:
        """Return the current learning rate."""

    @abstractmethod
    def set_lr(self, optimizer: Any, lr: float) -> None:
        """Set the learning rate on all parameter groups."""

    @abstractmethod
    def save_checkpoint(self, args: SaveCheckpointArgs) -> None:
        """Persist model weights and shared metadata."""

    @abstractmethod
    def load_checkpoint(
        self,
        path: Path,
        device: Any,
        *,
        model: Any | None = None,
    ) -> tuple[Any, Any, CheckpointState]:
        """Load weights; return (model, optimizer, metadata)."""

    @abstractmethod
    def forward_logits(
        self,
        model: Any,
        input_ids: list[int],
        device: Any,
    ) -> np.ndarray:
        """Forward pass returning logits with shape [seq_len, vocab_size]."""

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Set backend RNG seed for reproducible sampling (optional override)."""
