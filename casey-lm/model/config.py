"""Shared configuration types for all LM implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ImplName = Literal["torch", "tinygrad", "c"]


@dataclass
class LanguageModelConfig:
    vocab_size: int
    context_length: int
    embedding_dim: int
    num_decoder_layers: int
    num_heads: int
    dim_feedforward: int
    dropout: float


@dataclass
class TrainConfig:
    embedding_dim: int = 128
    num_decoder_layers: int = 3
    num_heads: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    lr: float = 1e-3
    lr_schedule: str = "cosine_warmup"
    warmup_steps: int | None = None
    warmup_fraction: float = 0.1
    min_lr_ratio: float = 0.1
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    epochs: int = 5
    max_batches: int | None = None
    max_val_batches: int | None = None
    log_every: int = 50
    save_every_epoch: bool = True
