"""
C/C++ backend stub for train/run.

Future pybind11 module `casey_lm_c` should expose:

    class Model:
        def __init__(self, config: dict): ...
        def forward(self, input_ids: np.ndarray) -> np.ndarray: ...
        def train_step(self, input_ids, labels, lr) -> float: ...
        def num_parameters(self) -> int: ...
        def save(self, path: str) -> None: ...
        @staticmethod
        def load(path: str) -> Model: ...

    def set_seed(seed: int) -> None: ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.backends.base import Backend, CheckpointState, SaveCheckpointArgs
from model.config import ImplName, LanguageModelConfig

_NOT_BUILT = (
    "C backend is not built. Build the casey_lm_c extension (pybind11) and install "
    "with BUILD_C=1 before using --impl c."
)


class CBackend(Backend):
    name: ImplName = "c"

    def __init__(self) -> None:
        try:
            import casey_lm_c  # type: ignore[import-not-found]

            self._native = casey_lm_c
        except ImportError:
            self._native = None

    def _require_native(self) -> Any:
        if self._native is None:
            raise RuntimeError(_NOT_BUILT)
        return self._native

    def resolve_device(self, requested: str | None) -> str:
        return requested or "cpu"

    def build_model(self, config: LanguageModelConfig, device: str) -> Any:
        native = self._require_native()
        return native.Model(_model_config_to_dict(config))

    def count_parameters(self, model: Any) -> int:
        return int(model.num_parameters())

    def set_training_mode(self, model: Any, training: bool) -> None:
        model.set_training(training)

    def prepare_batch(self, batch: torch.Tensor, device: str) -> np.ndarray:
        return batch.numpy().astype(np.int64)

    def train_step(
        self,
        model: Any,
        optimizer: Any,
        batch: np.ndarray,
        *,
        vocab_size: int,
    ) -> float:
        inputs = batch[:, :-1]
        labels = batch[:, 1:]
        return float(model.train_step(inputs, labels, optimizer.lr))

    def eval_batch(
        self,
        model: Any,
        batch: np.ndarray,
        *,
        vocab_size: int,
    ) -> tuple[float, int, int]:
        inputs = batch[:, :-1]
        labels = batch[:, 1:]
        return model.eval_batch(inputs, labels)

    def create_optimizer(self, model: Any, lr: float) -> Any:
        return _COptimizer(lr)

    def get_lr(self, optimizer: Any) -> float:
        return float(optimizer.lr)

    def set_lr(self, optimizer: Any, lr: float) -> None:
        optimizer.lr = lr
        
    def set_seed(self, seed: int) -> None:
        raise RuntimeError(_NOT_BUILT)

    def save_checkpoint(self, args: SaveCheckpointArgs) -> None:
        native = self._require_native()
        args.path.parent.mkdir(parents=True, exist_ok=True)
        weights_path = args.path.with_suffix(".bin")
        args.model.save(str(weights_path))
        native.save_checkpoint_meta(
            str(args.path),
            impl=args.impl,
            epoch=args.epoch,
            global_step=args.global_step,
            model_config=_model_config_to_dict(args.model_config),
            train_config=_train_config_to_dict(args.train_config),
            data_config=_data_config_to_dict(args.data_config),
            val_loss=args.val_loss,
            val_accuracy=args.val_accuracy,
            weights_path=str(weights_path),
        )

    def load_checkpoint(
        self,
        path: Path,
        device: str,
        *,
        model: Any | None = None,
    ) -> tuple[Any, Any, CheckpointState]:
        raise RuntimeError(_NOT_BUILT)

    def forward_logits(
        self,
        model: Any,
        input_ids: list[int],
        device: str,
    ) -> np.ndarray:
        native = self._require_native()
        arr = np.array([input_ids], dtype=np.int64)
        logits = model.forward(arr)
        return np.asarray(logits[0])

    def set_seed(self, seed: int) -> None:
        if self._native is not None:
            self._native.set_seed(seed)


class _COptimizer:
    def __init__(self, lr: float) -> None:
        self.lr = lr


def _model_config_to_dict(config: LanguageModelConfig) -> dict:
    from dataclasses import asdict

    return asdict(config)


def _train_config_to_dict(config) -> dict:
    from dataclasses import asdict

    return asdict(config)


def _data_config_to_dict(config) -> dict:
    from dataclasses import asdict

    return asdict(config)
