"""Tinygrad backend for train/run."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.lm_dataloader import DataConfig
from model.backends.base import Backend, CheckpointState, SaveCheckpointArgs
from model.config import ImplName, LanguageModelConfig, TrainConfig
from model.tinygrad_impl.language_model import TinygradLanguageModel


def _map_device_label(requested: str) -> str:
    key = requested.lower()
    if key in ("cuda", "gpu"):
        return "CUDA"
    if key == "mps":
        return "METAL"
    if key == "cpu":
        return "LLVM"
    return requested.upper()


def _default_device_label() -> str:
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    from tinygrad import Tensor

    return str(Tensor([0]).device).split(":")[0]


# TODO: could use tinygrad safe_save if we want to not use .pt files for all
def _state_dict_to_numpy(state_dict: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in state_dict.items():
        if hasattr(value, "numpy"):
            out[key] = np.asarray(value.numpy())
        else:
            out[key] = np.asarray(value)
    return out


def _state_dict_from_numpy(
    state_dict: dict[str, np.ndarray],
    *,
    target_shapes: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    from tinygrad import Tensor

    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        arr = np.asarray(value)
        if target_shapes is not None and key in target_shapes:
            expected = target_shapes[key]
            if arr.shape != expected and {arr.shape, expected} == {(), (1,)}:
                arr = arr.reshape(expected)
        out[key] = Tensor(arr)
    return out


class TinygradBackend(Backend):
    name: ImplName = "tinygrad"

    def resolve_device(self, requested: str | None) -> str:
        if requested is None:
            return _default_device_label()
        device = _map_device_label(requested)
        os.environ["DEVICE"] = device
        return device

    def build_model(self, config: LanguageModelConfig, device: str) -> TinygradLanguageModel:
        if device != "default":
            os.environ["DEVICE"] = device
        return TinygradLanguageModel(config)

    def count_parameters(self, model: Any) -> int:
        import tinygrad.nn.state as state

        return sum(p.numel() for p in state.get_parameters(model))

    def set_training_mode(self, model: Any, training: bool) -> None:
        from tinygrad import Tensor

        Tensor.training = training

    def prepare_batch(self, batch: torch.Tensor, device: str) -> Any:
        from tinygrad import Tensor

        data = batch.numpy().astype(np.int32)
        if device == "default":
            return Tensor(data)
        return Tensor(data, device=device)

    def _split_batch(self, batch: Any) -> tuple[Any, Any]:
        return batch[:, :-1], batch[:, 1:]

    def _compute_loss(self, logits: Any, labels: Any) -> Any:
        return logits.sparse_categorical_crossentropy(labels)

    def train_step(
        self,
        model: Any,
        optimizer: Any,
        batch: Any,
        *,
        vocab_size: int,
    ) -> float:
        del vocab_size
        self.set_training_mode(model, training=True)
        inputs, labels = self._split_batch(batch)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = self._compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        return float(np.asarray(loss.numpy()).item())

    def eval_batch(
        self,
        model: Any,
        batch: Any,
        *,
        vocab_size: int,
    ) -> tuple[float, int, int]:
        del vocab_size
        self.set_training_mode(model, training=False)
        inputs, labels = self._split_batch(batch)
        logits = model(inputs)
        loss = self._compute_loss(logits, labels)
        predictions = logits.argmax(axis=-1)
        pred_np = predictions.numpy()
        labels_np = labels.numpy()
        num_correct = int(np.sum(pred_np == labels_np))
        num_tokens = int(labels_np.size)
        return float(np.asarray(loss.numpy()).item()), num_correct, num_tokens

    def create_optimizer(self, model: Any, lr: float) -> Any:
        import tinygrad.nn.optim as optim
        import tinygrad.nn.state as state

        return optim.Adam(state.get_parameters(model), lr=lr)

    def get_lr(self, optimizer: Any) -> float:
        lr = optimizer.lr
        if hasattr(lr, "numpy"):
            return float(np.asarray(lr.numpy()).item())
        return float(lr)

    def set_lr(self, optimizer: Any, lr: float) -> None:
        from tinygrad import Tensor

        current = optimizer.lr
        if isinstance(current, Tensor):
            optimizer.lr = Tensor(lr, device=current.device, dtype=current.dtype)
        else:
            optimizer.lr = lr

    def set_seed(self, seed: int) -> None:
        from tinygrad import Tensor

        Tensor.manual_seed(seed)
        np.random.seed(seed)

    def save_checkpoint(self, args: SaveCheckpointArgs) -> None:
        args.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint_payload(args)
        payload["model_state_dict"] = _state_dict_to_numpy(payload["model_state_dict"])
        torch.save(payload, args.path)

    def load_checkpoint(
        self,
        path: Path,
        device: str,
        *,
        model: Any | None = None,
    ) -> tuple[Any, Any, CheckpointState]:
        import tinygrad.nn.state as state

        if device != "default":
            os.environ["DEVICE"] = device
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        impl: ImplName = ckpt.get("impl", "tinygrad")
        if impl != "tinygrad":
            raise ValueError(f"Checkpoint impl={impl!r} does not match tinygrad backend")

        model_config = LanguageModelConfig(**ckpt["model_config"])
        if model is None:
            model = self.build_model(model_config, device)
        model_tensors = state.get_state_dict(model)
        model_shapes = {key: tuple(tensor.shape) for key, tensor in model_tensors.items()}
        state.load_state_dict(
            model,
            _state_dict_from_numpy(ckpt["model_state_dict"], target_shapes=model_shapes),
        )
        self.set_training_mode(model, training=False)

        train_config = TrainConfig(**ckpt["train_config"])
        optimizer = self.create_optimizer(model, train_config.lr)

        data_config = DataConfig(**ckpt["data_config"])
        ckpt_state = CheckpointState(
            impl=impl,
            epoch=int(ckpt["epoch"]),
            global_step=int(ckpt.get("global_step", 0)),
            model_config=model_config,
            train_config=train_config,
            data_config=data_config,
            val_loss=float(ckpt.get("val_loss", float("inf"))),
            val_accuracy=float(ckpt.get("val_accuracy", 0.0)),
            raw=ckpt,
        )
        return model, optimizer, ckpt_state

    def forward_logits(
        self,
        model: Any,
        input_ids: list[int],
        device: str,
    ) -> np.ndarray:
        from tinygrad import Tensor

        self.set_training_mode(model, training=False)
        if device == "default":
            tensor = Tensor([input_ids])
        else:
            tensor = Tensor([input_ids], device=device)
        logits = model(tensor)
        return np.asarray(logits[0].numpy())

    def _checkpoint_payload(self, args: SaveCheckpointArgs) -> dict:
        import tinygrad.nn.state as state

        return {
            "impl": args.impl,
            "epoch": args.epoch,
            "global_step": args.global_step,
            "model_state_dict": state.get_state_dict(args.model),
            "model_config": asdict(args.model_config),
            "train_config": asdict(args.train_config),
            "data_config": asdict(args.data_config),
            "val_loss": args.val_loss,
            "val_accuracy": args.val_accuracy,
        }
