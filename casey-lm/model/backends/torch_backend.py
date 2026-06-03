"""PyTorch backend for train/run."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from data.lm_dataloader import DataConfig
from model.backends.base import Backend, CheckpointState, SaveCheckpointArgs
from model.config import ImplName, LanguageModelConfig, TrainConfig
from model.torch_impl.language_model import TorchLanguageModel


class TorchBackend(Backend):
    name: ImplName = "torch"

    def resolve_device(self, requested: str | None) -> torch.device:
        if requested is not None:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def build_model(self, config: LanguageModelConfig, device: torch.device) -> TorchLanguageModel:
        return TorchLanguageModel(config).to(device)

    def count_parameters(self, model: Any) -> int:
        return sum(p.numel() for p in model.parameters())

    def set_training_mode(self, model: Any, training: bool) -> None:
        if training:
            model.train()
        else:
            model.eval()

    def prepare_batch(self, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        return batch.to(device)

    def _split_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return batch[:, :-1], batch[:, 1:]

    def train_step(
        self,
        model: Any,
        optimizer: Any,
        batch: torch.Tensor,
        *,
        vocab_size: int,
    ) -> float:
        inputs, labels = self._split_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def eval_batch(
        self,
        model: Any,
        batch: torch.Tensor,
        *,
        vocab_size: int,
    ) -> tuple[float, int, int]:
        inputs, labels = self._split_batch(batch)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
        )
        predictions = logits.argmax(dim=-1)
        num_correct = int((predictions == labels).sum().item())
        num_tokens = labels.numel()
        return float(loss.item()), num_correct, num_tokens

    def create_optimizer(self, model: Any, lr: float) -> Adam:
        return Adam(model.parameters(), lr=lr)

    def get_lr(self, optimizer: Adam) -> float:
        return float(optimizer.param_groups[0]["lr"])

    def set_lr(self, optimizer: Adam, lr: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = lr

    def save_checkpoint(self, args: SaveCheckpointArgs) -> None:
        args.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "impl": args.impl,
            "epoch": args.epoch,
            "global_step": args.global_step,
            "model_state_dict": args.model.state_dict(),
            "optimizer_state_dict": args.optimizer.state_dict(),
            "model_config": asdict(args.model_config),
            "train_config": asdict(args.train_config),
            "data_config": asdict(args.data_config),
            "val_loss": args.val_loss,
            "val_accuracy": args.val_accuracy,
        }
        torch.save(payload, args.path)

    def load_checkpoint(
        self,
        path: Path,
        device: torch.device,
        *,
        model: Any | None = None,
    ) -> tuple[TorchLanguageModel, Adam, CheckpointState]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        impl: ImplName = ckpt.get("impl", "torch")
        if impl != "torch":
            raise ValueError(f"Checkpoint impl={impl!r} does not match torch backend")

        model_config = LanguageModelConfig(**ckpt["model_config"])
        if model is None:
            model = self.build_model(model_config, device)
        model.load_state_dict(ckpt["model_state_dict"])
        self.set_training_mode(model, training=False)

        train_config = TrainConfig(**ckpt["train_config"])
        optimizer = self.create_optimizer(model, train_config.lr)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        data_config = DataConfig(**ckpt["data_config"])
        state = CheckpointState(
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
        return model, optimizer, state

    @torch.no_grad()
    def forward_logits(
        self,
        model: Any,
        input_ids: list[int],
        device: torch.device,
    ) -> np.ndarray:
        self.set_training_mode(model, training=False)
        tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits = model(tensor)
        return logits[0].detach().cpu().numpy()

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
