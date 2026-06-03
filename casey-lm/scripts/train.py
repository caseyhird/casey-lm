#!/usr/bin/env python3
"""Train Casey LM using any registered backend (--impl torch|tinygrad|c)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from data.lm_dataloader import DataConfig, get_dataloaders
from scripts.lr_schedule import PlateauState, describe_lr_schedule, lr_at_step
from model.backends import get_backend, registered_impls
from model.backends.base import Backend, SaveCheckpointArgs
from model.config import ImplName, LanguageModelConfig, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Casey LM")
    parser.add_argument(
        "--impl",
        choices=registered_impls(),
        default="torch",
        help="Model implementation backend",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/train"))
    parser.add_argument("--device", type=str, default=None, help="Device (backend-specific, e.g. cuda/cpu/mps)")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")

    # Data
    parser.add_argument("--dataset", choices=["wikitext", "shakespeare"], default="wikitext")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128, help="Tokens per training sequence")
    parser.add_argument("--num-workers", type=int, default=0)

    # Model / train
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine_warmup", "plateau"],
        default="cosine_warmup",
        help="Learning rate schedule (default: linear warmup + cosine decay)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Linear warmup steps (default: warmup_fraction * total train steps)",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.1,
        help="Warmup as a fraction of total training steps when --warmup-steps is unset",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Minimum LR for cosine schedule, as a fraction of --lr",
    )
    parser.add_argument(
        "--plateau-factor",
        type=float,
        default=0.5,
        help="LR reduction factor for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=2,
        help="Epochs without val-loss improvement before LR reduction (plateau schedule)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=None, help="Limit train batches per epoch (debug)")
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-save-epoch-checkpoints", action="store_true")
    return parser.parse_args()


def batches_per_epoch(num_loader_batches: int, max_batches: int | None) -> int:
    if max_batches is None:
        return num_loader_batches
    return min(num_loader_batches, max_batches)


def compute_total_training_steps(
    num_loader_batches: int,
    *,
    epochs: int,
    start_epoch: int,
    max_batches: int | None,
) -> int:
    steps_per_epoch = batches_per_epoch(num_loader_batches, max_batches)
    remaining_epochs = max(epochs - start_epoch, 0)
    return steps_per_epoch * remaining_epochs


class TrainLogger:
    """Structured numeric metrics written to metrics.csv."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = log_dir / "metrics.csv"
        self._file = self.metrics_path.open("w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "timestamp",
                "epoch",
                "step",
                "split",
                "loss",
                "accuracy",
                "lr",
            ],
        )
        self._writer.writeheader()

    def log(
        self,
        *,
        epoch: int,
        step: int,
        split: str,
        loss: float,
        accuracy: float | None = None,
        lr: float | None = None,
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "step": step,
            "split": split,
            "loss": f"{loss:.6f}",
            "accuracy": "" if accuracy is None else f"{accuracy:.6f}",
            "lr": "" if lr is None else f"{lr:.6g}",
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class TextLogger:
    """Human-readable run log written to train.log (separate from metrics.csv)."""

    def __init__(self, log_dir: Path, *, echo: bool = True, append: bool = False):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "train.log"
        self._echo = echo
        mode = "a" if append else "w"
        self._file: TextIO = self.log_path.open(mode, encoding="utf-8")
        self.info(f"Opened text log at {self.log_path} (mode={'append' if append else 'write'})")

    def _write(self, level: str, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"{timestamp} [{level}] {message}"
        self._file.write(line + "\n")
        self._file.flush()
        if self._echo:
            print(line)

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARNING", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def close(self) -> None:
        self.info("Closed text log.")
        self._file.close()


def save_checkpoint(
    backend: Backend,
    path: Path,
    *,
    impl: ImplName,
    model: Any,
    optimizer: Any,
    epoch: int,
    global_step: int,
    model_config: LanguageModelConfig,
    train_config: TrainConfig,
    data_config: DataConfig,
    val_loss: float,
    val_accuracy: float,
    text_logger: TextLogger | None = None,
) -> None:
    try:
        backend.save_checkpoint(
            SaveCheckpointArgs(
                path=path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                model_config=model_config,
                train_config=train_config,
                data_config=data_config,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                impl=impl,
            )
        )
        if text_logger is not None:
            text_logger.info(f"Saved checkpoint to {path}")
    except OSError as exc:
        if text_logger is not None:
            text_logger.error(f"Failed to save checkpoint to {path}: {exc}")
        raise


def train_epoch(
    backend: Backend,
    model: Any,
    loader,
    optimizer: Any,
    device: Any,
    vocab_size: int,
    train_config: TrainConfig,
    total_steps: int,
    *,
    epoch: int,
    logger: TrainLogger,
    text_logger: TextLogger,
    global_step: int,
    max_batches: int | None,
    log_every: int,
) -> tuple[float, int]:
    backend.set_training_mode(model, training=True)
    total_loss = 0.0
    num_batches = 0
    epoch_start_lr = backend.get_lr(optimizer)

    if max_batches is not None:
        text_logger.warning(
            f"Epoch {epoch + 1}: limiting training to {max_batches} batches (--max-batches)"
        )

    progress = tqdm(loader, desc=f"train epoch {epoch + 1}", leave=False)
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        native_batch = backend.prepare_batch(batch, device)

        if train_config.lr_schedule in ("constant", "cosine_warmup"):
            next_step = global_step + 1
            lr = lr_at_step(next_step, train_config, total_steps=total_steps)
            backend.set_lr(optimizer, lr)

        loss_value = backend.train_step(model, optimizer, native_batch, vocab_size=vocab_size)
        total_loss += loss_value
        num_batches += 1
        global_step += 1

        lr = backend.get_lr(optimizer)
        progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr:.2e}")
        if log_every > 0 and global_step % log_every == 0:
            logger.log(epoch=epoch + 1, step=global_step, split="train", loss=loss_value, lr=lr)

    mean_loss = total_loss / max(num_batches, 1)
    epoch_end_lr = backend.get_lr(optimizer)
    text_logger.info(
        f"Epoch {epoch + 1} train finished: {num_batches} batches, mean_loss={mean_loss:.6f}, "
        f"step={global_step}, lr={epoch_start_lr:.6g}->{epoch_end_lr:.6g}"
    )
    return mean_loss, global_step


def evaluate(
    backend: Backend,
    model: Any,
    loader,
    device: Any,
    vocab_size: int,
    *,
    max_batches: int | None,
    text_logger: TextLogger | None = None,
) -> tuple[float, float]:
    backend.set_training_mode(model, training=False)
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    if max_batches is not None and text_logger is not None:
        text_logger.warning(f"Limiting validation to {max_batches} batches (--max-val-batches)")

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        native_batch = backend.prepare_batch(batch, device)
        loss, num_correct, num_tokens = backend.eval_batch(
            model, native_batch, vocab_size=vocab_size
        )
        total_loss += loss
        total_correct += num_correct
        total_tokens += num_tokens
        num_batches += 1

    mean_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_tokens, 1)
    return mean_loss, accuracy


def write_run_config(
    output_dir: Path,
    *,
    impl: ImplName,
    model_config: LanguageModelConfig,
    train_config: TrainConfig,
    data_config: DataConfig,
    device: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "impl": impl,
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "data_config": asdict(data_config),
                "device": device,
            },
            indent=2,
        )
    )


def align_lr_after_resume(
    backend: Backend,
    optimizer: Any,
    train_config: TrainConfig,
    global_step: int,
    total_steps: int,
) -> None:
    """Set LR from schedule state when resuming (replaces torch scheduler restore)."""
    if train_config.lr_schedule == "cosine_warmup":
        lr = lr_at_step(global_step, train_config, total_steps=total_steps)
        backend.set_lr(optimizer, lr)
    elif train_config.lr_schedule == "constant":
        backend.set_lr(optimizer, train_config.lr)


def main() -> None:
    args = parse_args()
    impl: ImplName = args.impl
    backend = get_backend(impl)
    device = backend.resolve_device(args.device)

    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    text_logger = TextLogger(log_dir, append=args.resume is not None)
    metrics_logger = TrainLogger(log_dir)

    try:
        text_logger.info(f"Backend: {impl}")
        if args.device is None:
            text_logger.info(f"No device requested; selected {device}")
        else:
            text_logger.info(f"Using requested device: {device}")

        data_config = DataConfig(
            dataset=args.dataset,
            tokenizer_name=args.tokenizer,
            batch_size=args.batch_size,
            block_size=args.block_size,
            num_workers=args.num_workers,
        )
        train_config = TrainConfig(
            embedding_dim=args.embedding_dim,
            num_decoder_layers=args.num_decoder_layers,
            num_heads=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            lr=args.lr,
            lr_schedule=args.lr_schedule,
            warmup_steps=args.warmup_steps,
            warmup_fraction=args.warmup_fraction,
            min_lr_ratio=args.min_lr_ratio,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            epochs=args.epochs,
            max_batches=args.max_batches,
            max_val_batches=args.max_val_batches,
            log_every=args.log_every,
            save_every_epoch=not args.no_save_epoch_checkpoints,
        )

        if train_config.embedding_dim % train_config.num_heads != 0:
            text_logger.warning(
                "embedding_dim is not divisible by num_heads; attention may fail or behave unexpectedly"
            )

        text_logger.info(
            f"Run output directory: {output_dir.resolve()} | "
            f"dataset={data_config.dataset} block_size={data_config.block_size} "
            f"batch_size={data_config.batch_size} tokenizer={data_config.tokenizer_name}"
        )
        text_logger.info(
            f"Model: embedding_dim={train_config.embedding_dim} layers={train_config.num_decoder_layers} "
            f"heads={train_config.num_heads} ff={train_config.dim_feedforward} dropout={train_config.dropout}"
        )
        text_logger.info(
            f"Train: epochs={train_config.epochs} lr={train_config.lr} "
            f"lr_schedule={train_config.lr_schedule} save_every_epoch={train_config.save_every_epoch}"
        )

        text_logger.info("Loading dataset and building dataloaders...")
        train_loader, val_loader, vocab_size, tokenizer = get_dataloaders(data_config)
        steps_per_epoch_count = batches_per_epoch(len(train_loader), train_config.max_batches)
        text_logger.info(
            f"Data ready: vocab_size={vocab_size} train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} effective_train_batches/epoch={steps_per_epoch_count}"
        )
        tokenizer.save_pretrained(output_dir / "tokenizer")
        text_logger.info(f"Saved tokenizer to {output_dir / 'tokenizer'}")

        model_config = LanguageModelConfig(
            vocab_size=vocab_size,
            context_length=data_config.block_size,
            embedding_dim=train_config.embedding_dim,
            num_decoder_layers=train_config.num_decoder_layers,
            num_heads=train_config.num_heads,
            dim_feedforward=train_config.dim_feedforward,
            dropout=train_config.dropout,
        )

        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")
        plateau_state: PlateauState | None = None

        if args.resume is not None:
            if not args.resume.exists():
                text_logger.error(f"Resume checkpoint not found: {args.resume}")
                raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
            text_logger.info(f"Loading checkpoint from {args.resume}")
            model, optimizer, ckpt_state = backend.load_checkpoint(
                args.resume.resolve(), device
            )
            if ckpt_state.impl != impl:
                text_logger.warning(
                    f"Checkpoint impl={ckpt_state.impl!r} differs from --impl={impl!r}"
                )
            start_epoch = ckpt_state.epoch
            global_step = ckpt_state.global_step
            best_val_loss = ckpt_state.val_loss
            model_config = ckpt_state.model_config
            train_config = ckpt_state.train_config
            data_config = ckpt_state.data_config
            text_logger.info(
                f"Resumed training from epoch {start_epoch}, step {global_step}, "
                f"checkpoint val_loss={best_val_loss:.6f}"
            )
        else:
            model = backend.build_model(model_config, device)
            optimizer = backend.create_optimizer(model, train_config.lr)

        num_params = backend.count_parameters(model)
        text_logger.info(f"Built model with {num_params:,} parameters on {device}")

        total_steps = compute_total_training_steps(
            len(train_loader),
            epochs=train_config.epochs,
            start_epoch=start_epoch,
            max_batches=train_config.max_batches,
        )
        text_logger.info(
            describe_lr_schedule(
                train_config,
                total_steps=total_steps,
                batches_per_epoch_count=steps_per_epoch_count,
            )
        )

        if args.resume is not None:
            align_lr_after_resume(
                backend, optimizer, train_config, global_step, total_steps
            )
            text_logger.info(
                f"Aligned LR after resume: lr={backend.get_lr(optimizer):.6g} at step {global_step}"
            )

        if train_config.lr_schedule == "plateau":
            plateau_state = PlateauState(current_lr=backend.get_lr(optimizer))

        write_run_config(
            output_dir,
            impl=impl,
            model_config=model_config,
            train_config=train_config,
            data_config=data_config,
            device=str(device),
        )
        text_logger.info(f"Wrote run config to {output_dir / 'config.json'}")

        if start_epoch >= train_config.epochs:
            text_logger.warning(
                f"Start epoch ({start_epoch}) is already >= total epochs ({train_config.epochs}); nothing to train"
            )

        text_logger.info("Starting training loop")
        for epoch in range(start_epoch, train_config.epochs):
            text_logger.info(
                f"Epoch {epoch + 1}/{train_config.epochs} started (lr={backend.get_lr(optimizer):.6g})"
            )
            train_loss, global_step = train_epoch(
                backend,
                model,
                train_loader,
                optimizer,
                device,
                vocab_size,
                train_config,
                total_steps,
                epoch=epoch,
                logger=metrics_logger,
                text_logger=text_logger,
                global_step=global_step,
                max_batches=train_config.max_batches,
                log_every=train_config.log_every,
            )
            val_loss, val_accuracy = evaluate(
                backend,
                model,
                val_loader,
                device,
                vocab_size,
                max_batches=train_config.max_val_batches,
                text_logger=text_logger,
            )

            if train_config.lr_schedule == "plateau" and plateau_state is not None:
                lr_before = backend.get_lr(optimizer)
                new_lr, reduced = plateau_state.step_epoch(val_loss, train_config)
                backend.set_lr(optimizer, new_lr)
                if reduced:
                    text_logger.info(
                        f"ReduceLROnPlateau triggered at epoch {epoch + 1}: "
                        f"lr {lr_before:.6g} -> {new_lr:.6g} (val_loss={val_loss:.6f})"
                    )

            epoch_lr = backend.get_lr(optimizer)
            metrics_logger.log(
                epoch=epoch + 1, step=global_step, split="train_epoch", loss=train_loss, lr=epoch_lr
            )
            metrics_logger.log(
                epoch=epoch + 1, step=global_step, split="val", loss=val_loss, accuracy=val_accuracy
            )

            text_logger.info(
                f"Epoch {epoch + 1}/{train_config.epochs} complete | "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_accuracy:.6f} lr={epoch_lr:.6g}"
            )

            save_checkpoint(
                backend,
                checkpoint_dir / "last.pt",
                impl=impl,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                model_config=model_config,
                train_config=train_config,
                data_config=data_config,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                text_logger=text_logger,
            )
            if val_loss < best_val_loss:
                previous_best = best_val_loss
                best_val_loss = val_loss
                save_checkpoint(
                    backend,
                    checkpoint_dir / "best.pt",
                    impl=impl,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                    model_config=model_config,
                    train_config=train_config,
                    data_config=data_config,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    text_logger=text_logger,
                )
                if previous_best == float("inf"):
                    text_logger.info(
                        f"New best checkpoint at epoch {epoch + 1}: val_loss={val_loss:.6f} "
                        f"val_acc={val_accuracy:.6f} (first validation)"
                    )
                else:
                    text_logger.info(
                        f"New best checkpoint at epoch {epoch + 1}: val_loss={val_loss:.6f} "
                        f"val_acc={val_accuracy:.6f} (improved from {previous_best:.6f})"
                    )
            else:
                text_logger.info(
                    f"No improvement at epoch {epoch + 1}: val_loss={val_loss:.6f} "
                    f"(best remains {best_val_loss:.6f})"
                )

            if train_config.save_every_epoch:
                save_checkpoint(
                    backend,
                    checkpoint_dir / f"epoch_{epoch + 1}.pt",
                    impl=impl,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                    model_config=model_config,
                    train_config=train_config,
                    data_config=data_config,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    text_logger=text_logger,
                )

        text_logger.info(
            f"Training finished successfully. best_val_loss={best_val_loss:.6f} | "
            f"checkpoints={checkpoint_dir.resolve()} | metrics={metrics_logger.metrics_path.resolve()}"
        )
    except Exception as exc:
        text_logger.error(f"Training failed: {exc}")
        text_logger.error(traceback.format_exc())
        raise
    finally:
        metrics_logger.close()
        text_logger.close()


if __name__ == "__main__":
    main()
