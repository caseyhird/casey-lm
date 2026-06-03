#!/usr/bin/env python3
"""Generate text with a trained Casey LM checkpoint (any backend)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from model.backends import get_backend, registered_impls
from model.backends.checkpoint_meta import read_impl_from_run
from model.config import ImplName
from utils.sampling import generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained Casey LM checkpoint")
    parser.add_argument(
        "--impl",
        choices=registered_impls(),
        default=None,
        help="Backend (default: read from checkpoint or config.json)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/train/checkpoints/best.pt"),
        help="Path to a checkpoint from train.py",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory (config.json, tokenizer/). Defaults to parent of checkpoints/.",
    )
    parser.add_argument("--prompt", type=str, default="The", help="Text prompt to continue")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Tokens to generate after the prompt")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 = greedy argmax)",
    )
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--device", type=str, default=None, help="Device (backend-specific)")
    return parser.parse_args()


def resolve_run_dir(checkpoint: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def load_tokenizer(run_dir: Path, tokenizer_name: str) -> PreTrainedTokenizerBase:
    saved = run_dir / "tokenizer"
    if saved.is_dir():
        return AutoTokenizer.from_pretrained(saved)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def resolve_impl(
    args_impl: str | None,
    checkpoint_path: Path,
    run_dir: Path,
) -> ImplName:
    if args_impl is not None:
        return args_impl  # type: ignore[return-value]
    return read_impl_from_run(run_dir, checkpoint_path)


def main() -> None:
    args = parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    run_dir = resolve_run_dir(checkpoint, args.run_dir.resolve() if args.run_dir else None)
    impl = resolve_impl(args.impl, checkpoint, run_dir)
    backend = get_backend(impl)
    device = backend.resolve_device(args.device)

    model, _, ckpt_state = backend.load_checkpoint(checkpoint, device)
    model_config = ckpt_state.model_config
    data_config = ckpt_state.data_config
    tokenizer_name = data_config.tokenizer_name
    tokenizer = load_tokenizer(run_dir, tokenizer_name)

    rng = None
    if args.seed is not None:
        backend.set_seed(args.seed)
        rng = np.random.default_rng(args.seed)

    print(f"Checkpoint: {checkpoint}")
    print(f"Backend:    {impl}")
    print(f"Run dir:    {run_dir}")
    print(f"Device:     {device}")
    print(f"Context:    {model_config.context_length} tokens")
    if ckpt_state.val_loss < float("inf"):
        print(f"Val loss:   {ckpt_state.val_loss:.4f} (epoch {ckpt_state.epoch})")
    print()
    print(f"Prompt: {args.prompt!r}")
    print("-" * 60)

    output = generate(
        backend,
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_context=model_config.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        rng=rng,
    )
    print(output)
    print("-" * 60)


if __name__ == "__main__":
    main()
