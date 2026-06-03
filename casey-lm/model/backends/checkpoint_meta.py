"""Read implementation name from a checkpoint without loading full weights."""

from __future__ import annotations

import json
from pathlib import Path

from model.config import ImplName


def read_impl_from_run(run_dir: Path, checkpoint_path: Path) -> ImplName:
    config_path = run_dir / "config.json"
    if config_path.is_file():
        run_meta = json.loads(config_path.read_text())
        if "impl" in run_meta:
            return run_meta["impl"]

    return read_impl_from_checkpoint(checkpoint_path)


def read_impl_from_checkpoint(checkpoint_path: Path) -> ImplName:
    suffix = checkpoint_path.suffix.lower()
    if suffix == ".pt":
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return ckpt.get("impl", "torch")

    if suffix == ".bin":
        meta_path = checkpoint_path.with_suffix(".meta.json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            return meta.get("impl", "c")

    raise ValueError(
        f"Cannot determine backend for checkpoint {checkpoint_path}. "
        "Pass --impl explicitly or ensure config.json exists in the run directory."
    )
