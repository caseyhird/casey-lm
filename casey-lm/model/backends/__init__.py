"""Backend registry."""

from __future__ import annotations

from model.backends.base import Backend
from model.backends.c_backend import CBackend
from model.backends.tinygrad_backend import TinygradBackend
from model.backends.torch_backend import TorchBackend
from model.config import ImplName

_BACKENDS: dict[ImplName, Backend] = {
    "torch": TorchBackend(),
    "tinygrad": TinygradBackend(),
    "c": CBackend(),
}


def get_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        valid = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"Unknown implementation {name!r}. Choose from: {valid}")
    return _BACKENDS[name]  # type: ignore[return-value]


def registered_impls() -> list[str]:
    return sorted(_BACKENDS)
