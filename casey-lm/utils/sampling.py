"""Framework-agnostic text generation helpers."""

from __future__ import annotations

import numpy as np


def sample_next_token(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one token from a logits vector of shape [vocab_size]."""
    if temperature <= 0:
        return int(np.argmax(logits))

    if rng is None:
        rng = np.random.default_rng()

    scaled = logits / temperature
    if top_k > 0:
        k = min(top_k, scaled.shape[-1])
        threshold = np.partition(scaled, -k)[-k]
        scaled = np.where(scaled < threshold, float("-inf"), scaled)

    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def generate(
    backend,
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    max_context: int,
    temperature: float,
    top_k: int,
    device,
    rng: np.random.Generator | None = None,
) -> str:
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    if not encoded:
        encoded = [tokenizer.bos_token_id or tokenizer.eos_token_id or 0]

    tokens = list(encoded)
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        context = tokens[-max_context:]
        logits = backend.forward_logits(model, context, device)
        next_logits = logits[-1]
        next_id = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        tokens.append(next_id)

        if eos_id is not None and next_id == eos_id:
            break

    return tokenizer.decode(tokens, skip_special_tokens=True)
