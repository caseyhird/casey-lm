from __future__ import annotations

import math

from jaxtyping import Float
from tinygrad import Tensor, nn


class SelfAttention:
    def __init__(self, d_model: int, n_head: int, dropout_p: float):
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        d_head = d_model // n_head
        self.attention_heads = [
            AttentionHead(d_model, d_head, dropout_p) for _ in range(n_head)
        ]
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_head

    def __call__(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        head_outs = [h(x, x, x) for h in self.attention_heads]
        merged = Tensor.cat(*head_outs, dim=-1)
        return self.out_proj(merged)


class AttentionHead:
    def __init__(self, d_in: int, d_head: int, dropout_p: float):
        self.wq = nn.Linear(d_in, d_head)
        self.wk = nn.Linear(d_in, d_head)
        self.wv = nn.Linear(d_in, d_head)
        self.dropout_p = dropout_p
        self.scale = 1.0 / math.sqrt(d_head)

    def __call__(
        self,
        x_q: Float[Tensor, "batch seq_len d_in"],
        x_k: Float[Tensor, "batch seq_len d_in"],
        x_v: Float[Tensor, "batch seq_len d_in"],
    ) -> Float[Tensor, "batch seq_len d_head"]:
        q = self.wq(x_q)
        k = self.wk(x_k)
        v = self.wv(x_v)

        scores = q @ k.transpose(-2, -1) * self.scale
        scores = _apply_causal_mask(scores)
        attn = scores.softmax(axis=-1)
        if Tensor.training:
            attn = attn.dropout(self.dropout_p)
        return attn @ v


def _apply_causal_mask(scores: Tensor) -> Tensor:
    """Mask future positions with a large negative value before softmax."""
    seq_len = scores.shape[-1]
    mask = Tensor.triu(Tensor.ones(seq_len, seq_len), diagonal=1)
    return scores + mask * -1e9
