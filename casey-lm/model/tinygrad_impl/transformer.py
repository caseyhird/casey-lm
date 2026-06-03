from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float
from tinygrad import Tensor, nn

from .self_attention import SelfAttention


@dataclass
class TransformerDecoderLayerConfig:
    d_model: int
    n_head: int
    dim_feedforward: int
    dropout_p: float


class TransformerDecoderLayer:
    def __init__(self, config: TransformerDecoderLayerConfig):
        self.self_attention = SelfAttention(
            config.d_model, config.n_head, config.dropout_p
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout_p = config.dropout_p

    def __call__(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        x = x + self.self_attention(self.norm1(x))
        normed = self.norm2(x)
        ff = self.linear1(normed).relu()
        if Tensor.training:
            ff = ff.dropout(self.dropout_p)
        x = x + self.linear2(ff)
        return x


@dataclass
class TransformerConfig:
    num_decoder_layers: int
    decoder_layer_config: TransformerDecoderLayerConfig


class Transformer:
    def __init__(self, config: TransformerConfig):
        self.decoder_layers = [
            TransformerDecoderLayer(config.decoder_layer_config)
            for _ in range(config.num_decoder_layers)
        ]
        self.final_norm = nn.LayerNorm(config.decoder_layer_config.d_model)

    def __call__(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        for layer in self.decoder_layers:
            x = layer(x)
        return self.final_norm(x)
