from micrograd.nn import MLP
from jaxtyping import Int, Float
from typing import List
from .self_attention import SelfAttention
from .layer_norm import LayerNorm

class Transformer:
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int):
        self.layers = [TransformerDecoderLayer(n_heads, num_layers) for _ in range(num_layers)]

    def __call__(self, x: Int[List, "batch_size sequence_length d_model"]) -> Float[List, "batch_size sequence_length d_model"]:
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoderLayer:
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.self_attention = SelfAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.mlp = MLP(d_model, [d_ff, d_model])
        self.norm2 = LayerNorm(d_model)


    def __call__(self, x: Int[List, "batch_size sequence_length"]) -> Float[List, "batch_size sequence_length vocab_size"]:
        # self attention
        x = self.self_attention(x)
        # norm
        # TODO
        # linear
        # relu
        # linear
        # norm
        pass