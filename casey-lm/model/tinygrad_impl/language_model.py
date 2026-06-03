from __future__ import annotations

from jaxtyping import Float
from tinygrad import Tensor, nn

from model.config import LanguageModelConfig
from model.tinygrad_impl.transformer import (
    Transformer,
    TransformerConfig,
    TransformerDecoderLayerConfig,
)


class TinygradLanguageModel:
    def __init__(self, config: LanguageModelConfig):
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(
            config.context_length, config.embedding_dim
        )
        self.transformer = Transformer(
            TransformerConfig(
                num_decoder_layers=config.num_decoder_layers,
                decoder_layer_config=TransformerDecoderLayerConfig(
                    d_model=config.embedding_dim,
                    n_head=config.num_heads,
                    dim_feedforward=config.dim_feedforward,
                    dropout_p=config.dropout,
                ),
            )
        )
        self.unembedding = nn.Linear(config.embedding_dim, config.vocab_size)

    def __call__(
        self, x: Float[Tensor, "batch sequence_length"]
    ) -> Float[Tensor, "batch sequence_length vocab_size"]:
        seq_len = x.shape[1]
        positions = Tensor.arange(seq_len)
        pos_emb = self.position_embedding(positions)
        token_emb = self.token_embedding(x)
        hidden = token_emb + pos_emb
        hidden = self.transformer(hidden)
        return self.unembedding(hidden)
