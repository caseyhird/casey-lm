from dataclasses import dataclass
from jaxtyping import Int, Float
from typing import List
from .embedding import Embedding

@dataclass
class MicrogradLanguageModelConfig:
    vocab_size: int
    context_length: int
    embedding_dim: int
    transformer_config: None #TransformerConfig
    mlp_config: None #MLPConfig

class MicrogradLanguageModel:
    def __init__(self, config: MicrogradLanguageModelConfig):
        # token embedding
        self.token_embedding = Embedding(config.vocab_size, config.embedding_dim)
        # position embedding
        self.position_embedding = Embedding(config.context_length, config.embedding_dim)
        # transformer
        # mlp
        # unembedding (linear layer)
        pass

    def __call__(self, x: Int[List, "batch_size sequence_length"]) -> Float[List, "batch_size sequence_length vocab_size"]:
        pass