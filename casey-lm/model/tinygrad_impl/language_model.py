from tinygrad import Tensor, nn
from dataclasses import dataclass
from .transformer import Transformer, TransformerConfig
from .mlp import MLP, MLPConfig
from jaxtyping import Float

@dataclass
class TinygradLanguageModelConfig():
    vocab_size: int
    context_length: int
    embedding_dim: int
    transformer_config: TransformerConfig
    mlp_config: MLPConfig

class TinygradLanguageModel:
    def __init__(self, config: TinygradLanguageModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.transformer = Transformer(config.transformer_config)
        self.mlp = MLP(config.mlp_config)
        self.unembedding = nn.Linear(config.embedding_dim, config.vocab_size)

    def __call__(self, x: Float[Tensor, "batch sequence_length"]) -> Float[Tensor, "batch sequence_length vocab_size"]:
        x: Float[Tensor, "batch sequence_length embedding_dim"] = self.token_embedding(x) + self.position_embedding(x)
        x: Float[Tensor, "batch sequence_length embedding_dim"] = self.transformer(x)
        x: Float[Tensor, "batch sequence_length embedding_dim"] = self.mlp(x)
        x: Float[Tensor, "batch sequence_length vocab_size"] = self.unembedding(x)
        return x