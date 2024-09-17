from torch import nn, arange, Tensor, zeros_like, triu, ones
from jaxtyping import Int, Float
from dataclasses import dataclass
@dataclass
class LanguageModelConfig():
    vocab_size: int
    context_length: int
    embedding_dim: int
    num_decoder_layers: int
    num_heads: int
    dim_feedforward: int
    dropout: float

class TorchLanguageModel(nn.Module):
    def __init__(
            self,
            config: LanguageModelConfig
        ):
        super().__init__()
        # position embedding
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        # token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        # alternate transformer and mlp layers
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads
            ),
            num_layers=config.num_decoder_layers,
            norm=nn.LayerNorm(config.embedding_dim)
        )
        # unembedding layer
        self.unembedding = nn.Linear(config.embedding_dim, config.vocab_size)


    def forward(self, x: Int[Tensor, "batch_size sequence_length"]) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        # Embed position values for each position in the sequence
        pos_embeddings: Float[Tensor, "sequence_length embedding_dim"] = self.position_embedding(arange(0, x.size(1)))
        # Embed the token values for each token in the sequence
        token_embeddings: Float[Tensor, "batch_size sequence_length embedding_dim"] = self.token_embedding(x)
        # Combine the position and token embeddings
        combined_embeddings: Float[Tensor, "batch_size sequence_length embedding_dim"] = pos_embeddings + token_embeddings
        # Pass the combined embeddings through the transformer
        # Note: decoder requires "memory" from the last layer of the encoder. We want a decoder only model so
        # we pass in a dummy tensor of zeros that has the same shape as the combined embeddings.
        dummy_memory = zeros_like(combined_embeddings)
        seq_len = combined_embeddings.shape[1]
        # TODO: generate_square_subsequent_mask ?
        causal_mask = triu(ones(seq_len, seq_len), diagonal=1).bool()
        # Run decoder with causal mask
        transformer_output: Float[Tensor, "batch_size sequence_length embedding_dim"] = self.transformer(
            combined_embeddings,
            dummy_memory,
            tgt_is_causal=True,
            tgt_mask=causal_mask
        )
        # Pass the transformer output through the unembedding layer
        unembedded_output: Float[Tensor, "batch_size sequence_length vocab_size"] = self.unembedding(transformer_output)
        return unembedded_output