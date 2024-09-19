from tinygrad import Tensor, nn
from dataclasses import dataclass
from jaxtyping import Float
from .self_attention import SelfAttention

@dataclass
class TransformerDecoderLayerConfig():
    d_model: int
    n_head: int
    dim_feedforward: int
    dropout_p: float

# TODO: other types of transformer layers
# This is decoder only right now. Not supporting encoders, or encoder-decoder
class TransformerDecoderLayer:
    def __init__(self, config: TransformerDecoderLayerConfig):
        super().__init__()
        self.self_attention = SelfAttention(config.d_model, config.n_head, config.dropout_p)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear1 = nn.Linear(config.dim_feedforward, config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
    def __call__(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        x = x + self.self_attention(x)
        x = self.norm1(x)
        x = x + self.linear1(x)
        x = x.relu() # TODO: support other activation functions
        x = x + self.linear2(x)
        x = self.norm2(x)
        return x

@dataclass
class TransformerConfig():
    num_decoder_layers: int
    decoder_layer_config: TransformerDecoderLayerConfig
    # TODO: can add other options avialable in pytorch

class Transformer:
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.decoder_layers = [TransformerDecoderLayer(config.decoder_layer_config) for _ in range(config.num_decoder_layers)]

    def __call__(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        for layer in self.decoder_layers:
            x = layer(x)
        return x



