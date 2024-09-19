from tinygrad import Tensor, nn
from dataclasses import dataclass
from jaxtyping import Float


@dataclass
class MLPConfig():
    d_model: int
    d_hidden: int
    num_layers: int
    dropout_p: float
    # TODO: support other activation functions

class MLP:
    def __init__(self, config: MLPConfig):
        assert config.num_layers >= 2, "MLP num_layers must be at least 2"
        self.linear_first = nn.Linear(config.d_model, config.d_hidden)
        self.inner_layers = [nn.Linear(config.d_hidden, config.d_hidden) for _ in range(config.num_layers - 2)]
        self.linear_last = nn.Linear(config.d_hidden, config.d_model)
        self.dropout_p = config.dropout_p

    def __call__(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        x = self.linear_first(x).relu()
        for layer in self.inner_layers:
            x = layer(x).relu() # TODO: support other activation functions
            x = x.dropout(self.dropout_p)
        x = x.relu() 
        return self.linear2(x)