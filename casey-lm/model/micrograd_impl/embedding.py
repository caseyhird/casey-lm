from micrograd.nn import Layer
from jaxtyping import Int, Float
from typing import List

class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.layer = Layer(num_embeddings, embedding_dim)

    def forward(self, x: Int[List, "batch_size sequence_length"]) -> Float[List, "batch_size sequence_length embedding_dim"]:
        def one_hot_encode(x: Int[List, "sequence_length"]) -> Int[List, "sequence_length num_embeddings"]:
            return [1 if i == x[j] else 0 for i in range(self.num_embeddings) for j in range(len(x))]
        one_hot = one_hot_encode(x)
        return self.layer(one_hot)