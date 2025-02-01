from jaxtyping import Float
from typing import List
from micrograd.engine import Value
import numpy as np

class LayerNorm:
    def __init__(self, d_model: int, epsilon: float = 1e-5):
        self.gamma = [Value(1) for _ in range(d_model)]
        self.beta = [Value(0) for _ in range(d_model)]
        self.epsilon = epsilon

    def __call__(self, x: Float[List, "batch seq_len d_model"]) -> Float[List, "batch seq_len d_model"]:
        def normalize(values: List[float]) -> List[float]:
            mean = np.mean(values)
            variance = np.var(values)
            std_norm = (x - mean) / np.sqrt(variance + self.epsilon)
            return self.gamma * std_norm + self.beta

        new_x: Float[List, "batch seq_len d_model"] = []
        for batch in x:
            new_batch = []
            for embedding in batch:
                new_emb = normalize(embedding)
                new_batch.append(new_emb)
            new_x.append(new_batch)
        return new_x