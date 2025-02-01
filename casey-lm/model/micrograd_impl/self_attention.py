from jaxtyping import Float
from typing import List
from micrograd.nn import Layer
import numpy as np

class SelfAttention:
    def __init__(self, d_model: int, n_head: int):
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        d_head = d_model // n_head
        self.attention_heads = [AttentionHead(d_model, d_head) for _ in range(n_head)]
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

    def __call__(self, x: Float[List, "batch seq_len d_model"]) -> Float[List, "batch seq_len d_model"]:
        out: Float[List, "n_head batch seq_len d_out"] = [h(x) for h in self.attention_heads]
        result: Float[List, "batch seq_len d_model"] = []
        for i in range(len(out)):
            new_element = [item for sublist in out[i] for item in sublist]
            result.append(new_element)
        return result
        

class AttentionHead:
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.wq = Layer(d_in, d_out)
        self.wk = Layer(d_in, d_out)
        self.wv = Layer(d_in, d_out)
        # TODO add dropout

    def __call__(
            self, 
            x_q: Float[List, "batch seq_len d_in"], 
            x_k: Float[List, "batch seq_len d_in"], 
            x_v: Float[List, "batch seq_len d_out"],
            ) -> Float[List, "batch seq_len d_out"]:
        q: Float[List, "batch seq_len seq_len"] = self.wq(x_q)
        k: Float[List, "batch seq_len seq_len"] = self.wk(x_k)
        v: Float[List, "batch seq_len d_out"] = self.wv(x_v)

        qk: Float[List, "batch seq_len seq_len"] = np.einsum("b s d_in, b s d_in -> b s s", q, k)

        def softmax(values: List[float]) -> List[float]:
            logits = np.log(np.exp(values) / sum(np.exp(values), axis=0))
            return np.exp(logits)

        a: Float[List, "batch seq_len seq_len"] = []
        for batch_qk in qk:
            batch_a = []
            for row_qk in batch_qk:
                sm = softmax(row_qk)
                batch_a.append(sm)
            a.append(batch_a)
        a: Float[List, "batch seq_len seq_len"] = qk.softmax(dim=1)

        def lower_tril(values: List[List[float]]):
            return [[0 if j > i else val for j, val in enumerate(row)] for i, row in enumerate(values)]
        a = [lower_tril(batch) for batch in a]

        o: Float[List, "batch seq_len d_out"] = []
        for batch_a, batch_v in zip(a,v):
            o.append(np.array(batch_a) @ np.array(batch_v))
        return o
        