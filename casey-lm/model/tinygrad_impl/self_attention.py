from tinygrad import Tensor, nn
from jaxtyping import Float

# TODO: types of attention based on wehther q, k, v are the same or different
# e.g. self attention (q=k=v) or cross attention (q!=k=v)
# TODO: types of attention based on mask
# e.g. causal attention (mask future tokens) or full attention (no mask)
# TODO: types of attention based on how multiple heads are computed and combined
# e.g. split embeddings before computing attention or use full embedding as input for each head
# have head outputs concatenate or sum to get final output
# TODO: try multiple heads in a single tensor and see if that affects performance
class SelfAttention:
    def __init__(self, d_model: int, n_head: int, dropout_p: float):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        d_head = d_model // n_head
        self.attention_heads = [AttentionHead(d_model, d_head, dropout_p) for _ in range(n_head)]
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

    def __call__(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        o: Float[Tensor, "batch seq_len d_model"] = Tensor.cat([h(x) for h in self.attention_heads], dim=-1)
        return o
        

class AttentionHead:
    def __init__(self, d_in: int, d_out: int, dropout_p: float):
        super().__init__()
        self.wq = nn.Linear(d_in, d_out)
        self.wk = nn.Linear(d_in, d_out)
        self.wv = nn.Linear(d_in, d_out)
        self.dropout_p = dropout_p

    def __call__(
            self, 
            x_q: Float[Tensor, "batch seq_len d_in"], 
            x_k: Float[Tensor, "batch seq_len d_in"], 
            x_v: Float[Tensor, "batch seq_len d_out"],
            ) -> Float[Tensor, "batch seq_len d_out"]:
        q: Float[Tensor, "batch seq_len seq_len"] = self.wq(x_q)
        k: Float[Tensor, "batch seq_len seq_len"] = self.wk(x_k)
        v: Float[Tensor, "batch seq_len d_out"] = self.wv(x_v)

        qk: Float[Tensor, "batch seq_len seq_len"] = Tensor.einsum("b s d_in, b s d_in -> b s s", q, k)
        a: Float[Tensor, "batch seq_len seq_len"] = qk.softmax(dim=1)
        a = a.dropout(self.dropout_p)
        a = a.tril()

        o: Float[Tensor, "batch seq_len d_out"] = Tensor.einsum("b s s, b s d_out -> b s d_out", a, v)
        return o
        