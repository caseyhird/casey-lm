# casey-lm

development:
- to run in colab, open an ipynb notebook in github, then change the url from 'github.com' to 'githubtocolab.com' (as instructed [here](https://stackoverflow.com/questions/62596466/how-can-i-run-notebooks-of-a-github-project-in-google-colab))

todo:
- add readme details
- allow options for tokenization method, attention type, norm type, activation function, positional encoding, dataset, optimizer, scheduler, and other hyperparameters
- other impls: tinygrad, scratch, lua torch, julia mlj, julia flux
- Implement other attention types
    - types of attention based on whether q, k, v are the same or different, e.g. self attention (q=k=v) or cross attention (q!=k=v)
    - types of attention based on mask, e.g. causal attention (mask future tokens) or full attention (no mask)
    - types of attention based on how multiple heads are computed and combined, e.g. split embeddings before computing attention or use full embedding as input for each head
    - have head outputs concatenate or sum to get final output
    - methodso of computation, e.g. check if multiple heads in a single tensor improves performance
- Consider cross-fw abstractions, e.g. between self-attention in tinygrad and micrograd
- More sophisticated architecture (maybe)
- Edge optimized version
- Pruning, quantization, etc.
- Distributed training
- Implement autograd, other parts of training from scratch as well
