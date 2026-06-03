# casey-lm

A small decoder-only language model with multiple backends (PyTorch, TinyGrad, and a C implementation). Use the training and generation scripts to train on WikiText or Shakespeare and sample text from a checkpoint.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The C backend (`--impl c`) additionally requires building the native library:

```bash
make -C casey-lm/model/c_impl
```

## Train a model

Run `casey-lm/scripts/train.py` from the repo root (or `cd casey-lm` and use `scripts/train.py` — paths below assume the repo root).

### Quick start

**PyTorch** (default; uses CUDA when available, else MPS on Apple Silicon, else CPU):

```bash
python casey-lm/scripts/train.py \
  --impl torch \
  --output-dir runs/my_torch \
  --epochs 5
```

**TinyGrad** (often faster on Mac with Metal):

```bash
python casey-lm/scripts/train.py \
  --impl tinygrad \
  --output-dir runs/my_tinygrad \
  --epochs 5
```

**Smoke test** (one epoch, few batches):

```bash
python casey-lm/scripts/train.py \
  --impl torch \
  --output-dir runs/smoke \
  --epochs 1 \
  --max-batches 10 \
  --max-val-batches 5
```

### What gets written

Each run directory (e.g. `runs/my_tinygrad/`) contains:

| Path | Description |
|------|-------------|
| `config.json` | Model, data, and training settings used for the run |
| `tokenizer/` | Saved tokenizer (needed for generation) |
| `checkpoints/last.pt` | Latest epoch |
| `checkpoints/best.pt` | Lowest validation loss |
| `checkpoints/epoch_N.pt` | Per-epoch snapshots (unless `--no-save-epoch-checkpoints`) |
| `logs/metrics.csv` | Step/epoch loss and accuracy |
| `logs/train.log` | Human-readable training log |

### Resume training

```bash
python casey-lm/scripts/train.py \
  --impl tinygrad \
  --output-dir runs/my_tinygrad \
  --resume runs/my_tinygrad/checkpoints/last.pt
```

Use the same `--impl` as when the checkpoint was created. The script restores model, optimizer, epoch, and step; learning rate is realigned to the schedule at the resumed step.

### Useful training flags

| Flag | Default | Notes |
|------|---------|--------|
| `--impl` | `torch` | `torch`, `tinygrad`, or `c` |
| `--output-dir` | `runs/train` | All artifacts for this run |
| `--dataset` | `wikitext` | `wikitext` or `shakespeare` |
| `--tokenizer` | `gpt2` | Hugging Face tokenizer name |
| `--block-size` | `128` | Context length (tokens per sequence) |
| `--batch-size` | `32` | |
| `--embedding-dim` | `128` | Must be divisible by `--num-heads` |
| `--num-decoder-layers` | `3` | |
| `--num-heads` | `4` | |
| `--dim-feedforward` | `512` | |
| `--lr` | `1e-3` | |
| `--lr-schedule` | `cosine_warmup` | `constant`, `cosine_warmup`, or `plateau` |
| `--epochs` | `5` | |
| `--device` | (auto) | e.g. `cuda`, `cpu`, `mps` (torch) or `METAL` / `CPU` (tinygrad) |
| `--max-batches` | (all) | Cap train batches per epoch (debug) |
| `--log-every` | `50` | Train loss rows in `metrics.csv` |

Example: smaller model on Shakespeare for a fast experiment:

```bash
python casey-lm/scripts/train.py \
  --impl torch \
  --output-dir runs/shakespeare_small \
  --dataset shakespeare \
  --embedding-dim 64 \
  --num-decoder-layers 2 \
  --num-heads 2 \
  --batch-size 16 \
  --block-size 64 \
  --epochs 10
```

## Generate text

After training, use `casey-lm/scripts/run.py` with a checkpoint. The script loads the run’s tokenizer from `--run-dir` (inferred from the checkpoint path when omitted) and picks the backend from the checkpoint unless you pass `--impl`.

```bash
python casey-lm/scripts/run.py \
  --checkpoint runs/my_tinygrad/checkpoints/best.pt \
  --prompt "The meaning of life is" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 40
```

Greedy decoding (deterministic):

```bash
python casey-lm/scripts/run.py \
  --checkpoint runs/my_tinygrad/checkpoints/best.pt \
  --prompt "Once upon a time" \
  --temperature 0
```

Reproducible sampling:

```bash
python casey-lm/scripts/run.py \
  --checkpoint runs/my_tinygrad/checkpoints/best.pt \
  --prompt "Hello" \
  --seed 42
```

| Flag | Default | Notes |
|------|---------|--------|
| `--checkpoint` | `runs/train/checkpoints/best.pt` | Path to `.pt` from training |
| `--run-dir` | parent of `checkpoints/` | Holds `config.json` and `tokenizer/` |
| `--impl` | from checkpoint | Override only if needed |
| `--max-new-tokens` | `100` | Tokens generated after the prompt |
| `--temperature` | `0.8` | `0` = argmax |
| `--top-k` | `40` | `0` disables top-k filtering |

## Notebooks (Colab)

To run a notebook in Google Colab, open it on GitHub and change the URL from `github.com` to `githubtocolab.com` ([instructions](https://stackoverflow.com/questions/62596466/how-can-i-run-notebooks-of-a-github-project-in-google-colab)). See `casey-lm/notebooks/` for Colab-oriented training examples.

## Roadmap

- More hyperparameter and architecture options (attention type, norm, activations, optimizers, etc.)
- Additional backends and cross-framework abstractions
- Edge optimization, quantization, distributed training
