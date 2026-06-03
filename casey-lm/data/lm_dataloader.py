"""Language-modeling dataloaders (next-token prediction)."""

from dataclasses import dataclass
from typing import Literal

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DatasetName = Literal["wikitext", "shakespeare"]

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


@dataclass
class DataConfig:
    dataset: DatasetName = "wikitext"
    tokenizer_name: str = "gpt2"
    batch_size: int = 32
    block_size: int = 128
    num_workers: int = 0
    seed: int = 42


def _prepare_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_text_splits(config: DataConfig) -> tuple[Dataset, Dataset]:
    if config.dataset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train = dataset["train"].filter(lambda row: row["text"].strip())
        val = dataset["validation"].filter(lambda row: row["text"].strip())
        return train, val

    if config.dataset == "shakespeare":
        dataset = load_dataset("text", data_files={"all": SHAKESPEARE_URL}, split="all")
        split = dataset.train_test_split(test_size=0.1, seed=config.seed)
        return split["train"], split["test"]

    raise ValueError(f"Unknown dataset: {config.dataset}")


def _chunk_split(
    split: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
) -> Dataset:
    tokenized = split.map(
        lambda examples: {"input_ids": tokenizer(examples["text"])["input_ids"]},
        batched=True,
        remove_columns=split.column_names,
    )

    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_length = (len(concatenated) // block_size) * block_size
        chunks = [
            concatenated[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        return {"input_ids": chunks}

    return tokenized.map(group_texts, batched=True)


def get_dataloaders(
    config: DataConfig,
) -> tuple[DataLoader, DataLoader, int, PreTrainedTokenizerBase]:
    """Build train/val dataloaders for causal LM on fixed-length token blocks."""
    tokenizer = _prepare_tokenizer(config.tokenizer_name)
    train_split, val_split = _load_text_splits(config)

    train_dataset = _chunk_split(train_split, tokenizer, config.block_size)
    val_dataset = _chunk_split(val_split, tokenizer, config.block_size)

    def collate(features):
        input_ids = [f["input_ids"] for f in features]
        return torch.tensor(input_ids, dtype=torch.long)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "collate_fn": collate,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset.shuffle(seed=config.seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    return train_loader, val_loader, len(tokenizer), tokenizer
