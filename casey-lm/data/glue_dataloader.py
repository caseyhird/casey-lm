"""Backward-compatible wrapper around lm_dataloader."""

from .lm_dataloader import DataConfig, get_dataloaders


def gen_dataloaders(batch_size=32, max_length=128, dataset="wikitext"):
    config = DataConfig(batch_size=batch_size, block_size=max_length, dataset=dataset)
    train_loader, val_loader, vocab_size, _tokenizer = get_dataloaders(config)
    return train_loader, val_loader, vocab_size
