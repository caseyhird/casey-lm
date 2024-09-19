from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

def gen_dataloaders(batch_size=32, max_length=128):
    dataset = load_dataset('glue', 'sst2')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_examples(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_examples, batched=True)

    # Used to create batches of tokens
    def data_collator(features):
        input_ids = [f['input_ids'] for f in features]
        return torch.tensor(input_ids)

    train_dataloader = DataLoader(tokenized_dataset['train'].shuffle(seed=42), batch_size=batch_size, collate_fn=data_collator)
    val_dataloader = DataLoader(tokenized_dataset['validation'].shuffle(seed=42), batch_size=batch_size, collate_fn=data_collator)
    
    return train_dataloader, val_dataloader, tokenizer.vocab_size