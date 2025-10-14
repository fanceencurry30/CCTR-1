# -*- coding: utf-8 -*-
"""
Load data, build vocabulary, create DataLoader
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
from torch.utils.data import DistributedSampler

class CharDataset(Dataset):
    def __init__(self, file_path, char2id, max_seq_len):
        self.char2id = char2id
        self.max_seq_len = max_seq_len
        with open(file_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx].strip()
        # Only keep characters that exist in the vocabulary, skip those not in the dictionary
        input_ids = [self.char2id[c] for c in text[:self.max_seq_len] if c in self.char2id]

        # If filtered result is empty, skip this sample
        if len(input_ids) == 0:
            return None  # Skip this sample

        # Target token processing, shift right by one position
        target_ids = input_ids[1:]  # Direct right shift, no padding with 0

        # If target_ids is empty, skip this sample
        if len(target_ids) == 0:
            return None  

        return torch.tensor(input_ids[:-1]), torch.tensor(target_ids)  # Ensure input_ids and target_ids have matching lengths


# **Correction: collate_fn needs to be a global function, not a class method**
def collate_fn(batch):
    # Filter out None values
    batch = [b for b in batch if b is not None]

    # If all samples are filtered out, return default padded batch
    if len(batch) == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)  # Return empty tensor

    # Stack samples in batch into a tensor
    input_ids, target_ids = zip(*batch)
    return torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0), \
           torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=0)


def load_vocab(vocab_path):
    """Load vocabulary without adding special tokens"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        char2id = json.load(f)
    return char2id  # Return directly, ensuring no <pad> token


def get_dataloader(config, rank, world_size):
    """Create DataLoader that supports multi-GPU training"""
    char2id = load_vocab(config.vocab_path)
    dataset = CharDataset(config.data_dir, char2id, config.max_seq_len)

    # Key: Use DistributedSampler to let each GPU only process its own portion of data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # Create DataLoader and pass in collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn  # Correction: pass in globally defined collate_fn
    )
    return dataloader, char2id
