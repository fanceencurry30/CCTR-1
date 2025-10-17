# -*- coding: utf-8 -*-
"""
加载数据、构建字典、创建DataLoader
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
        # 仅保留字典里有的字符，跳过那些不在字典中的字符
        input_ids = [self.char2id[c] for c in text[:self.max_seq_len] if c in self.char2id]

        # 如果过滤后为空，跳过该样本
        if len(input_ids) == 0:
            return None  # 跳过该样本

        # 目标 Token 处理，右移一位
        target_ids = input_ids[1:]  # 直接右移，不填充 0

        # 如果 target_ids 为空，则跳过该样本
        if len(target_ids) == 0:
            return None  

        return torch.tensor(input_ids[:-1]), torch.tensor(target_ids)  # 保证 input_ids 和 target_ids 长度匹配


# **修正：collate_fn 需要是一个全局函数，而不是类方法**
def collate_fn(batch):
    # 过滤掉 None 值
    batch = [b for b in batch if b is not None]

    # 如果所有样本都被过滤掉了，返回默认填充的 batch
    if len(batch) == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)  # 返回空 tensor

    # 将 batch 中的样本堆叠成一个 tensor
    input_ids, target_ids = zip(*batch)
    return torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0), \
           torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=0)


def load_vocab(vocab_path):
    """加载字典，不额外添加特殊标记"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        char2id = json.load(f)
    return char2id  # 直接返回，确保没有 <pad>


def get_dataloader(config, rank, world_size):
    """创建支持多 GPU 训练的 DataLoader"""
    char2id = load_vocab(config.vocab_path)
    dataset = CharDataset(config.data_dir, char2id, config.max_seq_len)

    # 关键：使用 DistributedSampler，让每个 GPU 只处理自己的一部分数据
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # 创建 DataLoader，并传入 collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn  # 修正：传入全局定义的 collate_fn
    )
    return dataloader, char2id
