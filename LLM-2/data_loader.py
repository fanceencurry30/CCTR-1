import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from config import *
from tqdm import tqdm


class CharVocab:
    def __init__(self, dictionary_path):
        # 加载字典并初始化词汇表
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        # 添加特殊token
        self.char_to_idx.update({PAD_TOKEN: 0, MASK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3})
        self.idx_to_char.update({0: PAD_TOKEN, 1: MASK_TOKEN, 2: SOS_TOKEN, 3: EOS_TOKEN})

    def encode(self, text):
        # 将文本编码为token ID序列
        return [self.char_to_idx.get(char, self.char_to_idx[MASK_TOKEN]) for char in text]

    def decode(self, tokens):
        # 将token ID解码为文本
        return ''.join([self.idx_to_char.get(token, MASK_TOKEN) for token in tokens])


class BlankFillingDataset(Dataset):
    def __init__(self, texts, vocab, groups=5, max_len=MAX_SEQ_LEN):
        # 初始化数据集
        self.vocab = vocab
        self.groups = groups
        self.max_len = max_len
        # 直接存储原始文本段落，避免在初始化时进行任何重度预处理
        print("正在初始化数据集...")
        self.segments = [seg.strip() for seg in texts if seg.strip()]
        print(f"数据集初始化完成，共加载 {len(self.segments)} 条原始数据。")

    def __len__(self):
        # 数据集总大小是 原始段落数 * 每段生成的样本组数
        return len(self.segments) * self.groups

    def __getitem__(self, idx):
        # --- 在这里进行即时样本生成 ---

        # 1. 根据索引idx计算出它对应哪个原始段落和哪个处理组
        segment_idx = idx // self.groups
        group_idx = idx % self.groups

        # 2. 获取原始段落并进行编码
        segment = self.segments[segment_idx]
        tokens = [self.vocab.char_to_idx[SOS_TOKEN]] + self.vocab.encode(segment) + [
            self.vocab.char_to_idx[EOS_TOKEN]]

        # 3. 填充或截断到最大长度
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [self.vocab.char_to_idx[PAD_TOKEN]] * (self.max_len - len(tokens))

        # 4. 执行与之前相同的“完形填空”和“加噪声”逻辑
        # 排除SOS、EOS和PAD的掩码位置
        valid_positions = [i for i, t in enumerate(tokens) if t not in [self.vocab.char_to_idx[SOS_TOKEN],
                                                                            self.vocab.char_to_idx[EOS_TOKEN],
                                                                            self.vocab.char_to_idx[PAD_TOKEN]]]

        masked_tokens = tokens.copy()
        mask_positions = [0] * len(tokens)
        mask_labels = [-100] * len(tokens)

        # 如果没有有效位置可以掩码，则直接返回
        if not valid_positions:
            return {
                "input_ids": torch.tensor(masked_tokens),
                "labels": torch.tensor(mask_labels),
                "mask_positions": torch.tensor(mask_positions)
            }
        
        # 根据组别确定掩码比例
        num_masks = max(1, int(len(valid_positions) * (group_idx + 1) / self.groups))
        mask_indices = random.sample(valid_positions, min(num_masks, len(valid_positions)))

        # 应用掩码
        for pos in mask_indices:
            masked_tokens[pos] = self.vocab.char_to_idx[MASK_TOKEN]
            mask_labels[pos] = tokens[pos]
            mask_positions[pos] = 1

        # 添加噪声
        noise_indices = [i for i in valid_positions if i not in mask_indices]
        if noise_indices:
            noise_count = max(1, int(len(valid_positions) * NOISE_P))
            noise_positions = random.sample(noise_indices, min(noise_count, len(noise_indices)))
            for pos in noise_positions:
                masked_tokens[pos] = random.randint(4, self.vocab.vocab_size - 1)

        # 5. 将处理好的数据转换为Tensor并返回
        return {
            "input_ids": torch.tensor(masked_tokens),
            "labels": torch.tensor(mask_labels),
            "mask_positions": torch.tensor(mask_positions)
        }