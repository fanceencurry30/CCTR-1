# -*- coding: utf-8 -*-
"""
配置训练超参数、数据路径、模型参数
"""

class Config:
    # 数据路径
    data_dir = "/home/u2024000980/sanliwan LLM/data/Deepseeksanliwanfantienglish.txt"  # 训练文本路径
    vocab_path = "/home/u2024000980/sanliwan LLM/data/merged_char_to_idx.json"        # 字典文件路径
    
    # 模型参数
    d_model = 1024          # 模型维度
    num_layers = 8          # Transformer层数
    num_heads = 8           # 注意力头数
    d_ff = 4096             # 前馈网络维度
    max_seq_len = 512       # 序列最大长度
    dropout = 0.1           # Dropout概率
    
    # 训练参数
    batch_size = 64       # 批次大小
    learning_rate = 1e-4    # 学习率
    num_epochs = 10         # 训练轮次
    device = "cuda"         # 训练设备（"cuda" 或 "cpu"）
    
config = Config()