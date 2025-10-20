# model.py
# 定义基于Transformer的字符填空语言模型

import torch
import torch.nn as nn
from config import *

class SparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries):
        # 初始化可学习稀疏注意力模块
        super().__init__()
        self.embed_dim = int(embed_dim)  # 确保embed_dim是整数
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = self.embed_dim // num_heads

        # 可学习的查询向量
        self.query_vectors = nn.Parameter(torch.randn(1, num_queries, self.embed_dim))
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale = self.head_dim ** -0.5
        # 将输出线性层移到初始化函数中
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        # 稀疏注意力前向传播
        batch_size, seq_len, _ = x.size()
        queries = self.query_vectors.expand(batch_size, -1, -1)  # [batch_size, num_queries, embed_dim]
        keys = self.key(x)  # [batch_size, seq_len, embed_dim]
        values = self.value(x)  # [batch_size, seq_len, embed_dim]

        # 重塑为多头形式
        queries = queries.view(batch_size, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, values)  # [batch_size, num_heads, num_queries, head_dim]

        # 重塑回原始维度
        out = out.transpose(1, 2).contiguous().view(batch_size, self.num_queries, self.embed_dim)
        # 通过在初始化时定义的线性层处理
        out = self.output_proj(out)
        # 在 num_queries 维度上取平均，将其压缩成一个全局上下文向量
        out = out.mean(dim=1, keepdim=True)
        # 将全局上下文向量扩展回原始序列长度
        out = out.expand(-1, seq_len, -1)
        return out

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 num_heads=NUM_HEADS, num_queries=SPARSE_QUERIES, dropout=DROPOUT):
        # 初始化Transformer模型
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 词嵌入层
        self.positional_encoding = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, embed_dim))  # 位置编码

        # Transformer编码器层
        self.sparse_attention = nn.ModuleList([
            SparseAttention(embed_dim, num_heads, num_queries) for _ in range(num_layers)
        ])
        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        self.layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        # 输出层
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # 前向传播
        embeds = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]  # 加入位置编码

        # Transformer编码
        for attn, ff, norm in zip(self.sparse_attention, self.feed_forward, self.layer_norm):
            attn_out = attn(embeds)  # 稀疏注意力
            embeds = norm(embeds + attn_out)  # 残差连接与层归一化
            ff_out = ff(embeds)  # 前馈网络
            embeds = norm(embeds + ff_out)  # 残差连接与层归一化

        logits = self.output(embeds)  # 输出logits
        return logits