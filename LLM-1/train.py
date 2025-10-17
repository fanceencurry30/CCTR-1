import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from model import CharTransformer
from data_loader import get_dataloader
from config import Config
import torch.nn as nn
import os

def train(rank, world_size):
    """初始化 DDP 训练"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # 设备信息
    device = torch.device(f"cuda:{rank}")
    print(f"[进程 {rank}] 正在使用 {device} ({torch.cuda.get_device_name(rank)})")
    
    # 加载配置
    config = Config()
    config.device = device  # 设定设备
    dataloader, char2id = get_dataloader(config, rank, world_size)
    
    # 计算词表大小
    vocab_size = len(char2id)
    print(f"[进程 {rank}] 字典大小: {len(char2id)}")  # 打印字典大小


    # 取一个 batch 进行检查
    sample = next(iter(dataloader))
    inputs, targets = sample
    inputs, targets = inputs.to(device), targets.to(device)
    print(f"[进程 {rank}] Batch size: {inputs.shape} | Memory: {inputs.element_size() * inputs.nelement() / 1e6} MB")
    
    # 初始化模型
    model = CharTransformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len
    ).to(device)

    model = DDP(model, device_ids=[rank], output_device=rank)  # 使用 DDP

    # 定义优化器和损失函数（不忽略 <PAD>，因为 <PAD> 已被移除）
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()  # **去掉 ignore_index，确保计算全部字符**

    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"[进程 {rank}] Epoch {epoch} Loss: {total_loss / len(dataloader):.4f}")

        # 只在 rank 0 进程保存模型
        if rank == 0:
            torch.save(model.module.state_dict(), f"char_transformer_epoch{epoch}.pt")

    dist.destroy_process_group()  # 结束分布式进程组

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # **自动检测 GPU 数量**
    print(f"检测到 {world_size} 张 GPU，启动分布式训练...")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    
