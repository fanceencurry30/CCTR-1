import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from model import TransformerLM
from data_loader import CharVocab, BlankFillingDataset
from config import *
from tqdm import tqdm


# ============ 在这里指定要使用的 GPU 列表 ============
GPU_LIST = [1, 2, 3, 5]  # 指定使用 0~5 号卡
WORLD_SIZE = len(GPU_LIST)


def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(GPU_LIST[rank])


def cleanup_distributed():
    """销毁分布式环境"""
    dist.destroy_process_group()


def train(rank, world_size):
    setup_distributed(rank, world_size)

    # 加载词汇表
    vocab = CharVocab(DICTIONARY_PATH)

    # 加载训练数据
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]

    dataset = BlankFillingDataset(texts, vocab)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # 初始化模型
    model = TransformerLM(
        vocab_size=vocab.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_queries=SPARSE_QUERIES,
        dropout=DROPOUT
    ).to(GPU_LIST[rank])

    model = DDP(model, device_ids=[GPU_LIST[rank]], output_device=GPU_LIST[rank])

    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # rank=0负责创建保存目录
    if rank == 0:
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0

        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)
        else:
            progress_bar = train_loader

        for batch in progress_bar:
            inputs = batch["input_ids"].to(GPU_LIST[rank], non_blocking=True)
            labels = batch["labels"].to(GPU_LIST[rank], non_blocking=True)
            mask_pos = batch["mask_positions"].to(GPU_LIST[rank], non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            mask_logits = logits[mask_pos.bool()]
            mask_labels = labels[mask_pos.bool()]
            loss = criterion(mask_logits, mask_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if rank == 0:
                progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        # 跨GPU同步平均 loss
        avg_loss_tensor = torch.tensor(total_loss / len(train_loader), device=GPU_LIST[rank])
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

        if rank == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")
            save_path = os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch + 1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, save_path)
            print(f"✅ 模型已保存到: {save_path}")

    cleanup_distributed()


if __name__ == "__main__":
    print(f"🚀 启动分布式训练，使用 GPU {GPU_LIST} 共 {WORLD_SIZE} 张卡")
    torch.multiprocessing.spawn(train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
