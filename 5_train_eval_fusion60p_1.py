import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta

from fusion_model_60p import CrossAttentionFusion
from config_fusion import FEATURE_DIM, NUM_HEADS, HIDDEN_DIM, DROPOUT, BATCH_SIZE, EPOCHS, LEARNING_RATE, FUSION_MODEL_SAVE_PATH

class PreprocessedDataset(Dataset):
    def __init__(self, pt_file_path):
        print(f"Loading preprocessed data from '{pt_file_path}'...")
        self.data = torch.load(pt_file_path, map_location='cpu')
        self.ocr_c100 = self.data['ocr_c100']
        self.lm_c100 = self.data['lm_c100']
        self.gt_c100 = self.data['gt_c100']
        self.total_len = self.ocr_c100.shape[0]
        print(f"Data loading completed, contains {self.total_len} samples.")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        return (
            self.ocr_c100[idx],
            self.lm_c100[idx],
            self.gt_c100[idx]
        )

def setup(rank, world_size, port='12368'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    timeout = timedelta(minutes=30)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)

def cleanup():
    dist.destroy_process_group()

def train_epoch(model, train_loader, optimizer, criterion, device, rank):
    model.train()
    total_loss = 0
    train_iterator = tqdm(train_loader, desc=f"Training Progress (Rank {rank})", disable=(rank != 0))
    for data in train_iterator:
        ocr_c100, lm_c100, gt_c100 = data
        ocr_c100 = ocr_c100.to(device)
        lm_c100 = lm_c100.to(device)
        gt_c100 = gt_c100.to(device)
        optimizer.zero_grad()
        output = model(ocr_c100, lm_c100)
        loss_ce = criterion['ce'](output, gt_c100.squeeze(1).argmax(dim=-1))
        loss = loss_ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device, rank):
    model.eval()
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        val_iterator = tqdm(val_loader, desc=f"Validation Progress (Rank {rank})", disable=(rank != 0))
        for data in val_iterator:
            ocr_c100, lm_c100, gt_c100 = data
            ocr_c100 = ocr_c100.to(device)
            lm_c100 = lm_c100.to(device)
            gt_c100 = gt_c100.to(device)
            output = model(ocr_c100, lm_c100)
            pred_idx = output.argmax(dim=-1)
            gt_idx = gt_c100.squeeze(1).argmax(dim=-1)
            total_correct += (pred_idx == gt_idx).sum().item()
            total_tokens += gt_c100.size(0)
    return total_correct / total_tokens if total_tokens > 0 else 0.0

def run_training_stage1(rank, world_size, pt_file_path, config, port='12368'):
    setup(rank, world_size, port)
    device = rank

    if rank == 0:
        print(f"Rank {rank}: Starting Stage 1 training...")

    full_dataset = PreprocessedDataset(pt_file_path)
    indices = list(range(len(full_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(len(indices) * config['train_split'])
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, sampler=val_sampler)

    model = CrossAttentionFusion().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    model = DDP(model, device_ids=[device], find_unused_parameters=False)

    criterion = {
        'ce': nn.CrossEntropyLoss()
    }

    checkpoint_path = f"{config['save_path']}_latest_fusion_all.pth"
    start_epoch = 0
    best_val_acc = 0.0
    patience = 3
    no_improve_epochs = 0

    if os.path.exists(checkpoint_path):
        try:
            map_location = f'cuda:{rank}'
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_acc = checkpoint['best_acc']
            if rank == 0:
                print(f"Successfully loaded Stage 1 checkpoint from '{checkpoint_path}'. Will continue training from epoch {start_epoch}.")
        except Exception as e:
            if rank == 0:
                print(f"Failed to load checkpoint '{checkpoint_path}': {e}. Will start training from scratch.")
    else:
        if rank == 0:
            print(f"Checkpoint '{checkpoint_path}' not found, will start training from scratch.")

    for epoch in range(start_epoch, config['epochs']):
        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, rank)
        val_acc = evaluate(model, val_loader, device, rank)

        if rank == 0:
            print(f"Stage 1, Epoch {epoch+1}/{config['epochs']}, Training Loss: {train_loss:.4f}, Validation Top-1 Accuracy: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve_epochs = 0
                torch.save(model.module.state_dict(), f"{config['save_path']}_best_fusion_all.pth")  #--------------Save
                print(f"Saved Stage 1 best model, validation accuracy: {best_val_acc:.4f}")
            else:
                no_improve_epochs += 1

            # New: Learning rate decay and termination logic
            if no_improve_epochs > 0 and no_improve_epochs % 2 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8
                print(f"Validation accuracy not improved for {no_improve_epochs} consecutive epochs, learning rate decayed to {optimizer.param_groups[0]['lr']:.6e}")

            if no_improve_epochs >= 20:
                print(f"Validation accuracy not improved for {no_improve_epochs} consecutive epochs, terminating training.")
                break

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_val_acc
            }
            torch.save(checkpoint, checkpoint_path)

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPU detected, this script is designed for distributed GPU training.")
        return

    pt_file_path = 'final_all_nan.pt'
    if not os.path.exists(pt_file_path):
        print(f"Error: Data file '{pt_file_path}' not found.")
        return

    config = {
        'train_split': 0.8,
        'batch_size': BATCH_SIZE,
        'feature_dim': FEATURE_DIM,
        'num_heads': NUM_HEADS,
        'hidden_dim': HIDDEN_DIM,
        'dropout': DROPOUT,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'save_path': FUSION_MODEL_SAVE_PATH,
    }

    mp.set_start_method('spawn', force=True)
    mp.spawn(run_training_stage1,
             args=(world_size, pt_file_path, config, '12368'),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()