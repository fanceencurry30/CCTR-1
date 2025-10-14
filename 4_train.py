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
from tqdm import tqdm

def print_hyperparameters(config, rank):
    if rank == 0:
        print("----------- Hyperparameters -----------")
        # Filter out built-in attributes, only print hyperparameters we care about
        for key, value in config.__class__.__dict__.items():
            if not key.startswith('__'):
                print(f"{key:<25}: {getattr(config, key)}")
        print("---------------------------------------")

def train(rank, world_size):
    """Initialize DDP training"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    config = Config()
    print_hyperparameters(config, rank)
    
    # Device info
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        print(f"[Main Process] Using {torch.cuda.get_device_name(rank)}")
        os.makedirs("checkpoints", exist_ok=True)
    
    # Load config
    config.device = device
    dataloader, char2id = get_dataloader(config, rank, world_size)
    
    # Calculate vocabulary size
    vocab_size = len(char2id)
    if rank == 0:
        print(f"[Main Process] Vocabulary size: {vocab_size}")
    
    # Initialize model
    model = CharTransformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len
    ).to(device)

    model = DDP(model, device_ids=[rank], output_device=rank)

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    # --- Checkpoint resume logic ---
    if config.continue_training:
        if os.path.exists(config.load_checkpoint_path):
            if rank == 0:
                print(f"--- Checkpoint found, resuming training from {config.load_checkpoint_path} ---")
            
            # Load checkpoint to the device corresponding to the current process
            checkpoint = torch.load(config.load_checkpoint_path, map_location=device)
            
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            if rank == 0:
                print(f"--- Successfully resumed! Training will start from epoch {start_epoch} ---")
        else:
            if rank == 0:
                print(f"--- Checkpoint {config.load_checkpoint_path} not found. Training will start from scratch. ---")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        dataloader.sampler.set_epoch(epoch) # Ensure shuffling for distributed sampling each epoch
        total_loss = 0
        
        # Show tqdm progress bar only on main process (rank 0)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0))
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
            # Show real-time loss on progress bar
            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Wait for all processes to finish current epoch
        dist.barrier()

        # Only print average loss and save model on main process
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint with all states
            save_path = f"checkpoints/char_transformer_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Checkpoint saved to: {save_path}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPUs, starting distributed training...")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    
