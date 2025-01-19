import os
from pathlib import Path

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import random

import torch
import models
from data.dataset import CLIPGraspingDataset
from torch.utils.data import DataLoader


@hydra.main(config_path="cfgs", config_name="train")
def main(cfg):
    # Set random seeds
    seed = cfg['train']['random_seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print("Random seeds set.")

    hydra_dir = Path(os.getcwd())
    checkpoint_path = hydra_dir / 'checkpoints'
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path \
        if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None
    print("Checkpoint paths set.")

    cfg['data']['amt_data'] = '/content/snare/amt'
    cfg['data']['folds'] = 'folds_adversarial'
    cfg['data']['clip_lang_feats'] = '/content/drive/MyDrive/Research/langfeat-512-clipViT32.json.gz'
    cfg['data']['clip_img_feats'] = '/content/drive/MyDrive/Research/shapenet-clipViT32-frames.json.gz'
    print("Configuration paths set.")

    # Detect available accelerator and force GPU usage if available
    if torch.cuda.is_available():
        accelerator = "cuda"
        devices = torch.cuda.device_count()  # Use all available GPUs
        strategy = "ddp_find_unused_parameters_false" if devices > 1 else "auto"
        print(f"Using GPU with {devices} devices")
    else:
        accelerator = "cpu"
        devices = "auto"
        strategy = None
        print("CUDA not available. Using CPU.")

    # Print device details for debugging
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Initialize Trainer with updated settings
    trainer = Trainer(
        accelerator=accelerator,  # Automatically selects "cuda" or "cpu"
        devices=devices,  # Use all available GPUs
        strategy=strategy,  # Use DDP for multi-GPU training
        fast_dev_run=cfg['debug'],  # Debug mode
        callbacks=[ModelCheckpoint(
            monitor=cfg['wandb']['saver']['monitor'],
            dirpath=checkpoint_path,
            filename='{epoch:04d}-{val_acc:.5f}',
            save_top_k=1,
            save_last=True,
        )],
        max_epochs=cfg['train']['max_epochs'],
        enable_progress_bar=True,  # New progress bar handling
    )
    print(f"Trainer initialized with accelerator: {accelerator}, devices: {devices}")

    # Dataset Initialization
    print("Loading datasets...")
    train = CLIPGraspingDataset(cfg, mode='train')
    print(f"Loaded training dataset with {len(train)} entries.")

    print("Initializing CLIPGraspingDataset for validation...")
    valid = CLIPGraspingDataset(cfg, mode='valid')
    print(f"Loaded validation dataset with {len(valid)} entries.")

    print("Initializing CLIPGraspingDataset for testing...")
    test = CLIPGraspingDataset(cfg, mode='test')
    print(f"Loaded test dataset with {len(test)} entries.")

    # DataLoader Initialization
    print("Initializing DataLoaders...")
    train_loader = DataLoader(train, batch_size=cfg['train']['batch_size'], pin_memory=True, shuffle=True)
    print("Training DataLoader initialized.")

    print("Initializing Validation DataLoader...")
    valid_loader = DataLoader(valid, batch_size=cfg['train']['batch_size'], pin_memory=True)
    print("Validation DataLoader initialized.")

    print("Initializing Test DataLoader...")
    test_loader = DataLoader(test, batch_size=cfg['train']['batch_size'], pin_memory=True)
    print("Test DataLoader initialized.")

    # Model Initialization
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.names[cfg['train']['model']](cfg, train, valid).to(device)
    print(f"Model is on device: {device}")

    # Print GPU memory status
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Resume from checkpoint if available
    if last_checkpoint and cfg['train']['load_from_last_ckpt']:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint)
        trainer.current_epoch = last_ckpt['epoch']
        trainer.global_step = last_ckpt['global_step']
        del last_ckpt

    # Start Training
    print("Starting training...")
    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=valid_loader,
    )
    print("Training completed!")

    # Start Testing
    print("Starting testing...")
    trainer.test(
        test_dataloaders=test_loader,
        ckpt_path='best'
    )
    print("Testing completed!")


if __name__ == "__main__":
    main()
