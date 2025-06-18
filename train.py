#!/usr/bin/env python3
"""
ATICE-Net Training Script
Advanced Copy-Move Forgery Detection Network
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
import random
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import ATICENet
from losses import TotalLoss
from dataloaders import CASIADataLoader
from utils.metrics import calculate_metrics, calculate_batch_metrics
from utils.visualization import plot_training_curves, visualize_results


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config):
    """Get device for training"""
    if config['hardware']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif config['hardware']['device'] == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    return device


def create_directories(config):
    """Create necessary directories"""
    dirs = [
        config['paths']['checkpoints_dir'],
        config['paths']['logs_dir'],
        config['paths']['results_dir']
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    batch_metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'iou': 0.0
    }
    
    num_batches = len(train_loader)
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        outputs = model(images)
        predictions = outputs['output']
        
        # Calculate loss
        loss, loss_dict = criterion(predictions, {'mask': masks})
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            batch_metric = calculate_batch_metrics(
                predictions, masks, 
                threshold=config['evaluation']['threshold']
            )
        
        # Update running totals
        total_loss += loss.item()
        for key in batch_metrics:
            batch_metrics[key] += batch_metric[key]
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'F1': f"{batch_metric['f1']:.4f}",
            'IoU': f"{batch_metric['iou']:.4f}"
        })
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    for key in batch_metrics:
        batch_metrics[key] /= num_batches
    
    return avg_loss, batch_metrics, loss_dict


def validate_epoch(model, val_loader, criterion, device, epoch, config):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    batch_metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'iou': 0.0
    }
    
    num_batches = len(val_loader)
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = outputs['output']
            
            # Calculate loss
            loss, loss_dict = criterion(predictions, {'mask': masks})
            
            # Calculate metrics
            batch_metric = calculate_batch_metrics(
                predictions, masks, 
                threshold=config['evaluation']['threshold']
            )
            
            # Update running totals
            total_loss += loss.item()
            for key in batch_metrics:
                batch_metrics[key] += batch_metric[key]
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'F1': f"{batch_metric['f1']:.4f}",
                'IoU': f"{batch_metric['iou']:.4f}"
            })
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    for key in batch_metrics:
        batch_metrics[key] /= num_batches
    
    return avg_loss, batch_metrics


def save_checkpoint(model, optimizer, epoch, loss, metrics, config, is_best=False):
    """Save model checkpoint"""
    checkpoint_dir = config['paths']['checkpoints_dir']
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': config
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model with F1: {metrics['f1']:.4f}")
    
    # Keep only last 5 checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if len(checkpoints) > 5:
        for checkpoint_file in checkpoints[:-5]:
            os.remove(os.path.join(checkpoint_dir, checkpoint_file))


def main():
    parser = argparse.ArgumentParser(description='Train ATICE-Net')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    create_directories(config)
    
    # Get device
    device = get_device(config)
    
    # Initialize data loaders
    data_loader = CASIADataLoader(config)
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    
    # Initialize model
    model = ATICENet(config).to(device)
    
    # Print model info
    param_info = model.count_parameters()
    print(f"Model parameters: {param_info['trainable_millions']:.2f}M trainable")
    
    # Initialize loss function
    criterion = TotalLoss(config)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['scheduler_step_size'],
        gamma=config['training']['scheduler_gamma']
    )
    
    # Initialize tensorboard
    log_dir = os.path.join(config['paths']['logs_dir'], datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_f1 = 0.0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['metrics']['f1']
        print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")
    
    # Training loop
    train_losses = []
    val_losses = []
    train_metrics_history = {metric: [] for metric in ['accuracy', 'f1', 'iou']}
    val_metrics_history = {metric: [] for metric in ['accuracy', 'f1', 'iou']}
    
    print(f"Starting training for {config['training']['num_epochs']} epochs...")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Train
        train_loss, train_metrics, train_loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, config
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        for metric in train_metrics_history:
            train_metrics_history[metric].append(train_metrics[metric])
            val_metrics_history[metric].append(val_metrics[metric])
        
        # Tensorboard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        
        for metric in ['accuracy', 'f1', 'iou']:
            writer.add_scalar(f'{metric.upper()}/Train', train_metrics[metric], epoch)
            writer.add_scalar(f'{metric.upper()}/Val', val_metrics[metric], epoch)
        
        # Log learning rate
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print(f"Train IoU: {train_metrics['iou']:.4f}, Val IoU: {val_metrics['iou']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
        
        save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, config, is_best)
        
        # Save training curves
        if (epoch + 1) % 10 == 0:
            plot_training_curves(
                train_losses, val_losses,
                train_metrics_history, val_metrics_history,
                save_path=os.path.join(config['paths']['results_dir'], 'training_curves.png')
            )
    
    # Save final model
    final_checkpoint = {
        'epoch': config['training']['num_epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'best_f1': best_f1
    }
    
    final_path = os.path.join(config['paths']['checkpoints_dir'], 'final_model.pth')
    torch.save(final_checkpoint, final_path)
    
    print(f"\nTraining completed!")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Final model saved to: {final_path}")
    
    # Close tensorboard
    writer.close()


if __name__ == '__main__':
    main() 