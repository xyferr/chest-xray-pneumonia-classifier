"""
Training pipeline for Chest X-Ray Pneumonia Classification.

Includes:
- Training loop with mixed precision support
- Multiple optimizer choices (AdamW, SGD, Adam)
- Learning rate schedulers (Cosine, Step, Plateau)
- Early stopping
- Checkpoint saving
- TensorBoard and Weights & Biases logging
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    set_seed,
    get_device,
    setup_logging,
    get_logger,
    TrainingConfig,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    AverageMeter,
    MetricsTracker,
    ensure_dir,
)
from dataset import get_data_loaders, CLASS_NAMES
from models import create_model, count_parameters


# =============================================================================
# Optimizer Factory
# =============================================================================

def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Training configuration
    
    Returns:
        Configured optimizer
    """
    # Separate backbone and classifier parameters for different learning rates
    if hasattr(model, 'backbone') and hasattr(model, 'classifier'):
        # Transfer learning model - use different LRs
        backbone_params = model.backbone.parameters()
        classifier_params = model.classifier.parameters()
        
        param_groups = [
            {'params': backbone_params, 'lr': config.learning_rate * 0.1},
            {'params': classifier_params, 'lr': config.learning_rate},
        ]
    else:
        param_groups = model.parameters()
    
    optimizer_name = config.optimizer.lower()
    
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


# =============================================================================
# Scheduler Factory
# =============================================================================

def create_scheduler(
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> Optional[Any]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer instance
        config: Training configuration
        steps_per_epoch: Number of batches per epoch
    
    Returns:
        Configured scheduler or None
    """
    scheduler_name = config.scheduler.lower()
    
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01,
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7,
        )
    elif scheduler_name == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=steps_per_epoch,
        )
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


# =============================================================================
# Loss Function
# =============================================================================

def create_loss_function(
    class_weights: Optional[torch.Tensor] = None,
    use_class_weights: bool = True,
    device: torch.device = None,
) -> nn.Module:
    """
    Create loss function with optional class weighting.
    
    Args:
        class_weights: Tensor of class weights
        use_class_weights: Whether to use class weights
        device: Target device
    
    Returns:
        Loss function module
    """
    if use_class_weights and class_weights is not None:
        weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    return criterion


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: TrainingConfig,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[Any] = None,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Compute device
        epoch: Current epoch number
        config: Training configuration
        scaler: Gradient scaler for mixed precision
        scheduler: Learning rate scheduler (OneCycleLR)
        writer: TensorBoard writer
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    logger = get_logger()
    
    # Metrics tracking
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc')
    
    # Progress bar
    pbar = tqdm(
        train_loader,
        desc=f'Epoch {epoch}/{config.num_epochs} [Train]',
        leave=False,
    )
    
    global_step = epoch * len(train_loader)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if config.mixed_precision and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Step scheduler if OneCycleLR
        if scheduler is not None and config.scheduler.lower() == 'onecycle':
            scheduler.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = correct / labels.size(0)
        
        # Update meters
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(accuracy, labels.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}',
        })
        
        # Log to TensorBoard
        if writer is not None and batch_idx % config.log_interval == 0:
            step = global_step + batch_idx
            writer.add_scalar('Train/Loss_step', loss.item(), step)
            writer.add_scalar('Train/Acc_step', accuracy, step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], step)
    
    return {
        'loss': loss_meter.avg,
        'accuracy': acc_meter.avg,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: TrainingConfig,
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Compute device
        epoch: Current epoch number
        config: Training configuration
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc')
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(
        val_loader,
        desc=f'Epoch {epoch}/{config.num_epochs} [Val]',
        leave=False,
    )
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        if config.mixed_precision:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Get predictions
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = correct / labels.size(0)
        
        # Update meters
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(accuracy, labels.size(0))
        
        # Collect for metrics
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of PNEUMONIA
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}',
        })
    
    # Calculate additional metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Per-class accuracy
    normal_mask = all_labels == 0
    pneumonia_mask = all_labels == 1
    
    normal_acc = (all_preds[normal_mask] == all_labels[normal_mask]).mean() if normal_mask.sum() > 0 else 0
    pneumonia_acc = (all_preds[pneumonia_mask] == all_labels[pneumonia_mask]).mean() if pneumonia_mask.sum() > 0 else 0
    
    return {
        'loss': loss_meter.avg,
        'accuracy': acc_meter.avg,
        'normal_accuracy': normal_acc,
        'pneumonia_accuracy': pneumonia_acc,
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: TrainingConfig) -> Dict[str, Any]:
    """
    Main training function.
    
    Args:
        config: Training configuration
    
    Returns:
        Dictionary with training results
    """
    # Setup
    set_seed(config.seed)
    device = get_device()
    logger = setup_logging(
        log_dir=config.output_dir,
        log_to_file=True,
    )
    
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    logger.info(f"Config: {config.to_dict()}")
    
    # Create output directories
    checkpoint_dir = ensure_dir(Path(config.output_dir) / 'checkpoints')
    
    # Setup TensorBoard
    writer = None
    if config.use_tensorboard:
        tb_dir = ensure_dir(Path(config.output_dir) / 'tensorboard')
        writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"TensorBoard logs: {tb_dir}")
    
    # Setup Weights & Biases
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.to_dict(),
                name=f"{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            logger.info("Weights & Biases initialized")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
            config.use_wandb = False
    
    # Load data
    logger.info("Loading datasets...")
    data = get_data_loaders(config)
    train_loader = data['train']
    val_loader = data['val']
    class_weights = data['class_weights']
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    logger.info(f"Creating model: {config.model_name}")
    model = create_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    )
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    
    logger.info(f"Optimizer: {config.optimizer}, LR: {config.learning_rate}")
    logger.info(f"Scheduler: {config.scheduler}")
    
    # Create loss function
    criterion = create_loss_function(
        class_weights=class_weights if config.use_class_weights else None,
        use_class_weights=config.use_class_weights,
        device=device,
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.mixed_precision and device.type == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        mode='min',
        verbose=True,
    ) if config.early_stopping else None
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
    }
    
    logger.info("\nStarting training loop...")
    
    for epoch in range(1, config.num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config.num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler,
            scheduler=scheduler if config.scheduler.lower() == 'onecycle' else None,
            writer=writer,
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            config=config,
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        logger.info(
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}"
        )
        logger.info(
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f} "
            f"(Normal: {val_metrics['normal_accuracy']:.4f}, "
            f"Pneumonia: {val_metrics['pneumonia_accuracy']:.4f})"
        )
        logger.info(f"Learning Rate: {current_lr:.2e}")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(current_lr)
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalars('Loss', {
                'train': train_metrics['loss'],
                'val': val_metrics['loss'],
            }, epoch)
            writer.add_scalars('Accuracy', {
                'train': train_metrics['accuracy'],
                'val': val_metrics['accuracy'],
            }, epoch)
            writer.add_scalar('LR', current_lr, epoch)
        
        # W&B logging
        if config.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'lr': current_lr,
            })
        
        # Step scheduler (except OneCycleLR which steps per batch)
        if scheduler is not None and config.scheduler.lower() != 'onecycle':
            if config.scheduler.lower() == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Check for best model
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            
            # Save best model
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_metrics['loss'],
                metrics=val_metrics,
                filepath=str(checkpoint_dir / 'best_model.pt'),
                scheduler=scheduler,
                config=config,
            )
            logger.info(f"✓ New best model saved! (Val Loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if epoch % config.checkpoint_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_metrics['loss'],
                metrics=val_metrics,
                filepath=str(checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'),
                scheduler=scheduler,
                config=config,
            )
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
    
    # Training complete
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Best Val Acc: {best_val_acc:.4f}")
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=val_metrics['loss'],
        metrics=val_metrics,
        filepath=str(checkpoint_dir / 'final_model.pt'),
        scheduler=scheduler,
        config=config,
    )
    
    # Close writers
    if writer is not None:
        writer.close()
    
    if config.use_wandb:
        wandb.finish()
    
    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'history': history,
        'model_path': str(checkpoint_dir / 'best_model.pt'),
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Chest X-Ray Pneumonia Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data
    parser.add_argument(
        '--data-dir', type=str, default='data/chest_xray',
        help='Path to data directory',
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs',
        help='Output directory for checkpoints and logs',
    )
    
    # Model
    parser.add_argument(
        '--model', type=str, default='resnet50',
        choices=['baseline_cnn', 'resnet18', 'resnet34', 'resnet50', 
                 'efficientnet_b0', 'densenet121', 'mobilenet_v2'],
        help='Model architecture',
    )
    parser.add_argument(
        '--pretrained', action='store_true', default=True,
        help='Use pretrained weights',
    )
    parser.add_argument(
        '--freeze-backbone', action='store_true',
        help='Freeze backbone weights',
    )
    
    # Training
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size',
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate',
    )
    parser.add_argument(
        '--optimizer', type=str, default='adamw',
        choices=['adamw', 'adam', 'sgd'],
        help='Optimizer',
    )
    parser.add_argument(
        '--scheduler', type=str, default='cosine',
        choices=['cosine', 'step', 'plateau', 'onecycle', 'none'],
        help='Learning rate scheduler',
    )
    
    # Regularization
    parser.add_argument(
        '--weight-decay', type=float, default=1e-4,
        help='Weight decay',
    )
    parser.add_argument(
        '--no-augmentation', action='store_true',
        help='Disable data augmentation',
    )
    
    # Early stopping
    parser.add_argument(
        '--no-early-stopping', action='store_true',
        help='Disable early stopping',
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Early stopping patience',
    )
    
    # Logging
    parser.add_argument(
        '--no-tensorboard', action='store_true',
        help='Disable TensorBoard logging',
    )
    parser.add_argument(
        '--wandb', action='store_true',
        help='Enable Weights & Biases logging',
    )
    parser.add_argument(
        '--wandb-project', type=str, default='chest-xray-pneumonia',
        help='W&B project name',
    )
    
    # Misc
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--no-mixed-precision', action='store_true',
        help='Disable mixed precision training',
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create config from args
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        weight_decay=args.weight_decay,
        use_augmentation=not args.no_augmentation,
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
        use_tensorboard=not args.no_tensorboard,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        seed=args.seed,
        mixed_precision=not args.no_mixed_precision,
    )
    
    # Run training
    results = train(config)
    
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Best Model: {results['model_path']}")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"Best Val Acc: {results['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
