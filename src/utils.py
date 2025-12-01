"""
Utility functions for the Chest X-Ray Pneumonia Classifier.

Includes:
- Seed setting for reproducibility
- Device detection (CPU/GPU/MPS)
- Configuration management
- Logging utilities
- Visualization helpers
"""

import os
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime


# =============================================================================
# Seed & Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


# =============================================================================
# Device Detection
# =============================================================================

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device for computation.
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU (default: True)
    
    Returns:
        torch.device: The selected device (cuda, mps, or cpu)
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"Using CUDA GPU: {gpu_name}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using Apple MPS (Metal Performance Shaders)")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU (no GPU available)")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU (as requested)")
    
    return device


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about available compute devices.
    
    Returns:
        Dict containing device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count(),
    }
    
    if info["cuda_available"]:
        info["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
            }
            for i in range(info["cuda_device_count"])
        ]
    
    return info


# =============================================================================
# Configuration Management
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Data paths
    data_dir: str = "data/chest_xray"
    output_dir: str = "outputs"
    
    # Model settings
    model_name: str = "resnet50"  # 'baseline_cnn', 'resnet18', 'resnet50', 'efficientnet_b0'
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 2
    
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Optimizer settings
    optimizer: str = "adamw"  # 'adamw', 'sgd', 'adam'
    momentum: float = 0.9  # For SGD
    
    # Scheduler settings
    scheduler: str = "cosine"  # 'cosine', 'step', 'plateau', 'none'
    step_size: int = 10  # For StepLR
    gamma: float = 0.1  # For StepLR
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Data augmentation
    use_augmentation: bool = True
    image_size: int = 224
    
    # Class imbalance handling
    use_weighted_sampler: bool = True
    use_class_weights: bool = True
    
    # Logging
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "chest-xray-pneumonia"
    wandb_entity: Optional[str] = None
    log_interval: int = 10  # Log every N batches
    
    # Misc
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_interval: int = 5  # Save every N epochs
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Adjust num_workers for Windows
        if os.name == 'nt':  # Windows
            self.num_workers = min(self.num_workers, 0)  # Windows multiprocessing issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    
    model_path: str = "outputs/best_model.pt"
    model_name: str = "resnet50"
    num_classes: int = 2
    image_size: int = 224
    device: str = "auto"  # 'auto', 'cuda', 'cpu', 'mps'
    class_names: List[str] = field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])
    threshold: float = 0.5  # Classification threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# Logging Utilities
# =============================================================================

def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_filename: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (default: INFO)
        log_to_file: Whether to save logs to file
        log_filename: Custom log filename
    
    Returns:
        Configured logger instance
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    logger = logging.getLogger('chest_xray')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        if log_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'training_{timestamp}.log'
        
        file_handler = logging.FileHandler(
            Path(log_dir) / log_filename,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'chest_xray') -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# =============================================================================
# Model Utilities
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
    scheduler: Optional[Any] = None,
    config: Optional[TrainingConfig] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        epoch: Current epoch
        loss: Current loss value
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        config: Training configuration (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config.to_dict()
    
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)
    logging.getLogger('chest_xray').info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load tensors to
    
    Returns:
        Checkpoint dictionary with metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load checkpoint with weights_only=False for compatibility with older checkpoints
    # This is safe since we're loading our own trained models
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors a metric and stops training when no improvement is seen
    for a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/metrics
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger = get_logger()
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:  # mode == 'max'
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(
                    f"EarlyStopping: {self.counter}/{self.patience} "
                    f"(best: {self.best_score:.4f}, current: {score:.4f})"
                )
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.logger.info("Early stopping triggered!")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


# =============================================================================
# Metrics Tracking
# =============================================================================

class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """
    
    def __init__(self, name: str = 'Metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.
        
        Args:
            val: New value
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricsTracker:
    """Track multiple metrics during training."""
    
    def __init__(self, metric_names: List[str]):
        """
        Initialize metrics tracker.
        
        Args:
            metric_names: List of metric names to track
        """
        self.metrics = {name: AverageMeter(name) for name in metric_names}
    
    def update(self, metrics_dict: Dict[str, float], n: int = 1):
        """
        Update all metrics.
        
        Args:
            metrics_dict: Dictionary of metric values
            n: Number of samples
        """
        for name, value in metrics_dict.items():
            if name in self.metrics:
                self.metrics[name].update(value, n)
    
    def reset(self):
        """Reset all metrics."""
        for meter in self.metrics.values():
            meter.reset()
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics."""
        return {name: meter.avg for name, meter in self.metrics.items()}
    
    def __str__(self):
        return " | ".join(str(meter) for meter in self.metrics.values())


# =============================================================================
# File Utilities
# =============================================================================

def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None
    
    checkpoints = list(checkpoint_path.glob("*.pt")) + list(checkpoint_path.glob("*.pth"))
    
    if not checkpoints:
        return None
    
    # Sort by modification time and return latest
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test seed setting
    set_seed(42)
    print(f"✓ Seed set to 42")
    
    # Test device detection
    device = get_device()
    print(f"✓ Device: {device}")
    
    # Test device info
    info = get_device_info()
    print(f"✓ Device info: {info}")
    
    # Test config
    config = TrainingConfig()
    print(f"✓ Config created: {config.model_name}")
    
    # Test logging
    logger = setup_logging(log_to_file=False)
    logger.info("Test log message")
    print("✓ Logging configured")
    
    # Test early stopping
    es = EarlyStopping(patience=3, mode='min', verbose=False)
    for loss in [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88]:
        if es(loss):
            print(f"✓ Early stopping triggered at loss={loss}")
            break
    
    # Test average meter
    meter = AverageMeter('loss')
    for val in [0.5, 0.4, 0.3]:
        meter.update(val)
    print(f"✓ AverageMeter: {meter}")
    
    print("\nAll utility tests passed!")
