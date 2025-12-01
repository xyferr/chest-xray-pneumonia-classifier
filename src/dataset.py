"""
Dataset module for Chest X-Ray Pneumonia Classification.

Includes:
- ChestXRayDataset: Custom dataset class
- Data transforms (train/val/test)
- Data loaders with class imbalance handling
- Augmentation pipelines
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Callable, Any

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets

from utils import get_logger, TrainingConfig


# =============================================================================
# Constants
# =============================================================================

# ImageNet normalization (used for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class names
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
NUM_CLASSES = 2


# =============================================================================
# Data Transforms
# =============================================================================

def get_train_transforms(image_size: int = 224, use_augmentation: bool = True) -> transforms.Compose:
    """
    Get training data transforms with augmentation.
    
    Args:
        image_size: Target image size (default: 224 for pretrained models)
        use_augmentation: Whether to apply data augmentation
    
    Returns:
        Composed transforms for training data
    """
    if use_augmentation:
        transform_list = [
            transforms.Resize((image_size + 32, image_size + 32)),  # Resize slightly larger
            transforms.RandomCrop(image_size),  # Random crop to target size
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # Random erasing for regularization
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ]
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    
    return transforms.Compose(transform_list)


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test data transforms (no augmentation).
    
    Args:
        image_size: Target image size
    
    Returns:
        Composed transforms for validation/test data
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get inference transforms (same as validation).
    
    Args:
        image_size: Target image size
    
    Returns:
        Composed transforms for inference
    """
    return get_val_transforms(image_size)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized image tensor [C, H, W]
    
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    if tensor.device.type != 'cpu':
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    return tensor * std + mean


# =============================================================================
# Custom Dataset
# =============================================================================

class ChestXRayDataset(Dataset):
    """
    Custom dataset for Chest X-Ray images.
    
    Supports:
    - Loading from directory structure (ImageFolder compatible)
    - Custom transforms
    - Class balancing information
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing train/val/test folders
            split: Data split ('train', 'val', or 'test')
            transform: Image transforms to apply
            target_transform: Target/label transforms to apply
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.target_transform = target_transform
        self.logger = get_logger()
        
        # Validate split directory exists
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")
        
        # Class mapping
        self.classes = CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.samples: List[Tuple[str, int]] = []
        self._load_samples()
        
        # Calculate class distribution
        self.class_counts = self._get_class_counts()
        self.class_weights = self._calculate_class_weights()
        
        self.logger.info(
            f"Loaded {len(self.samples)} images from {split} split | "
            f"NORMAL: {self.class_counts[0]}, PNEUMONIA: {self.class_counts[1]}"
        )
    
    def _load_samples(self) -> None:
        """Load all image paths and their labels."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        for class_name in self.classes:
            class_dir = self.split_dir / class_name
            
            if not class_dir.exists():
                self.logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_path), class_idx))
    
    def _get_class_counts(self) -> Dict[int, int]:
        """Get count of samples per class."""
        counts = {i: 0 for i in range(len(self.classes))}
        for _, label in self.samples:
            counts[label] += 1
        return counts
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalance.
        
        Uses inverse frequency weighting: weight = total / (num_classes * count)
        """
        total = len(self.samples)
        num_classes = len(self.classes)
        
        weights = []
        for i in range(num_classes):
            count = self.class_counts[i]
            if count > 0:
                weight = total / (num_classes * count)
            else:
                weight = 0.0
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Get weight for each sample (for WeightedRandomSampler).
        
        Returns:
            Tensor of weights, one per sample
        """
        sample_weights = []
        for _, label in self.samples:
            sample_weights.append(self.class_weights[label].item())
        return torch.tensor(sample_weights, dtype=torch.float32)
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample by index.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (image tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def get_image_path(self, idx: int) -> str:
        """Get the file path of an image by index."""
        return self.samples[idx][0]


# =============================================================================
# Data Loaders
# =============================================================================

def create_weighted_sampler(dataset: ChestXRayDataset) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for handling class imbalance.
    
    Args:
        dataset: ChestXRayDataset instance
    
    Returns:
        WeightedRandomSampler that oversamples minority class
    """
    sample_weights = dataset.get_sample_weights()
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow sampling with replacement for oversampling
    )
    
    return sampler


def get_data_loaders(
    config: TrainingConfig,
    data_dir: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        config: Training configuration
        data_dir: Override data directory from config
    
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    logger = get_logger()
    
    # Use provided data_dir or get from config
    root_dir = data_dir if data_dir else config.data_dir
    
    # Create transforms
    train_transform = get_train_transforms(
        image_size=config.image_size,
        use_augmentation=config.use_augmentation
    )
    val_transform = get_val_transforms(image_size=config.image_size)
    
    # Create datasets
    logger.info("Loading datasets...")
    
    train_dataset = ChestXRayDataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = ChestXRayDataset(
        root_dir=root_dir,
        split='val',
        transform=val_transform
    )
    
    test_dataset = ChestXRayDataset(
        root_dir=root_dir,
        split='test',
        transform=val_transform
    )
    
    # Create sampler for training (if using weighted sampling)
    train_sampler = None
    train_shuffle = True
    
    if config.use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_dataset)
        train_shuffle = False  # Sampler handles shuffling
        logger.info("Using WeightedRandomSampler for class balancing")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True  # Drop incomplete batches for training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    # Log dataset statistics
    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    # Store class weights in config for loss function
    class_weights = train_dataset.class_weights
    logger.info(f"Class weights: NORMAL={class_weights[0]:.4f}, PNEUMONIA={class_weights[1]:.4f}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'class_weights': class_weights,
    }


def get_single_loader(
    root_dir: str,
    split: str,
    batch_size: int = 32,
    image_size: int = 224,
    use_augmentation: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a single data loader for a specific split.
    
    Args:
        root_dir: Data root directory
        split: Data split ('train', 'val', or 'test')
        batch_size: Batch size
        image_size: Image size
        use_augmentation: Whether to use augmentation
        num_workers: Number of worker processes
    
    Returns:
        DataLoader for the specified split
    """
    if split == 'train' and use_augmentation:
        transform = get_train_transforms(image_size, use_augmentation=True)
    else:
        transform = get_val_transforms(image_size)
    
    dataset = ChestXRayDataset(
        root_dir=root_dir,
        split=split,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader


# =============================================================================
# Visualization Helpers
# =============================================================================

def visualize_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    num_images: int = 8,
    figsize: Tuple[int, int] = (16, 8),
) -> None:
    """
    Visualize a batch of images with labels.
    
    Args:
        images: Batch of image tensors [B, C, H, W]
        labels: Ground truth labels
        predictions: Model predictions (optional)
        num_images: Number of images to display
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    num_images = min(num_images, len(images))
    
    fig, axes = plt.subplots(2, num_images // 2, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_images):
        # Denormalize image
        img = denormalize(images[i].cpu())
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Create title
        true_label = CLASS_NAMES[labels[i].item()]
        title = f"True: {true_label}"
        
        if predictions is not None:
            pred_label = CLASS_NAMES[predictions[i].item()]
            title += f"\nPred: {pred_label}"
            
            # Color based on correctness
            color = 'green' if predictions[i] == labels[i] else 'red'
            axes[i].set_title(title, color=color)
        else:
            axes[i].set_title(title)
    
    plt.tight_layout()
    plt.show()


def get_dataset_stats(root_dir: str) -> Dict[str, Any]:
    """
    Get statistics about the dataset.
    
    Args:
        root_dir: Root directory of the dataset
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = ChestXRayDataset(root_dir=root_dir, split=split)
            stats[split] = {
                'total': len(dataset),
                'normal': dataset.class_counts[0],
                'pneumonia': dataset.class_counts[1],
                'imbalance_ratio': dataset.class_counts[1] / max(dataset.class_counts[0], 1),
            }
        except Exception as e:
            stats[split] = {'error': str(e)}
    
    return stats


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import logging
    from utils import setup_logging, TrainingConfig
    
    # Setup logging
    setup_logging(log_to_file=False)
    logger = get_logger()
    
    # Test with sample data
    data_dir = "data/chest_xray"
    
    print("=" * 60)
    print("Testing Dataset Module")
    print("=" * 60)
    
    # Test transforms
    print("\n1. Testing transforms...")
    train_tf = get_train_transforms(224, use_augmentation=True)
    val_tf = get_val_transforms(224)
    print(f"✓ Train transforms: {len(train_tf.transforms)} operations")
    print(f"✓ Val transforms: {len(val_tf.transforms)} operations")
    
    # Test dataset loading
    print("\n2. Testing dataset loading...")
    try:
        config = TrainingConfig(data_dir=data_dir)
        loaders = get_data_loaders(config)
        
        print(f"✓ Train loader: {len(loaders['train'])} batches")
        print(f"✓ Val loader: {len(loaders['val'])} batches")
        print(f"✓ Test loader: {len(loaders['test'])} batches")
        
        # Test batch loading
        print("\n3. Testing batch loading...")
        images, labels = next(iter(loaders['train']))
        print(f"✓ Batch shape: {images.shape}")
        print(f"✓ Labels shape: {labels.shape}")
        print(f"✓ Image range: [{images.min():.2f}, {images.max():.2f}]")
        
        # Test class weights
        print("\n4. Class weights for loss function:")
        print(f"   NORMAL: {loaders['class_weights'][0]:.4f}")
        print(f"   PNEUMONIA: {loaders['class_weights'][1]:.4f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  (This is expected if data directory doesn't exist)")
    
    # Get dataset stats
    print("\n5. Dataset statistics:")
    try:
        stats = get_dataset_stats(data_dir)
        for split, info in stats.items():
            if 'error' not in info:
                print(f"   {split}: {info['total']} total "
                      f"(NORMAL: {info['normal']}, PNEUMONIA: {info['pneumonia']}, "
                      f"ratio: {info['imbalance_ratio']:.2f})")
    except Exception as e:
        print(f"   Error getting stats: {e}")
    
    print("\n" + "=" * 60)
    print("Dataset module tests complete!")
    print("=" * 60)
