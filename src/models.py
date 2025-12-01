"""
Model architectures for Chest X-Ray Pneumonia Classification.

Includes:
- BaselineCNN: Simple 3-layer convolutional network
- TransferModel: Transfer learning with pretrained backbones (ResNet, EfficientNet, etc.)
- Model factory for easy model creation
"""

from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import get_logger, count_parameters


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_MODELS = [
    'baseline_cnn',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'densenet121',
    'densenet169',
    'vgg16',
    'mobilenet_v2',
    'mobilenet_v3_small',
    'mobilenet_v3_large',
]


# =============================================================================
# Baseline CNN
# =============================================================================

class BaselineCNN(nn.Module):
    """
    Simple baseline CNN for chest X-ray classification.
    
    Architecture:
    - 3 convolutional blocks (Conv2d + BatchNorm + ReLU + MaxPool)
    - Global average pooling
    - Fully connected classifier with dropout
    
    This serves as a baseline to compare against transfer learning models.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize the baseline CNN.
        
        Args:
            num_classes: Number of output classes (default: 2 for binary)
            in_channels: Number of input channels (default: 3 for RGB)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Convolutional blocks
        # Block 1: 3 -> 32 channels, 224 -> 112
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112
        )
        
        # Block 2: 32 -> 64 channels, 112 -> 56
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56
        )
        
        # Block 3: 64 -> 128 channels, 56 -> 28
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28
        )
        
        # Block 4: 128 -> 256 channels, 28 -> 14
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Output logits [B, num_classes]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature representation before classifier.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Feature tensor [B, 256]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x


# =============================================================================
# Transfer Learning Models
# =============================================================================

class TransferModel(nn.Module):
    """
    Transfer learning model with pretrained backbones.
    
    Supports:
    - ResNet variants (18, 34, 50, 101)
    - EfficientNet variants (B0, B1, B2)
    - DenseNet variants (121, 169)
    - VGG16
    - MobileNet variants (V2, V3)
    
    Features:
    - Option to freeze backbone for feature extraction
    - Custom classifier head
    - Gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize the transfer learning model.
        
        Args:
            model_name: Name of the pretrained backbone
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights
            dropout_rate: Dropout rate for classifier
        """
        super().__init__()
        
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.logger = get_logger()
        
        # Build the model
        self.backbone, self.feature_dim = self._create_backbone()
        self.classifier = self._create_classifier(dropout_rate)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        self.logger.info(
            f"Created {model_name} model | "
            f"Feature dim: {self.feature_dim} | "
            f"Pretrained: {pretrained} | "
            f"Frozen: {freeze_backbone}"
        )
    
    def _create_backbone(self) -> Tuple[nn.Module, int]:
        """
        Create the backbone network.
        
        Returns:
            Tuple of (backbone module, feature dimension)
        """
        weights = 'IMAGENET1K_V1' if self.pretrained else None
        
        # ResNet variants
        if self.model_name == 'resnet18':
            model = models.resnet18(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            
        elif self.model_name == 'resnet34':
            model = models.resnet34(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            
        elif self.model_name == 'resnet50':
            model = models.resnet50(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            
        elif self.model_name == 'resnet101':
            model = models.resnet101(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        
        # EfficientNet variants
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=weights)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            
        elif self.model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(weights=weights)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            
        elif self.model_name == 'efficientnet_b2':
            model = models.efficientnet_b2(weights=weights)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
        
        # DenseNet variants
        elif self.model_name == 'densenet121':
            model = models.densenet121(weights=weights)
            feature_dim = model.classifier.in_features
            model.classifier = nn.Identity()
            
        elif self.model_name == 'densenet169':
            model = models.densenet169(weights=weights)
            feature_dim = model.classifier.in_features
            model.classifier = nn.Identity()
        
        # VGG
        elif self.model_name == 'vgg16':
            model = models.vgg16_bn(weights=weights)
            feature_dim = model.classifier[0].in_features
            model.classifier = nn.Identity()
            # Add adaptive pooling for VGG
            model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # MobileNet variants
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=weights)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            
        elif self.model_name == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(weights=weights)
            feature_dim = model.classifier[0].in_features
            model.classifier = nn.Identity()
            
        elif self.model_name == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(weights=weights)
            feature_dim = model.classifier[0].in_features
            model.classifier = nn.Identity()
        
        else:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Supported models: {SUPPORTED_MODELS}"
            )
        
        return model, feature_dim
    
    def _create_classifier(self, dropout_rate: float) -> nn.Module:
        """
        Create the classifier head.
        
        Args:
            dropout_rate: Dropout rate
        
        Returns:
            Classifier module
        """
        return nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, self.num_classes),
        )
    
    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.logger.info("Backbone frozen - only classifier will be trained")
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """
        Unfreeze backbone parameters for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end.
                       If None, unfreezes all layers.
        """
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.logger.info("All backbone layers unfrozen")
        else:
            # Get all parameter groups
            params = list(self.backbone.named_parameters())
            
            # Unfreeze last N layers
            for name, param in params[-num_layers:]:
                param.requires_grad = True
            
            self.logger.info(f"Unfroze last {num_layers} parameter groups")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Output logits [B, num_classes]
        """
        features = self.backbone(x)
        
        # Handle different output shapes from backbones
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        output = self.classifier(features)
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature representation before classifier.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Feature tensor [B, feature_dim]
        """
        features = self.backbone(x)
        
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        return features
    
    def get_cam_layers(self) -> List[nn.Module]:
        """
        Get the layers to use for Grad-CAM visualization.
        
        Returns:
            List of layer modules for CAM
        """
        if 'resnet' in self.model_name:
            return [self.backbone.layer4[-1]]
        elif 'efficientnet' in self.model_name:
            return [self.backbone.features[-1]]
        elif 'densenet' in self.model_name:
            return [self.backbone.features.denseblock4]
        elif 'vgg' in self.model_name:
            return [self.backbone.features[-1]]
        elif 'mobilenet' in self.model_name:
            return [self.backbone.features[-1]]
        else:
            return []


# =============================================================================
# Model Factory
# =============================================================================

def create_model(
    model_name: str = 'resnet50',
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.5,
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for transfer learning)
        freeze_backbone: Whether to freeze backbone weights
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Initialized model
    """
    logger = get_logger()
    model_name = model_name.lower()
    
    if model_name == 'baseline_cnn':
        model = BaselineCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    elif model_name in SUPPORTED_MODELS:
        model = TransferModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {SUPPORTED_MODELS}"
        )
    
    # Log model info
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    logger.info(
        f"Model: {model_name} | "
        f"Total params: {total_params:,} | "
        f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)"
    )
    
    return model


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model architecture.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary with model information
    """
    model_name = model_name.lower()
    
    info = {
        'name': model_name,
        'supported': model_name in SUPPORTED_MODELS,
    }
    
    # Add specific info based on model type
    if model_name == 'baseline_cnn':
        info.update({
            'type': 'custom',
            'pretrained_available': False,
            'input_size': 224,
            'approximate_params': '~500K',
        })
    elif 'resnet' in model_name:
        info.update({
            'type': 'transfer_learning',
            'family': 'ResNet',
            'pretrained_available': True,
            'input_size': 224,
        })
    elif 'efficientnet' in model_name:
        info.update({
            'type': 'transfer_learning',
            'family': 'EfficientNet',
            'pretrained_available': True,
            'input_size': 224,
        })
    elif 'densenet' in model_name:
        info.update({
            'type': 'transfer_learning',
            'family': 'DenseNet',
            'pretrained_available': True,
            'input_size': 224,
        })
    elif 'mobilenet' in model_name:
        info.update({
            'type': 'transfer_learning',
            'family': 'MobileNet',
            'pretrained_available': True,
            'input_size': 224,
            'notes': 'Lightweight, good for mobile/edge deployment',
        })
    
    return info


def list_available_models() -> List[str]:
    """
    List all available model architectures.
    
    Returns:
        List of model names
    """
    return SUPPORTED_MODELS.copy()


# =============================================================================
# Model Export
# =============================================================================

def export_to_torchscript(
    model: nn.Module,
    save_path: str,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cpu',
) -> str:
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model
        save_path: Path to save the exported model
        input_size: Example input size (B, C, H, W)
        device: Device for tracing
    
    Returns:
        Path to saved model
    """
    logger = get_logger()
    
    model = model.to(device)
    model.eval()
    
    # Create example input
    example_input = torch.randn(*input_size).to(device)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save
    traced_model.save(save_path)
    logger.info(f"Model exported to TorchScript: {save_path}")
    
    return save_path


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cpu',
    opset_version: int = 11,
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        save_path: Path to save the exported model
        input_size: Example input size (B, C, H, W)
        device: Device for export
        opset_version: ONNX opset version
    
    Returns:
        Path to saved model
    """
    logger = get_logger()
    
    model = model.to(device)
    model.eval()
    
    # Create example input
    example_input = torch.randn(*input_size).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        example_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )
    
    logger.info(f"Model exported to ONNX: {save_path}")
    
    return save_path


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    from utils import setup_logging, set_seed
    
    # Setup
    setup_logging(log_to_file=False)
    set_seed(42)
    logger = get_logger()
    
    print("=" * 60)
    print("Testing Models Module")
    print("=" * 60)
    
    # Test input
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224)
    
    # Test Baseline CNN
    print("\n1. Testing BaselineCNN...")
    baseline = BaselineCNN(num_classes=2)
    output = baseline(test_input)
    print(f"✓ BaselineCNN output shape: {output.shape}")
    print(f"  Parameters: {count_parameters(baseline):,}")
    
    # Test features extraction
    features = baseline.get_features(test_input)
    print(f"  Feature shape: {features.shape}")
    
    # Test Transfer Models
    print("\n2. Testing Transfer Learning Models...")
    
    test_models = ['resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v2']
    
    for model_name in test_models:
        print(f"\n   Testing {model_name}...")
        model = create_model(
            model_name=model_name,
            num_classes=2,
            pretrained=False,  # Don't download weights for testing
            freeze_backbone=False,
        )
        output = model(test_input)
        print(f"   ✓ {model_name} output shape: {output.shape}")
    
    # Test frozen backbone
    print("\n3. Testing frozen backbone...")
    model = create_model(
        model_name='resnet18',
        num_classes=2,
        pretrained=False,
        freeze_backbone=True,
    )
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    print(f"✓ Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Test unfreezing
    model.unfreeze_backbone()
    trainable_after = count_parameters(model, trainable_only=True)
    print(f"✓ After unfreeze: {trainable_after:,} / {total:,} ({100*trainable_after/total:.1f}%)")
    
    # Test model factory
    print("\n4. Testing model factory...")
    print(f"Available models: {list_available_models()}")
    
    # Test model info
    print("\n5. Model information:")
    for name in ['baseline_cnn', 'resnet50', 'efficientnet_b0']:
        info = get_model_info(name)
        print(f"   {name}: {info}")
    
    # Test export (without actually saving)
    print("\n6. Export functions available:")
    print("   - export_to_torchscript()")
    print("   - export_to_onnx()")
    
    print("\n" + "=" * 60)
    print("Models module tests complete!")
    print("=" * 60)
