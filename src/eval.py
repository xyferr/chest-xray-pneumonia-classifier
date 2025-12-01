"""
Evaluation module for Chest X-Ray Pneumonia Classification.

Includes:
- Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- Confusion matrix visualization
- ROC curve plotting
- Grad-CAM heatmap generation for model interpretability
- Classification report
- Error analysis
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_logger, get_device, load_checkpoint, ensure_dir
from dataset import (
    ChestXRayDataset,
    get_val_transforms,
    denormalize,
    CLASS_NAMES,
)
from models import create_model


# =============================================================================
# Grad-CAM Implementation
# =============================================================================

class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    
    Generates heatmaps showing which regions of the input image the model
    focuses on for its predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Layer to compute CAM for (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            # Clone to avoid inplace modification issues
            self.activations = output.clone().detach()
        
        def backward_hook(module, grad_input, grad_output):
            # Clone to avoid inplace modification issues
            self.gradients = grad_output[0].clone().detach()
        
        # Store handles for cleanup
        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index (None = use predicted class)
        
        Returns:
            Tuple of (heatmap array, predicted class, confidence)
        """
        self.model.eval()
        
        # Enable gradients for this operation
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get prediction
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        
        # Use target class or predicted class
        if target_class is None:
            target_class = pred_class
        
        # Backward pass - create a one-hot target
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Check if gradients were captured
        if self.gradients is None or self.activations is None:
            # Return a blank heatmap if hooks didn't capture data
            cam = np.zeros((7, 7), dtype=np.float32)
            return cam, pred_class, confidence
        
        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Handle edge cases
        if cam.size == 0:
            cam = np.zeros((7, 7), dtype=np.float32)
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam, pred_class, confidence
    
    def generate_heatmap(
        self,
        cam: np.ndarray,
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Resize CAM to original image size.
        
        Args:
            cam: CAM array
            original_size: (height, width) of original image
        
        Returns:
            Resized heatmap
        """
        from PIL import Image
        
        # Resize using PIL
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_resized = cam_pil.resize((original_size[1], original_size[0]), Image.BILINEAR)
        
        return np.array(cam_resized) / 255.0


def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Get the appropriate target layer for Grad-CAM based on model architecture.
    
    Args:
        model: PyTorch model
        model_name: Name of the model architecture
    
    Returns:
        Target layer module
    """
    model_name = model_name.lower()
    
    if model_name == 'baseline_cnn':
        return model.conv4[-3]  # Last conv layer before pooling
    elif 'resnet' in model_name:
        return model.backbone.layer4[-1]
    elif 'efficientnet' in model_name:
        return model.backbone.features[-1]
    elif 'densenet' in model_name:
        return model.backbone.features.denseblock4
    elif 'vgg' in model_name:
        return model.backbone.features[-1]
    elif 'mobilenet' in model_name:
        return model.backbone.features[-1]
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def visualize_gradcam(
    image: torch.Tensor,
    cam: np.ndarray,
    pred_class: int,
    true_class: Optional[int] = None,
    confidence: float = 0.0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Visualize Grad-CAM heatmap overlaid on original image.
    
    Args:
        image: Original image tensor [C, H, W]
        cam: Grad-CAM heatmap
        pred_class: Predicted class index
        true_class: True class index (optional)
        confidence: Prediction confidence
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Denormalize image
    img = denormalize(image.cpu())
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    
    # Resize CAM to image size
    from PIL import Image as PILImage
    cam_resized = np.array(
        PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
            (img.shape[1], img.shape[0]), PILImage.BILINEAR
        )
    ) / 255.0
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
    
    # Title with prediction info
    pred_label = CLASS_NAMES[pred_class]
    title = f'Pred: {pred_label} ({confidence:.1%})'
    if true_class is not None:
        true_label = CLASS_NAMES[true_class]
        color = 'green' if pred_class == true_class else 'red'
        title += f'\nTrue: {true_label}'
        axes[2].set_title(title, color=color)
    else:
        axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# Evaluation Functions
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    return_predictions: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Compute device
        return_predictions: Whether to return raw predictions
    
    Returns:
        Dictionary with evaluation metrics and predictions
    """
    model.eval()
    logger = get_logger()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(data_loader, desc='Evaluating'):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    # Log metrics
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"AUC-PR:    {metrics['auc_pr']:.4f}")
    
    results = {'metrics': metrics}
    
    if return_predictions:
        results['predictions'] = all_preds
        results['labels'] = all_labels
        results['probabilities'] = all_probs
    
    return results


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities [N, num_classes]
    
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', pos_label=1)
    recall = recall_score(y_true, y_pred, average='binary', pos_label=1)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
    
    # Probability of positive class (PNEUMONIA)
    y_prob_positive = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(y_true, y_prob_positive)
    except ValueError:
        auc_roc = 0.0
    
    # AUC-PR (Average Precision)
    try:
        auc_pr = average_precision_score(y_true, y_prob_positive)
    except ValueError:
        auc_pr = 0.0
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Specificity (for medical context)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'precision_normal': precision_per_class[0],
        'precision_pneumonia': precision_per_class[1],
        'recall_normal': recall_per_class[0],
        'recall_pneumonia': recall_per_class[1],
        'f1_normal': f1_per_class[0],
        'f1_pneumonia': f1_per_class[1],
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize values
        save_path: Path to save figure
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'shrink': 0.8},
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities for positive class
        save_path: Path to save figure
        figsize: Figure size
    """
    # Get probability of positive class
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        marker='o', color='red', s=100,
        label=f'Optimal Threshold = {optimal_threshold:.2f}'
    )
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities for positive class
        save_path: Path to save figure
        figsize: Figure size
    """
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AP = {ap:.4f})')
    
    # Baseline (random classifier)
    baseline = y_true.mean()
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline = {baseline:.2f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# Error Analysis
# =============================================================================

def analyze_errors(
    model: nn.Module,
    dataset: ChestXRayDataset,
    device: torch.device,
    num_samples: int = 10,
    save_dir: Optional[str] = None,
) -> Dict[str, List[int]]:
    """
    Analyze misclassified samples.
    
    Args:
        model: PyTorch model
        dataset: Dataset to analyze
        device: Compute device
        num_samples: Number of error samples to visualize
        save_dir: Directory to save error visualizations
    
    Returns:
        Dictionary with indices of false positives and false negatives
    """
    model.eval()
    logger = get_logger()
    
    false_positives = []  # Normal classified as Pneumonia
    false_negatives = []  # Pneumonia classified as Normal
    
    transform = get_val_transforms(224)
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Analyzing errors'):
            image, label = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            
            output = model(image_tensor)
            pred = output.argmax(dim=1).item()
            
            if pred != label:
                if label == 0 and pred == 1:
                    false_positives.append(idx)
                elif label == 1 and pred == 0:
                    false_negatives.append(idx)
    
    logger.info(f"\nError Analysis:")
    logger.info(f"False Positives (Normal → Pneumonia): {len(false_positives)}")
    logger.info(f"False Negatives (Pneumonia → Normal): {len(false_negatives)}")
    
    # Visualize errors if save_dir provided
    if save_dir:
        save_dir = ensure_dir(save_dir)
        
        # Visualize some false positives
        for i, idx in enumerate(false_positives[:num_samples]):
            image, label = dataset[idx]
            _save_error_visualization(
                model, image, label, device,
                str(save_dir / f'false_positive_{i}.png'),
                'False Positive'
            )
        
        # Visualize some false negatives
        for i, idx in enumerate(false_negatives[:num_samples]):
            image, label = dataset[idx]
            _save_error_visualization(
                model, image, label, device,
                str(save_dir / f'false_negative_{i}.png'),
                'False Negative'
            )
    
    return {
        'false_positives': false_positives,
        'false_negatives': false_negatives,
    }


def _save_error_visualization(
    model: nn.Module,
    image: torch.Tensor,
    label: int,
    device: torch.device,
    save_path: str,
    error_type: str,
) -> None:
    """Save visualization of a misclassified sample."""
    # Get prediction
    model.eval()
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probs[0, pred].item()
    
    # Denormalize and visualize
    img = denormalize(image)
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(
        f'{error_type}\n'
        f'True: {CLASS_NAMES[label]}, Pred: {CLASS_NAMES[pred]} ({confidence:.1%})',
        color='red'
    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Full Evaluation Pipeline
# =============================================================================

def run_full_evaluation(
    model_path: str,
    data_dir: str,
    output_dir: str,
    model_name: str = 'resnet50',
    device: Optional[torch.device] = None,
    generate_gradcam: bool = True,
    num_gradcam_samples: int = 10,
) -> Dict[str, Any]:
    """
    Run full evaluation pipeline.
    
    Args:
        model_path: Path to saved model checkpoint
        data_dir: Path to data directory
        output_dir: Output directory for results
        model_name: Model architecture name
        device: Compute device
        generate_gradcam: Whether to generate Grad-CAM visualizations
        num_gradcam_samples: Number of Grad-CAM samples to generate
    
    Returns:
        Dictionary with all evaluation results
    """
    logger = get_logger()
    
    if device is None:
        device = get_device()
    
    output_dir = ensure_dir(output_dir)
    
    logger.info("=" * 60)
    logger.info("Running Full Evaluation Pipeline")
    logger.info("=" * 60)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = create_model(model_name=model_name, num_classes=2, pretrained=False)
    checkpoint_info = load_checkpoint(model_path, model, device=device)
    model = model.to(device)
    model.eval()
    
    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = ChestXRayDataset(
        root_dir=data_dir,
        split='test',
        transform=get_val_transforms(224),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = evaluate_model(model, test_loader, device, return_predictions=True)
    
    # Generate and save plots
    logger.info("Generating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        save_path=str(output_dir / 'confusion_matrix.png'),
    )
    logger.info(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    
    # ROC curve
    plot_roc_curve(
        results['labels'],
        results['probabilities'],
        save_path=str(output_dir / 'roc_curve.png'),
    )
    logger.info(f"Saved ROC curve to {output_dir / 'roc_curve.png'}")
    
    # PR curve
    plot_precision_recall_curve(
        results['labels'],
        results['probabilities'],
        save_path=str(output_dir / 'pr_curve.png'),
    )
    logger.info(f"Saved PR curve to {output_dir / 'pr_curve.png'}")
    
    # Grad-CAM visualizations
    if generate_gradcam:
        logger.info("Generating Grad-CAM visualizations...")
        gradcam_dir = ensure_dir(output_dir / 'gradcam')
        
        try:
            target_layer = get_gradcam_target_layer(model, model_name)
            gradcam = GradCAM(model, target_layer)
            
            # Generate for random samples
            indices = np.random.choice(len(test_dataset), min(num_gradcam_samples, len(test_dataset)), replace=False)
            
            for i, idx in enumerate(indices):
                image, label = test_dataset[idx]
                image_tensor = image.unsqueeze(0).to(device)
                
                cam, pred_class, confidence = gradcam(image_tensor)
                
                visualize_gradcam(
                    image=image,
                    cam=cam,
                    pred_class=pred_class,
                    true_class=label,
                    confidence=confidence,
                    save_path=str(gradcam_dir / f'gradcam_{i}.png'),
                )
            
            logger.info(f"Saved {num_gradcam_samples} Grad-CAM visualizations to {gradcam_dir}")
        except Exception as e:
            logger.warning(f"Could not generate Grad-CAM: {e}")
    
    # Error analysis
    logger.info("Running error analysis...")
    error_dir = ensure_dir(output_dir / 'errors')
    errors = analyze_errors(
        model=model,
        dataset=test_dataset,
        device=device,
        num_samples=10,
        save_dir=str(error_dir),
    )
    
    # Save classification report
    report = classification_report(
        results['labels'],
        results['predictions'],
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    
    # Save metrics to file
    metrics_path = output_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("=" * 50 + "\n\n")
        
        for metric, value in results['metrics'].items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(classification_report(
            results['labels'],
            results['predictions'],
            target_names=CLASS_NAMES,
        ))
    
    logger.info(f"Saved metrics to {metrics_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    
    return {
        'metrics': results['metrics'],
        'classification_report': report,
        'errors': errors,
        'output_dir': str(output_dir),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command line interface for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Chest X-Ray Pneumonia Classifier')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/chest_xray', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation', help='Output directory')
    parser.add_argument('--model-name', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--no-gradcam', action='store_true', help='Skip Grad-CAM generation')
    parser.add_argument('--num-gradcam', type=int, default=10, help='Number of Grad-CAM samples')
    
    args = parser.parse_args()
    
    from utils import setup_logging
    setup_logging(log_to_file=False)
    
    run_full_evaluation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        generate_gradcam=not args.no_gradcam,
        num_gradcam_samples=args.num_gradcam,
    )


if __name__ == "__main__":
    main()
