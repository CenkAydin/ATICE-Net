import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor to [0, 255] range
    
    Args:
        image_tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized image array [H, W, C]
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    # Convert to numpy
    image = image_tensor.detach().cpu().numpy()
    
    # Denormalize
    for i in range(3):
        image[i] = image[i] * std[i] + mean[i]
    
    # Clip to [0, 1] and convert to [0, 255]
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    
    # Transpose to [H, W, C]
    image = image.transpose(1, 2, 0)
    
    return image


def visualize_results(image, pred_mask, gt_mask=None, save_path=None, title=None):
    """
    Visualize prediction results
    
    Args:
        image: Input image tensor [C, H, W] or [B, C, H, W]
        pred_mask: Predicted mask tensor [1, H, W] or [B, 1, H, W]
        gt_mask: Ground truth mask tensor [1, H, W] or [B, 1, H, W] (optional)
        save_path: Path to save visualization (optional)
        title: Title for the plot (optional)
    """
    # Handle batch dimension
    if image.dim() == 4:
        image = image.squeeze(0)
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(0)
    if gt_mask is not None and gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(0)
    
    # Denormalize image
    image_np = denormalize_image(image)
    
    # Convert masks to numpy
    pred_np = pred_mask.detach().cpu().numpy().squeeze()
    if gt_mask is not None:
        gt_np = gt_mask.detach().cpu().numpy().squeeze()
    
    # Create figure
    if gt_mask is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(pred_np, cmap='hot', alpha=0.7)
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    # Overlay
    overlay = image_np.copy()
    pred_binary = (pred_np > 0.5).astype(np.uint8)
    overlay[pred_binary == 1] = [255, 0, 0]  # Red for forgery
    axes[2].imshow(overlay)
    axes[2].set_title('Prediction Overlay')
    axes[2].axis('off')
    
    # Ground truth (if available)
    if gt_mask is not None:
        axes[3].imshow(gt_np, cmap='hot', alpha=0.7)
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_predictions(image, pred_mask, gt_mask=None, save_dir='results', filename='prediction.png'):
    """
    Save prediction results to file
    
    Args:
        image: Input image tensor [C, H, W] or [B, C, H, W]
        pred_mask: Predicted mask tensor [1, H, W] or [B, 1, H, W]
        gt_mask: Ground truth mask tensor [1, H, W] or [B, 1, H, W] (optional)
        save_dir: Directory to save results
        filename: Filename for the saved image
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save visualization
    save_path = os.path.join(save_dir, filename)
    visualize_results(image, pred_mask, gt_mask, save_path=save_path)
    
    # Save individual components
    base_name = os.path.splitext(filename)[0]
    
    # Save predicted mask
    pred_np = pred_mask.detach().cpu().numpy().squeeze()
    pred_path = os.path.join(save_dir, f"{base_name}_pred.png")
    cv2.imwrite(pred_path, (pred_np * 255).astype(np.uint8))
    
    # Save ground truth mask if available
    if gt_mask is not None:
        gt_np = gt_mask.detach().cpu().numpy().squeeze()
        gt_path = os.path.join(save_dir, f"{base_name}_gt.png")
        cv2.imwrite(gt_path, (gt_np * 255).astype(np.uint8))


def create_comparison_grid(images, pred_masks, gt_masks=None, save_path=None, max_images=16):
    """
    Create a grid comparison of multiple predictions
    
    Args:
        images: List of image tensors
        pred_masks: List of predicted mask tensors
        gt_masks: List of ground truth mask tensors (optional)
        save_path: Path to save the grid
        max_images: Maximum number of images to display
    """
    n_images = min(len(images), max_images)
    
    if gt_masks is not None:
        fig, axes = plt.subplots(3, n_images, figsize=(2*n_images, 6))
    else:
        fig, axes = plt.subplots(2, n_images, figsize=(2*n_images, 4))
    
    for i in range(n_images):
        # Original image
        image_np = denormalize_image(images[i])
        axes[0, i].imshow(image_np)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Predicted mask
        pred_np = pred_masks[i].detach().cpu().numpy().squeeze()
        axes[1, i].imshow(pred_np, cmap='hot')
        axes[1, i].set_title(f'Pred {i+1}')
        axes[1, i].axis('off')
        
        # Ground truth (if available)
        if gt_masks is not None:
            gt_np = gt_masks[i].detach().cpu().numpy().squeeze()
            axes[2, i].imshow(gt_np, cmap='hot')
            axes[2, i].set_title(f'GT {i+1}')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(train_losses, val_losses, train_metrics=None, val_metrics=None, save_path=None):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: Dictionary of training metrics (optional)
        val_metrics: Dictionary of validation metrics (optional)
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Metrics curves
    if train_metrics and val_metrics:
        metrics = ['accuracy', 'f1', 'iou']
        for i, metric in enumerate(metrics):
            if metric in train_metrics and metric in val_metrics:
                row = (i + 1) // 2
                col = (i + 1) % 2
                axes[row, col].plot(train_metrics[metric], label=f'Train {metric.upper()}')
                axes[row, col].plot(val_metrics[metric], label=f'Val {metric.upper()}')
                axes[row, col].set_title(f'{metric.upper()} Score')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel(metric.upper())
                axes[row, col].legend()
                axes[row, col].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_attention_maps(image, attention_maps, save_path=None):
    """
    Visualize attention maps
    
    Args:
        image: Input image tensor [C, H, W]
        attention_maps: List of attention map tensors
        save_path: Path to save visualization
    """
    image_np = denormalize_image(image)
    
    n_maps = len(attention_maps)
    fig, axes = plt.subplots(2, n_maps, figsize=(3*n_maps, 6))
    
    # Original image
    for i in range(n_maps):
        axes[0, i].imshow(image_np)
        axes[0, i].set_title(f'Original')
        axes[0, i].axis('off')
    
    # Attention maps
    for i, att_map in enumerate(attention_maps):
        att_np = att_map.detach().cpu().numpy().squeeze()
        axes[1, i].imshow(att_np, cmap='hot')
        axes[1, i].set_title(f'Attention {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 