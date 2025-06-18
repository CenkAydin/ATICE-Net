import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU)
    
    Args:
        pred: Predicted mask [B, 1, H, W] or [H, W]
        target: Ground truth mask [B, 1, H, W] or [H, W]
        threshold: Threshold for binarization
    
    Returns:
        IoU score
    """
    # Ensure tensors are on CPU and convert to numpy
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Binarize predictions
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > threshold).astype(np.uint8)
    
    # Flatten
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_flat, target_flat).sum()
    union = np.logical_or(pred_flat, target_flat).sum()
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def calculate_f1_score(pred, target, threshold=0.5):
    """
    Calculate F1-score
    
    Args:
        pred: Predicted mask [B, 1, H, W] or [H, W]
        target: Ground truth mask [B, 1, H, W] or [H, W]
        threshold: Threshold for binarization
    
    Returns:
        F1-score
    """
    # Ensure tensors are on CPU and convert to numpy
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Binarize predictions
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > threshold).astype(np.uint8)
    
    # Flatten
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    
    # Calculate F1-score
    return f1_score(target_flat, pred_flat, zero_division=0)


def calculate_accuracy(pred, target, threshold=0.5):
    """
    Calculate accuracy
    
    Args:
        pred: Predicted mask [B, 1, H, W] or [H, W]
        target: Ground truth mask [B, 1, H, W] or [H, W]
        threshold: Threshold for binarization
    
    Returns:
        Accuracy score
    """
    # Ensure tensors are on CPU and convert to numpy
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Binarize predictions
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > threshold).astype(np.uint8)
    
    # Flatten
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    
    # Calculate accuracy
    return accuracy_score(target_flat, pred_flat)


def calculate_precision(pred, target, threshold=0.5):
    """
    Calculate precision
    
    Args:
        pred: Predicted mask [B, 1, H, W] or [H, W]
        target: Ground truth mask [B, 1, H, W] or [H, W]
        threshold: Threshold for binarization
    
    Returns:
        Precision score
    """
    # Ensure tensors are on CPU and convert to numpy
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Binarize predictions
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > threshold).astype(np.uint8)
    
    # Flatten
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    
    # Calculate precision
    return precision_score(target_flat, pred_flat, zero_division=0)


def calculate_recall(pred, target, threshold=0.5):
    """
    Calculate recall
    
    Args:
        pred: Predicted mask [B, 1, H, W] or [H, W]
        target: Ground truth mask [B, 1, H, W] or [H, W]
        threshold: Threshold for binarization
    
    Returns:
        Recall score
    """
    # Ensure tensors are on CPU and convert to numpy
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Binarize predictions
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > threshold).astype(np.uint8)
    
    # Flatten
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    
    # Calculate recall
    return recall_score(target_flat, pred_flat, zero_division=0)


def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate all metrics
    
    Args:
        pred: Predicted mask [B, 1, H, W] or [H, W]
        target: Ground truth mask [B, 1, H, W] or [H, W]
        threshold: Threshold for binarization
    
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    metrics['accuracy'] = calculate_accuracy(pred, target, threshold)
    metrics['precision'] = calculate_precision(pred, target, threshold)
    metrics['recall'] = calculate_recall(pred, target, threshold)
    metrics['f1'] = calculate_f1_score(pred, target, threshold)
    metrics['iou'] = calculate_iou(pred, target, threshold)
    
    return metrics


def calculate_batch_metrics(pred_batch, target_batch, threshold=0.5):
    """
    Calculate metrics for a batch of predictions
    
    Args:
        pred_batch: Batch of predicted masks [B, 1, H, W]
        target_batch: Batch of ground truth masks [B, 1, H, W]
        threshold: Threshold for binarization
    
    Returns:
        Dictionary containing average metrics
    """
    batch_size = pred_batch.shape[0]
    batch_metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'iou': 0.0
    }
    
    for i in range(batch_size):
        pred = pred_batch[i]
        target = target_batch[i]
        
        metrics = calculate_metrics(pred, target, threshold)
        
        for key in batch_metrics:
            batch_metrics[key] += metrics[key]
    
    # Average over batch
    for key in batch_metrics:
        batch_metrics[key] /= batch_size
    
    return batch_metrics 