#!/usr/bin/env python3
"""
ATICE-Net Testing Script
Advanced Copy-Move Forgery Detection Network
"""

import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import ATICENet
from dataloaders import CASIADataLoader
from utils.metrics import calculate_metrics, calculate_batch_metrics
from utils.visualization import visualize_results, save_predictions, create_comparison_grid
from utils.crf import apply_crf_postprocessing


def get_device(config):
    """Get device for testing"""
    if config['hardware']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif config['hardware']['device'] == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    return device


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ATICENet(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    return model


def evaluate_model(model, test_loader, device, config, save_results=True):
    """Evaluate model on test set"""
    model.eval()
    
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': []
    }
    
    all_predictions = []
    all_targets = []
    all_images = []
    
    results_dir = config['paths']['results_dir']
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Move data to device
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            file_names = batch['file']
            
            # Forward pass
            outputs = model(images, apply_crf=config['model']['use_crf'])
            predictions = outputs['output']
            
            # Calculate metrics
            batch_metrics = calculate_batch_metrics(
                predictions, masks, 
                threshold=config['evaluation']['threshold']
            )
            
            # Store metrics
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])
            
            # Store predictions and targets for detailed analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(masks.cpu().numpy())
            all_images.extend(images.cpu().numpy())
            
            # Save individual results
            if save_results:
                for i in range(images.shape[0]):
                    image = images[i]
                    pred_mask = predictions[i]
                    gt_mask = masks[i]
                    file_name = file_names[i]
                    
                    # Save visualization
                    save_predictions(
                        image, pred_mask, gt_mask,
                        save_dir=results_dir,
                        filename=f"{file_name}_result.png"
                    )
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics:
        avg_metrics[key] = np.mean(all_metrics[key])
        avg_metrics[f'{key}_std'] = np.std(all_metrics[key])
    
    return avg_metrics, all_predictions, all_targets, all_images


def test_single_image(model, image_path, device, config, save_path=None):
    """Test model on a single image"""
    import cv2
    from PIL import Image
    import torchvision.transforms as transforms
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image_size = tuple(config['dataset']['image_size'])
    image = cv2.resize(image, image_size)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor, apply_crf=config['model']['use_crf'])
        prediction = outputs['output']
    
    # Convert prediction to numpy
    pred_np = prediction.squeeze().cpu().numpy()
    
    # Visualize result
    if save_path:
        visualize_results(
            image_tensor, prediction,
            save_path=save_path,
            title=f"ATICE-Net Prediction: {os.path.basename(image_path)}"
        )
    
    return pred_np


def generate_comparison_grid(model, test_loader, device, config, num_samples=16):
    """Generate comparison grid of predictions"""
    model.eval()
    
    images = []
    pred_masks = []
    gt_masks = []
    
    print("Generating comparison grid...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating grid"):
            if len(images) >= num_samples:
                break
                
            batch_images = batch['image'].to(device)
            batch_masks = batch['mask'].to(device)
            
            outputs = model(batch_images, apply_crf=config['model']['use_crf'])
            batch_predictions = outputs['output']
            
            for i in range(batch_images.shape[0]):
                if len(images) >= num_samples:
                    break
                    
                images.append(batch_images[i])
                pred_masks.append(batch_predictions[i])
                gt_masks.append(batch_masks[i])
    
    # Create comparison grid
    grid_path = os.path.join(config['paths']['results_dir'], 'comparison_grid.png')
    create_comparison_grid(
        images, pred_masks, gt_masks,
        save_path=grid_path,
        max_images=num_samples
    )
    
    print(f"Comparison grid saved to: {grid_path}")


def save_evaluation_results(metrics, config, save_path=None):
    """Save evaluation results to JSON file"""
    if save_path is None:
        save_path = os.path.join(config['paths']['results_dir'], 'evaluation_results.json')
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'metrics': metrics
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to: {save_path}")


def print_evaluation_summary(metrics):
    """Print evaluation summary"""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'iou']:
        value = metrics[metric]
        std = metrics.get(f'{metric}_std', 0.0)
        print(f"{metric.upper():12}: {value:.4f} ± {std:.4f}")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Test ATICE-Net')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--single_image', type=str, default=None, help='Path to single image for testing')
    parser.add_argument('--no_save', action='store_true', help='Do not save results')
    parser.add_argument('--comparison_grid', action='store_true', help='Generate comparison grid')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get device
    device = get_device(config)
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Test single image if specified
    if args.single_image:
        print(f"Testing single image: {args.single_image}")
        
        if not os.path.exists(args.single_image):
            print(f"Error: Image file not found: {args.single_image}")
            return
        
        save_path = os.path.join(config['paths']['results_dir'], 'single_image_result.png')
        prediction = test_single_image(
            model, args.single_image, device, config, save_path
        )
        
        print(f"Prediction completed. Result saved to: {save_path}")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
        
        return
    
    # Initialize test data loader
    data_loader = CASIADataLoader(config)
    test_loader = data_loader.get_test_loader()
    
    # Evaluate model
    metrics, predictions, targets, images = evaluate_model(
        model, test_loader, device, config, 
        save_results=not args.no_save
    )
    
    # Print evaluation summary
    print_evaluation_summary(metrics)
    
    # Save evaluation results
    if not args.no_save:
        save_evaluation_results(metrics, config)
    
    # Generate comparison grid if requested
    if args.comparison_grid and not args.no_save:
        generate_comparison_grid(model, test_loader, device, config)
    
    print("\nTesting completed!")


if __name__ == '__main__':
    main() 