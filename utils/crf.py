import numpy as np
import torch
try:
    import pydensecrf.densecrf as dcrf
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("Warning: pydensecrf not available. CRF post-processing will be disabled.")


def apply_crf_postprocessing(image, mask, crf_params=None):
    """
    Apply Conditional Random Field post-processing to refine segmentation masks
    
    Args:
        image: Input image tensor [B, C, H, W] or numpy array [H, W, C]
        mask: Predicted mask tensor [B, 1, H, W] or numpy array [H, W]
        crf_params: CRF parameters dictionary
    
    Returns:
        Refined mask tensor or numpy array
    """
    if not CRF_AVAILABLE:
        print("CRF not available, returning original mask")
        return mask
    
    # Default CRF parameters
    if crf_params is None:
        crf_params = {
            'num_iterations': 10,
            'theta_alpha': 160,
            'theta_beta': 3,
            'theta_gamma': 3,
            'spatial_ker_weight': 3,
            'bilateral_ker_weight': 5,
            'compatibility': 10
        }
    
    # Convert tensors to numpy
    if torch.is_tensor(image):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image
    
    if torch.is_tensor(mask):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    
    # Handle batch dimension
    if image_np.ndim == 4:
        batch_size = image_np.shape[0]
        refined_masks = []
        
        for i in range(batch_size):
            img = image_np[i].transpose(1, 2, 0)  # [H, W, C]
            prob = mask_np[i, 0] if mask_np.ndim == 4 else mask_np[i]
            
            refined_mask = _apply_crf_single(img, prob, crf_params)
            refined_masks.append(refined_mask)
        
        refined_masks = np.array(refined_masks)
        
        # Convert back to tensor if input was tensor
        if torch.is_tensor(mask):
            refined_masks = torch.from_numpy(refined_masks).float()
            if mask.ndim == 4:
                refined_masks = refined_masks.unsqueeze(1)  # Add channel dimension
        
        return refined_masks
    else:
        # Single image
        refined_mask = _apply_crf_single(image_np, mask_np, crf_params)
        
        # Convert back to tensor if input was tensor
        if torch.is_tensor(mask):
            refined_mask = torch.from_numpy(refined_mask).float()
            if mask.ndim == 3:
                refined_mask = refined_mask.unsqueeze(0)  # Add batch dimension
        
        return refined_mask


def _apply_crf_single(image, prob, crf_params):
    """
    Apply CRF to a single image
    
    Args:
        image: Input image [H, W, C]
        prob: Probability mask [H, W]
        crf_params: CRF parameters
    
    Returns:
        Refined mask [H, W]
    """
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Create CRF
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
    
    # Set unary potentials
    U = np.zeros((2, image.shape[0], image.shape[1]), dtype=np.float32)
    U[0, :, :] = -np.log(prob + 1e-8)
    U[1, :, :] = -np.log(1 - prob + 1e-8)
    d.setUnaryEnergy(U)
    
    # Set pairwise potentials
    d.addPairwiseGaussian(
        sxy=crf_params['theta_alpha'], 
        compat=crf_params['compatibility']
    )
    d.addPairwiseBilateral(
        sxy=crf_params['theta_beta'], 
        srgb=crf_params['theta_gamma'], 
        rgbim=image, 
        compat=crf_params['compatibility']
    )
    
    # Inference
    Q = d.inference(crf_params['num_iterations'])
    refined_mask = np.array(Q)[1].reshape(image.shape[0], image.shape[1])
    
    return refined_mask


def apply_crf_batch(images, masks, crf_params=None):
    """
    Apply CRF to a batch of images and masks
    
    Args:
        images: Batch of images [B, C, H, W]
        masks: Batch of masks [B, 1, H, W]
        crf_params: CRF parameters
    
    Returns:
        Refined masks [B, 1, H, W]
    """
    return apply_crf_postprocessing(images, masks, crf_params)


def create_crf_params(iterations=10, theta_alpha=160, theta_beta=3, theta_gamma=3, compatibility=10):
    """
    Create CRF parameters dictionary
    
    Args:
        iterations: Number of CRF iterations
        theta_alpha: Spatial kernel weight
        theta_beta: Bilateral kernel weight (spatial)
        theta_gamma: Bilateral kernel weight (appearance)
        compatibility: Compatibility transform weight
    
    Returns:
        CRF parameters dictionary
    """
    return {
        'num_iterations': iterations,
        'theta_alpha': theta_alpha,
        'theta_beta': theta_beta,
        'theta_gamma': theta_gamma,
        'spatial_ker_weight': theta_alpha,
        'bilateral_ker_weight': theta_beta,
        'compatibility': compatibility
    }


def optimize_crf_params(image, mask, gt_mask, param_ranges=None):
    """
    Optimize CRF parameters for a given image and ground truth
    
    Args:
        image: Input image [H, W, C]
        mask: Predicted mask [H, W]
        gt_mask: Ground truth mask [H, W]
        param_ranges: Parameter ranges to search
    
    Returns:
        Best CRF parameters
    """
    if param_ranges is None:
        param_ranges = {
            'theta_alpha': [80, 160, 240],
            'theta_beta': [1, 3, 5],
            'theta_gamma': [1, 3, 5],
            'iterations': [5, 10, 15]
        }
    
    best_params = None
    best_iou = 0.0
    
    for theta_alpha in param_ranges['theta_alpha']:
        for theta_beta in param_ranges['theta_beta']:
            for theta_gamma in param_ranges['theta_gamma']:
                for iterations in param_ranges['iterations']:
                    params = create_crf_params(
                        iterations=iterations,
                        theta_alpha=theta_alpha,
                        theta_beta=theta_beta,
                        theta_gamma=theta_gamma
                    )
                    
                    refined_mask = _apply_crf_single(image, mask, params)
                    
                    # Calculate IoU
                    intersection = np.logical_and(refined_mask > 0.5, gt_mask > 0.5).sum()
                    union = np.logical_or(refined_mask > 0.5, gt_mask > 0.5).sum()
                    iou = intersection / union if union > 0 else 0.0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_params = params
    
    return best_params, best_iou 