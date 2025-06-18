import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """Consistency loss for multi-view training"""
    def __init__(self, temperature=0.1):
        super(ConsistencyLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, pred1, pred2, mask1=None, mask2=None):
        """
        Args:
            pred1: First prediction [B, 1, H, W]
            pred2: Second prediction [B, 1, H, W] 
            mask1: First ground truth mask [B, 1, H, W] (optional)
            mask2: Second ground truth mask [B, 1, H, W] (optional)
        """
        # Normalize predictions
        pred1_norm = F.normalize(pred1.view(pred1.size(0), -1), dim=1)
        pred2_norm = F.normalize(pred2.view(pred2.size(0), -1), dim=1)
        
        # Compute cosine similarity
        similarity = torch.sum(pred1_norm * pred2_norm, dim=1) / self.temperature
        
        # Consistency loss (maximize similarity)
        consistency_loss = -torch.mean(similarity)
        
        # If masks are provided, add mask consistency
        if mask1 is not None and mask2 is not None:
            mask_similarity = F.cosine_similarity(
                mask1.view(mask1.size(0), -1),
                mask2.view(mask2.size(0), -1),
                dim=1
            )
            mask_consistency = -torch.mean(mask_similarity)
            consistency_loss = consistency_loss + mask_consistency
        
        return consistency_loss


class MultiScaleConsistencyLoss(nn.Module):
    """Multi-scale consistency loss"""
    def __init__(self, temperature=0.1, scales=[1.0, 0.5, 0.25]):
        super(MultiScaleConsistencyLoss, self).__init__()
        self.temperature = temperature
        self.scales = scales
        self.consistency_loss = ConsistencyLoss(temperature)
    
    def forward(self, pred1_list, pred2_list, mask1=None, mask2=None):
        """
        Args:
            pred1_list: List of predictions at different scales
            pred2_list: List of predictions at different scales
            mask1: Ground truth mask (optional)
            mask2: Ground truth mask (optional)
        """
        total_loss = 0.0
        
        for i, (pred1, pred2) in enumerate(zip(pred1_list, pred2_list)):
            scale_weight = self.scales[i] if i < len(self.scales) else 1.0
            
            # Resize masks if provided
            curr_mask1 = None
            curr_mask2 = None
            if mask1 is not None and mask2 is not None:
                curr_mask1 = F.interpolate(mask1, size=pred1.shape[-2:], mode='bilinear', align_corners=False)
                curr_mask2 = F.interpolate(mask2, size=pred2.shape[-2:], mode='bilinear', align_corners=False)
            
            # Compute consistency loss
            loss = self.consistency_loss(pred1, pred2, curr_mask1, curr_mask2)
            total_loss += scale_weight * loss
        
        return total_loss / len(pred1_list) 