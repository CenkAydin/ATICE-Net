import torch
import torch.nn as nn
import torch.nn.functional as F
from .consistency_loss import ConsistencyLoss, MultiScaleConsistencyLoss
from .adversarial_loss import AdversarialLoss, FeatureMatchingLoss, GradientPenaltyLoss


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted mask [B, 1, H, W]
            target: Ground truth mask [B, 1, H, W]
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class EdgeLoss(nn.Module):
    """Edge-aware loss for better boundary detection"""
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.criterion = nn.BCELoss()
    
    def forward(self, pred, target, edge_map=None):
        """
        Args:
            pred: Predicted mask [B, 1, H, W]
            target: Ground truth mask [B, 1, H, W]
            edge_map: Edge map [B, 1, H, W] (optional)
        """
        if edge_map is not None:
            # Weight the loss by edge map
            edge_weight = edge_map + 1.0  # Add 1 to avoid zero weights
            weighted_pred = pred * edge_weight
            weighted_target = target * edge_weight
            return self.criterion(weighted_pred, weighted_target)
        else:
            return self.criterion(pred, target)


class TotalLoss(nn.Module):
    """Total loss combining all loss components"""
    def __init__(self, config):
        super(TotalLoss, self).__init__()
        self.config = config
        
        # Loss weights
        self.loss_weights = config['loss_weights']
        
        # Individual loss functions
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.edge_loss = EdgeLoss()
        self.consistency_loss = ConsistencyLoss()
        self.multi_scale_consistency = MultiScaleConsistencyLoss()
        
        # Adversarial losses
        if config['model']['use_adversarial']:
            self.adversarial_loss = AdversarialLoss(gan_mode='lsgan')
            self.feature_matching_loss = FeatureMatchingLoss()
            self.gradient_penalty_loss = GradientPenaltyLoss()
    
    def forward(self, predictions, targets, model_outputs=None, discriminator=None):
        """
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            model_outputs: Additional model outputs (optional)
            discriminator: Discriminator model (optional)
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Extract predictions
        if isinstance(predictions, dict):
            main_pred = predictions['output']
            supervision_preds = predictions.get('supervision_outputs', [])
            edge_maps = predictions.get('edge_maps', [])
            attention_maps = predictions.get('attention_maps', [])
        else:
            main_pred = predictions
            supervision_preds = []
            edge_maps = []
            attention_maps = []
        
        # Extract targets
        if isinstance(targets, dict):
            main_target = targets['mask']
            edge_target = targets.get('edge', None)
        else:
            main_target = targets
            edge_target = None
        
        # BCE Loss
        bce_loss = self.bce_loss(main_pred, main_target)
        total_loss += self.loss_weights['bce'] * bce_loss
        loss_dict['bce'] = bce_loss.item()
        
        # Dice Loss
        dice_loss = self.dice_loss(main_pred, main_target)
        total_loss += self.loss_weights['dice'] * dice_loss
        loss_dict['dice'] = dice_loss.item()
        
        # Edge Loss
        if edge_maps and len(edge_maps) > 0:
            edge_loss = self.edge_loss(main_pred, main_target, edge_maps[0])
            total_loss += self.loss_weights['edge'] * edge_loss
            loss_dict['edge'] = edge_loss.item()
        
        # Multi-scale supervision loss
        if supervision_preds:
            supervision_loss = 0.0
            for i, pred in enumerate(supervision_preds):
                # Resize target to match prediction size
                target_resized = F.interpolate(main_target, size=pred.shape[-2:], mode='nearest')
                supervision_loss += self.bce_loss(pred, target_resized)
            supervision_loss /= len(supervision_preds)
            total_loss += 0.1 * supervision_loss  # Small weight for supervision
            loss_dict['supervision'] = supervision_loss.item()
        
        # Consistency Loss (if multi-view data is available)
        if 'consistency' in self.loss_weights and model_outputs is not None:
            if 'view1_pred' in model_outputs and 'view2_pred' in model_outputs:
                consistency_loss = self.consistency_loss(
                    model_outputs['view1_pred'], 
                    model_outputs['view2_pred']
                )
                total_loss += self.loss_weights['consistency'] * consistency_loss
                loss_dict['consistency'] = consistency_loss.item()
        
        # Adversarial Loss
        if self.config['model']['use_adversarial'] and discriminator is not None:
            # Generator loss (fool discriminator)
            fake_validity = discriminator(torch.cat([model_outputs.get('image', main_pred), main_pred], dim=1))
            adversarial_loss = self.adversarial_loss(fake_validity, False)
            total_loss += self.loss_weights['adversarial'] * adversarial_loss
            loss_dict['adversarial'] = adversarial_loss.item()
            
            # Feature matching loss
            if 'real_features' in model_outputs and 'fake_features' in model_outputs:
                feature_loss = self.feature_matching_loss(
                    model_outputs['real_features'],
                    model_outputs['fake_features']
                )
                total_loss += 0.1 * feature_loss
                loss_dict['feature_matching'] = feature_loss.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def compute_discriminator_loss(self, discriminator, real_samples, fake_samples):
        """Compute discriminator loss for adversarial training"""
        if not self.config['model']['use_adversarial']:
            return 0.0, {}
        
        # Real samples
        real_validity = discriminator(real_samples)
        real_loss = self.adversarial_loss(real_validity, True)
        
        # Fake samples
        fake_validity = discriminator(fake_samples)
        fake_loss = self.adversarial_loss(fake_validity, False)
        
        # Gradient penalty
        gradient_penalty = self.gradient_penalty_loss(discriminator, real_samples, fake_samples)
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss + gradient_penalty
        
        return d_loss, {
            'd_real': real_loss.item(),
            'd_fake': fake_loss.item(),
            'd_gp': gradient_penalty.item(),
            'd_total': d_loss.item()
        } 