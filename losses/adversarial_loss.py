import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    """Adversarial loss for discriminator-based regularization"""
    def __init__(self, gan_mode='vanilla'):
        super(AdversarialLoss, self).__init__()
        self.gan_mode = gan_mode
        
        if gan_mode == 'vanilla':
            self.criterion = nn.BCELoss()
        elif gan_mode == 'lsgan':
            self.criterion = nn.MSELoss()
        elif gan_mode == 'wgan':
            self.criterion = None  # WGAN uses different loss
        else:
            raise ValueError(f"Unsupported GAN mode: {gan_mode}")
    
    def forward(self, discriminator_output, target_is_real):
        """
        Args:
            discriminator_output: Output from discriminator
            target_is_real: Whether the target is real (True) or fake (False)
        """
        if self.gan_mode == 'vanilla':
            # Create target labels
            if target_is_real:
                target = torch.ones_like(discriminator_output)
            else:
                target = torch.zeros_like(discriminator_output)
            
            return self.criterion(discriminator_output, target)
        
        elif self.gan_mode == 'lsgan':
            # Create target labels
            if target_is_real:
                target = torch.ones_like(discriminator_output)
            else:
                target = torch.zeros_like(discriminator_output)
            
            return self.criterion(discriminator_output, target)
        
        elif self.gan_mode == 'wgan':
            # WGAN loss
            if target_is_real:
                return -torch.mean(discriminator_output)
            else:
                return torch.mean(discriminator_output)
        
        else:
            raise ValueError(f"Unsupported GAN mode: {self.gan_mode}")


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for more stable adversarial training"""
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
        self.criterion = nn.L1Loss()
    
    def forward(self, real_features, fake_features):
        """
        Args:
            real_features: Features from real images
            fake_features: Features from generated images
        """
        if isinstance(real_features, list):
            # Multi-scale features
            total_loss = 0.0
            for real_feat, fake_feat in zip(real_features, fake_features):
                total_loss += self.criterion(real_feat, fake_feat)
            return total_loss / len(real_features)
        else:
            return self.criterion(real_features, fake_features)


class GradientPenaltyLoss(nn.Module):
    """Gradient penalty loss for WGAN-GP"""
    def __init__(self, lambda_gp=10.0):
        super(GradientPenaltyLoss, self).__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, discriminator, real_samples, fake_samples):
        """
        Args:
            discriminator: Discriminator model
            real_samples: Real samples
            fake_samples: Fake samples
        """
        batch_size = real_samples.size(0)
        
        # Create interpolated samples
        alpha = torch.rand(batch_size, 1, 1, 1).to(real_samples.device)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Get discriminator output for interpolated samples
        d_interpolated = discriminator(interpolated)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.lambda_gp * gradient_penalty 