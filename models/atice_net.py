import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import LightweightEncoder
from .fusion import MultiResolutionFusion
from .similarity import SimilarityComparison
from .edge_attention import MultiScaleEdgeAttention
from .decoder import Decoder


class Discriminator(nn.Module):
    """Discriminator for adversarial training"""
    def __init__(self, in_channels=4):  # 3 for image + 1 for mask
        super(Discriminator, self).__init__()
        
        self.features = nn.Sequential(
            # 64
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.features(x)
        validity = self.classifier(features)
        return validity


class ATICENet(nn.Module):
    """ATICE-Net: Advanced Copy-Move Forgery Detection Network"""
    def __init__(self, config):
        super(ATICENet, self).__init__()
        
        # Configuration
        self.config = config
        encoder_channels = config['model']['encoder_channels']
        decoder_channels = config['model']['decoder_channels']
        similarity_dim = config['model']['similarity_dim']
        edge_attention_dim = config['model']['edge_attention_dim']
        self.use_crf = config['model']['use_crf']
        self.use_adversarial = config['model']['use_adversarial']
        
        # Main components
        self.encoder = LightweightEncoder(in_channels=3, channels=encoder_channels)
        self.fusion = MultiResolutionFusion(channels=encoder_channels)
        self.similarity = SimilarityComparison(encoder_channels[-1], similarity_dim)
        self.edge_attention = MultiScaleEdgeAttention(encoder_channels, edge_attention_dim)
        self.decoder = Decoder(encoder_channels, decoder_channels)
        
        # Adversarial discriminator
        if self.use_adversarial:
            self.discriminator = Discriminator()
        
        # CRF post-processing (optional)
        if self.use_crf:
            self.crf_params = {
                'num_iterations': 10,
                'theta_alpha': 160,
                'theta_beta': 3,
                'theta_gamma': 3,
                'spatial_ker_weight': 3,
                'bilateral_ker_weight': 5,
                'compatibility': 10
            }
    
    def forward(self, x, apply_crf=False):
        """
        Args:
            x: Input image tensor [B, 3, H, W]
            apply_crf: Whether to apply CRF post-processing
        """
        # Encoder
        encoder_features = self.encoder(x)
        
        # Multi-resolution fusion
        fused_features = self.fusion(encoder_features)
        
        # Similarity comparison
        similarity_features = self.similarity(fused_features)
        
        # Edge attention
        edge_enhanced_features, edge_maps, attention_maps = self.edge_attention(encoder_features)
        
        # Decoder
        output, supervision_outputs = self.decoder(
            edge_enhanced_features, 
            similarity_features, 
            edge_maps[0] if edge_maps else None
        )
        
        # CRF post-processing (optional)
        if apply_crf and self.use_crf:
            output = self.apply_crf(x, output)
        
        return {
            'output': output,
            'supervision_outputs': supervision_outputs,
            'edge_maps': edge_maps,
            'attention_maps': attention_maps,
            'similarity_features': similarity_features,
            'encoder_features': encoder_features
        }
    
    def apply_crf(self, image, mask):
        """Apply Conditional Random Field post-processing"""
        try:
            import pydensecrf.densecrf as dcrf
            import numpy as np
            
            # Convert to numpy
            image_np = image.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            processed_masks = []
            
            for i in range(image_np.shape[0]):
                img = image_np[i].transpose(1, 2, 0)
                prob = mask_np[i, 0]
                
                # Create CRF
                d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)
                
                # Set unary potentials
                U = np.zeros((2, img.shape[0], img.shape[1]), dtype=np.float32)
                U[0, :, :] = -np.log(prob + 1e-8)
                U[1, :, :] = -np.log(1 - prob + 1e-8)
                d.setUnaryEnergy(U)
                
                # Set pairwise potentials
                d.addPairwiseGaussian(sxy=self.crf_params['theta_alpha'], compat=self.crf_params['compatibility'])
                d.addPairwiseBilateral(sxy=self.crf_params['theta_beta'], srgb=self.crf_params['theta_gamma'], 
                                     rgbim=img, compat=self.crf_params['compatibility'])
                
                # Inference
                Q = d.inference(self.crf_params['num_iterations'])
                processed_mask = np.array(Q)[1].reshape(img.shape[0], img.shape[1])
                processed_masks.append(processed_mask)
            
            # Convert back to tensor
            processed_tensor = torch.from_numpy(np.array(processed_masks)).unsqueeze(1).float()
            return processed_tensor.to(image.device)
            
        except ImportError:
            print("Warning: pydensecrf not available. Skipping CRF post-processing.")
            return mask
    
    def get_discriminator_output(self, image, mask):
        """Get discriminator output for adversarial training"""
        if not self.use_adversarial:
            return None
        
        # Concatenate image and mask
        input_d = torch.cat([image, mask], dim=1)
        return self.discriminator(input_d)
    
    def count_parameters(self):
        """Count total parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'total_millions': total_params / 1e6,
            'trainable_millions': trainable_params / 1e6
        } 