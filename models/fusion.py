import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Spatial attention module for feature refinement"""
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module for feature refinement"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class MultiResolutionFusion(nn.Module):
    """Multi-resolution feature fusion block with attention mechanisms"""
    def __init__(self, channels=[64, 128, 256, 512], fusion_dim=256):
        super(MultiResolutionFusion, self).__init__()
        self.channels = channels
        self.fusion_dim = fusion_dim
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], fusion_dim, 2, 2)
            for i in range(len(channels))
        ])
        
        # Attention modules
        self.spatial_attention = nn.ModuleList([
            SpatialAttention(fusion_dim) for _ in range(len(channels))
        ])
        
        self.channel_attention = nn.ModuleList([
            ChannelAttention(fusion_dim) for _ in range(len(channels))
        ])
        
        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_dim * len(channels), fusion_dim, 3, 1, 1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, fusion_dim, 3, 1, 1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.output_conv = nn.Conv2d(fusion_dim, fusion_dim, 1)
    
    def forward(self, features):
        """
        Args:
            features: List of features at different resolutions
                     [f1, f2, f3, f4] where f1 is highest resolution
        """
        # Upsample all features to the same resolution (highest)
        target_size = features[0].shape[-2:]
        upsampled_features = []
        
        for i, (feature, upsample_layer, spatial_att, channel_att) in enumerate(
            zip(features, self.upsample_layers, self.spatial_attention, self.channel_attention)
        ):
            # Upsample to target resolution
            if i == 0:
                upsampled = feature
            else:
                upsampled = upsample_layer(feature)
                if upsampled.shape[-2:] != target_size:
                    upsampled = F.interpolate(upsampled, size=target_size, mode='bilinear', align_corners=False)
            
            # Apply attention
            upsampled = spatial_att(upsampled)
            upsampled = channel_att(upsampled)
            upsampled_features.append(upsampled)
        
        # Concatenate and fuse
        fused = torch.cat(upsampled_features, dim=1)
        fused = self.fusion_conv(fused)
        output = self.output_conv(fused)
        
        return output 