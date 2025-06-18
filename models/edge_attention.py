import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeDetection(nn.Module):
    """Edge detection module using Sobel-like filters"""
    def __init__(self, in_channels):
        super(EdgeDetection, self).__init__()
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Register as buffer (not learnable parameters)
        self.register_buffer('sobel_x', sobel_x.repeat(in_channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(in_channels, 1, 1, 1))
        
        # Edge refinement
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1)
        )
    
    def forward(self, x):
        # Apply Sobel filters
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.size(1))
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.size(1))
        
        # Combine edges
        edges = torch.cat([edge_x, edge_y], dim=1)
        edges = self.edge_conv(edges)
        
        return edges


class EdgeAttention(nn.Module):
    """Edge-aware attention module for better segmentation boundaries"""
    def __init__(self, in_channels, attention_dim=64):
        super(EdgeAttention, self).__init__()
        self.in_channels = in_channels
        self.attention_dim = attention_dim
        
        # Edge detection
        self.edge_detector = EdgeDetection(in_channels)
        
        # Edge-aware attention
        self.edge_attention = nn.Sequential(
            nn.Conv2d(in_channels, attention_dim, 3, 1, 1),
            nn.BatchNorm2d(attention_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, attention_dim, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.feature_refinement = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1)
        )
        
        # Boundary enhancement
        self.boundary_enhancement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1)
        )
    
    def forward(self, x):
        # Detect edges
        edges = self.edge_detector(x)
        
        # Compute edge-aware attention
        attention = self.edge_attention(edges)
        
        # Apply attention to features
        attended_features = x * attention
        
        # Combine original and attended features
        combined = torch.cat([x, attended_features], dim=1)
        refined = self.feature_refinement(combined)
        
        # Enhance boundaries
        enhanced = self.boundary_enhancement(refined)
        
        return enhanced, edges, attention


class MultiScaleEdgeAttention(nn.Module):
    """Multi-scale edge attention for different resolution features"""
    def __init__(self, channels=[64, 128, 256, 512], attention_dim=64):
        super(MultiScaleEdgeAttention, self).__init__()
        self.channels = channels
        
        # Edge attention modules for each scale
        self.edge_attentions = nn.ModuleList([
            EdgeAttention(ch, attention_dim) for ch in channels
        ])
        
        # Cross-scale fusion
        self.cross_scale_fusion = nn.ModuleList([
            nn.Conv2d(channels[i] + (channels[i-1] if i > 0 else 0), channels[i], 1)
            for i in range(len(channels))
        ])
    
    def forward(self, features):
        """
        Args:
            features: List of features at different scales
        """
        enhanced_features = []
        edge_maps = []
        attention_maps = []
        
        for i, (feature, edge_att) in enumerate(zip(features, self.edge_attentions)):
            # Apply edge attention
            enhanced, edges, attention = edge_att(feature)
            
            # Cross-scale fusion
            if i > 0 and enhanced_features:
                # Upsample previous enhanced feature
                prev_enhanced = F.interpolate(
                    enhanced_features[-1], 
                    size=enhanced.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                # Fuse
                enhanced = torch.cat([enhanced, prev_enhanced], dim=1)
                enhanced = self.cross_scale_fusion[i](enhanced)
            
            enhanced_features.append(enhanced)
            edge_maps.append(edges)
            attention_maps.append(attention)
        
        return enhanced_features, edge_maps, attention_maps 