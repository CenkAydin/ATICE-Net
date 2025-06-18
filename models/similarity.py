import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnablePositionalOffset(nn.Module):
    """Learnable positional offsets for similarity comparison"""
    def __init__(self, max_offset=16, num_offsets=8):
        super(LearnablePositionalOffset, self).__init__()
        self.max_offset = max_offset
        self.num_offsets = num_offsets
        
        # Learnable offset parameters
        self.offset_x = nn.Parameter(torch.randn(num_offsets) * 0.1)
        self.offset_y = nn.Parameter(torch.randn(num_offsets) * 0.1)
        
        # Scale parameter
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate offset grid
        offsets = torch.stack([
            self.offset_x * self.scale * self.max_offset,
            self.offset_y * self.scale * self.max_offset
        ], dim=1)  # [num_offsets, 2]
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=torch.float32),
            torch.arange(W, device=x.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Apply offsets
        offset_features = []
        for i in range(self.num_offsets):
            offset_x = offsets[i, 0]
            offset_y = offsets[i, 1]
            
            # Sample with offset
            sample_x = grid_x + offset_x
            sample_y = grid_y + offset_y
            
            # Normalize to [-1, 1]
            sample_x = 2 * sample_x / (W - 1) - 1
            sample_y = 2 * sample_y / (H - 1) - 1
            
            grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
            grid = grid.repeat(B, 1, 1, 1)
            
            # Sample features
            offset_feat = F.grid_sample(x, grid, mode='bilinear', align_corners=False, padding_mode='reflection')
            offset_features.append(offset_feat)
        
        return torch.cat(offset_features, dim=1)


class LocalSimilarityModule(nn.Module):
    """Local similarity comparison module"""
    def __init__(self, in_channels, similarity_dim=128, patch_size=7):
        super(LocalSimilarityModule, self).__init__()
        self.patch_size = patch_size
        self.similarity_dim = similarity_dim
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, similarity_dim, 1),
            nn.BatchNorm2d(similarity_dim),
            nn.ReLU(inplace=True)
        )
        
        # Local similarity computation
        self.local_conv = nn.Conv2d(similarity_dim, similarity_dim, patch_size, padding=patch_size//2, groups=similarity_dim)
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(similarity_dim * 2, similarity_dim, 1),
            nn.BatchNorm2d(similarity_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Project features
        proj_features = self.projection(x)
        
        # Compute local similarity
        local_features = self.local_conv(proj_features)
        
        # Concatenate original and local features
        output = torch.cat([proj_features, local_features], dim=1)
        output = self.output_conv(output)
        
        return output


class DualScaleSimilarity(nn.Module):
    """Dual-scale similarity comparison module"""
    def __init__(self, in_channels, similarity_dim=128):
        super(DualScaleSimilarity, self).__init__()
        self.similarity_dim = similarity_dim
        
        # Large scale similarity
        self.large_scale = LocalSimilarityModule(in_channels, similarity_dim, patch_size=15)
        
        # Small scale similarity
        self.small_scale = LocalSimilarityModule(in_channels, similarity_dim, patch_size=7)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(similarity_dim * 2, similarity_dim, 1),
            nn.BatchNorm2d(similarity_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Compute dual-scale similarities
        large_sim = self.large_scale(x)
        small_sim = self.small_scale(x)
        
        # Fuse
        fused = torch.cat([large_sim, small_sim], dim=1)
        output = self.fusion(fused)
        
        return output


class SimilarityComparison(nn.Module):
    """LCSCC-style similarity comparison with dual-scale fusion and learnable offsets"""
    def __init__(self, in_channels, similarity_dim=128):
        super(SimilarityComparison, self).__init__()
        self.similarity_dim = similarity_dim
        
        # Learnable positional offsets
        self.positional_offsets = LearnablePositionalOffset()
        
        # Dual-scale similarity
        self.dual_scale_sim = DualScaleSimilarity(in_channels, similarity_dim)
        
        # Global context
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, similarity_dim, 1),
            nn.BatchNorm2d(similarity_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Conv2d(similarity_dim * 2, similarity_dim, 3, 1, 1),
            nn.BatchNorm2d(similarity_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(similarity_dim, similarity_dim, 1)
        )
    
    def forward(self, x):
        # Apply learnable positional offsets
        offset_features = self.positional_offsets(x)
        
        # Compute dual-scale similarity
        similarity_features = self.dual_scale_sim(x)
        
        # Global context
        global_context = self.global_context(x)
        global_context = F.interpolate(global_context, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Combine similarity and global context
        combined = torch.cat([similarity_features, global_context], dim=1)
        output = self.final_fusion(combined)
        
        return output 