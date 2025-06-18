import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """Partial Convolution for efficient feature extraction"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(PartialConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FastBlock(nn.Module):
    """Fast convolution block for lightweight processing"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(FastBlock, self).__init__()
        padding = kernel_size // 2
        
        self.conv1 = PartialConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = PartialConv2d(out_channels, out_channels, kernel_size, 1, padding)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return F.relu(out)


class LightweightEncoder(nn.Module):
    """Lightweight CNN-based encoder inspired by FasterNet"""
    def __init__(self, in_channels=3, channels=[64, 128, 256, 512]):
        super(LightweightEncoder, self).__init__()
        self.channels = channels
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            PartialConv2d(in_channels, channels[0], 7, 2, 3),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Encoder stages
        self.stage1 = self._make_stage(channels[0], channels[0], 2)
        self.stage2 = self._make_stage(channels[0], channels[1], 2)
        self.stage3 = self._make_stage(channels[1], channels[2], 2)
        self.stage4 = self._make_stage(channels[2], channels[3], 2)
        
        # Feature pyramid
        self.fpn = nn.ModuleList([
            nn.Conv2d(channels[0], channels[0], 1),
            nn.Conv2d(channels[1], channels[1], 1),
            nn.Conv2d(channels[2], channels[2], 1),
            nn.Conv2d(channels[3], channels[3], 1)
        ])
        
    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(FastBlock(in_channels, out_channels, stride=2))
        for _ in range(num_blocks - 1):
            layers.append(FastBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder path
        x1 = self.conv1(x)  # 1/4
        x2 = self.stage1(x1)  # 1/8
        x3 = self.stage2(x2)  # 1/16
        x4 = self.stage3(x3)  # 1/32
        x5 = self.stage4(x4)  # 1/64
        
        # Apply FPN
        features = []
        features.append(self.fpn[0](x2))  # 1/8
        features.append(self.fpn[1](x3))  # 1/16
        features.append(self.fpn[2](x4))  # 1/32
        features.append(self.fpn[3](x5))  # 1/64
        
        return features 