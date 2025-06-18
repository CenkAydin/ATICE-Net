import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Decoder block with skip connections"""
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super(DecoderBlock, self).__init__()
        
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Skip connection fusion
        if skip_channels > 0:
            self.skip_fusion = nn.Sequential(
                nn.Conv2d(out_channels + skip_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # Final refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1)
        )
    
    def forward(self, x, skip=None):
        # Upsample
        x = self.up_conv(x)
        
        # Skip connection
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.skip_fusion(x)
        
        # Refinement
        x = self.refinement(x)
        
        return x


class Decoder(nn.Module):
    """Decoder for ATICE-Net with multi-scale feature integration"""
    def __init__(self, encoder_channels=[64, 128, 256, 512], decoder_channels=[256, 128, 64, 32]):
        super(Decoder, self).__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[-(i+1)] if i == 0 else decoder_channels[i-1]
            out_ch = decoder_channels[i]
            skip_ch = encoder_channels[-(i+2)] if i < len(encoder_channels)-1 else 0
            
            self.decoder_blocks.append(DecoderBlock(in_ch, out_ch, skip_ch))
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], 3, 1, 1),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], 1, 1),
            nn.Sigmoid()
        )
        
        # Multi-scale supervision
        self.supervision_layers = nn.ModuleList([
            nn.Conv2d(ch, 1, 1) for ch in decoder_channels
        ])
    
    def forward(self, encoder_features, similarity_features=None, edge_features=None):
        """
        Args:
            encoder_features: List of encoder features [f1, f2, f3, f4]
            similarity_features: Optional similarity features
            edge_features: Optional edge features
        """
        # Reverse encoder features for decoder
        reversed_features = list(reversed(encoder_features))
        
        # Decoder path
        decoder_outputs = []
        x = reversed_features[0]
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Get skip connection
            skip = reversed_features[i+1] if i+1 < len(reversed_features) else None
            
            # Apply decoder block
            x = decoder_block(x, skip)
            
            # Add similarity features if available
            if similarity_features is not None and i == 0:
                # Upsample similarity features to match decoder resolution
                sim_feat = F.interpolate(similarity_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
                x = x + sim_feat
            
            # Add edge features if available
            if edge_features is not None and i == 0:
                # Upsample edge features to match decoder resolution
                edge_feat = F.interpolate(edge_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
                x = x + edge_feat
            
            decoder_outputs.append(x)
        
        # Final output
        final_output = self.final_conv(x)
        
        # Multi-scale supervision outputs
        supervision_outputs = []
        for i, (output, supervision_layer) in enumerate(zip(decoder_outputs, self.supervision_layers)):
            # Upsample to full resolution
            upsampled = F.interpolate(output, size=final_output.shape[-2:], mode='bilinear', align_corners=False)
            supervision_output = torch.sigmoid(supervision_layer(upsampled))
            supervision_outputs.append(supervision_output)
        
        return final_output, supervision_outputs 