import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.multi_branch.SPDDualBranch import SPDConv  # Import the existing SPDConv implementation

class Mix1SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """
    def __init__(self, channels, reduction=16):
        super(Mix1SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 8)  # Ensure minimum size
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        # Global average pooling
        y = self.avg_pool(x).view(b, c)
        # Channel attention
        y = self.fc(y).view(b, c, 1, 1)
        # Scale the input
        return x * y.expand_as(x)

class Mix1ResidualInceptionBlock(nn.Module):
    """
    Residual Inception Block with dilated convolutions and SE attention.
    """
    def __init__(self, in_channels, out_channels, path_channels, dilations=[1, 2, 3]):
        super(Mix1ResidualInceptionBlock, self).__init__()
        
        # Ensure in_channels match out_channels for residual connection
        self.adjust_channels = None
        if in_channels != out_channels:
            self.adjust_channels = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        # Path 1: 1x1 conv only
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, path_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(path_channels),
            nn.ReLU(inplace=True)
        )
        
        # Paths 2-4: 1x1 conv -> 3x3 dilated conv
        # Dynamically create paths based on dilations parameter
        self.paths = nn.ModuleList()
        for dilation in dilations:
            self.paths.append(nn.Sequential(
                nn.Conv2d(in_channels, path_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(path_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(path_channels, path_channels, kernel_size=3, padding=dilation, 
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(path_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Final 1x1 conv to combine outputs
        self.combine = nn.Sequential(
            nn.Conv2d(path_channels * (1 + len(dilations)), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # SE attention block
        self.se = Mix1SEBlock(out_channels)
        
        # ReLU after residual connection
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Save input for residual connection
        residual = x if self.adjust_channels is None else self.adjust_channels(x)
        
        # Process through path 1
        path1_out = self.path1(x)
        
        # Process through additional paths with dilated convolutions
        path_outputs = [path1_out]
        for path in self.paths:
            path_outputs.append(path(x))
        
        # Concatenate all path outputs
        combined = torch.cat(path_outputs, dim=1)
        
        # Final 1x1 convolution
        out = self.combine(combined)
        
        # Apply SE attention
        out = self.se(out)
        
        # Add residual connection and apply ReLU
        out = self.relu(out + residual)
        
        return out

class Mix1SpaceToDepthConv(nn.Module):
    """
    Space-to-Depth Convolution with custom channel adjustment.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Mix1SpaceToDepthConv, self).__init__()
        # After space-to-depth, channels increase by scale_factor^2
        self.spd = nn.PixelUnshuffle(scale_factor)
        
        # 1x1 conv to adjust channels
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(in_channels * (scale_factor ** 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Apply space-to-depth operation
        x = self.spd(x)
        # Adjust channels
        x = self.channel_adjust(x)
        return x

class MixModel1(nn.Module):
    """
    MixModel1 combining SPDConv, Residual Inception blocks with dilated convolutions,
    and Squeeze-and-Excitation attention for 32x32 images.
    
    Architecture follows the layer-by-layer design specified:
    1. SPDConv initial downsampling
    2. Channel adjustment
    3. Multi-stage feature extraction with residual inception blocks
    4. Progressive downsampling using SPDConv
    5. Global pooling and classification
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MixModel1, self).__init__()
        
        # Initial Layer: SPDConv (32x32x3 → 16x16x12)
        self.initial_spd = SPDConv(in_channels=3, out_channels=12, scale=2, kernel_size=3, padding=1)
        
        # Channel Adjustment: 1x1 Conv (16x16x12 → 16x16x64)
        self.initial_channel_adjust = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual Inception Block A1 (16x16x64 → 16x16x64)
        self.block_a1 = Mix1ResidualInceptionBlock(
            in_channels=64,
            out_channels=64,
            path_channels=16,
            dilations=[1, 2, 3]
        )
        
        # Residual Inception Block A2 (16x16x64 → 16x16x64)
        self.block_a2 = Mix1ResidualInceptionBlock(
            in_channels=64,
            out_channels=64,
            path_channels=16,
            dilations=[1, 2, 3]
        )
        
        # Downsampling: SPDConv (16x16x64 → 8x8x256)
        self.downsample1 = SPDConv(
            in_channels=64,
            out_channels=256,
            scale=2
        )
        
        # Channel Adjustment: 1x1 Conv (8x8x256 → 8x8x128)
        self.channel_adjust1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Residual Inception Block B1 (8x8x128 → 8x8x128)
        self.block_b1 = Mix1ResidualInceptionBlock(
            in_channels=128,
            out_channels=128,
            path_channels=32,
            dilations=[1, 2, 3]
        )
        
        # Residual Inception Block B2 (8x8x128 → 8x8x128)
        self.block_b2 = Mix1ResidualInceptionBlock(
            in_channels=128,
            out_channels=128,
            path_channels=32,
            dilations=[1, 2, 3]
        )
        
        # Downsampling: SPDConv (8x8x128 → 4x4x512)
        self.downsample2 = SPDConv(
            in_channels=128,
            out_channels=512,
            scale=2
        )
        
        # Channel Adjustment: 1x1 Conv (4x4x512 → 4x4x256)
        self.channel_adjust2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual Inception Block C1 (4x4x256 → 4x4x256)
        self.block_c1 = Mix1ResidualInceptionBlock(
            in_channels=256,
            out_channels=256,
            path_channels=64,
            dilations=[1, 2, 3]
        )
        
        # Global Feature Aggregation: Global Average Pooling (4x4x256 → 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Layer (256 → num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial downsampling: SPDConv (32x32x3 → 16x16x12)
        x = self.initial_spd(x)
        
        # Channel adjustment (16x16x12 → 16x16x64)
        x = self.initial_channel_adjust(x)
        
        # Stage A: 16x16 resolution
        x = self.block_a1(x)
        x = self.block_a2(x)
        
        # Downsampling to 8x8
        x = self.downsample1(x)
        x = self.channel_adjust1(x)
        
        # Stage B: 8x8 resolution
        x = self.block_b1(x)
        x = self.block_b2(x)
        
        # Downsampling to 4x4
        x = self.downsample2(x)
        x = self.channel_adjust2(x)
        
        # Stage C: 4x4 resolution
        x = self.block_c1(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
