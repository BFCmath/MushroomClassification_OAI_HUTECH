import torch
import torch.nn as nn
import torch.nn.functional as F
from SPDDualBranch import SPDConv  # Import the existing SPDConv implementation

class Mix2SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """
    def __init__(self, channels, reduction=16):
        super(Mix2SEBlock, self).__init__()
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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Mix2InceptionBlockA(nn.Module):
    """
    Inception block with four parallel paths and dilated convolutions.
    """
    def __init__(self, in_channels, filters=16):
        super(Mix2InceptionBlockA, self).__init__()
        
        # Path 1: 1x1 convolution only
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Path 2: 1x1 → 3x3 conv with dilation=1
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Path 3: 1x1 → 3x3 conv with dilation=2
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Path 4: 1x1 → 3x3 conv with dilation=3
        self.path4 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)
        return torch.cat([path1_out, path2_out, path3_out, path4_out], dim=1)

class Mix2InceptionBlockC(nn.Module):
    """
    Simplified inception block with two parallel paths.
    """
    def __init__(self, in_channels, filters=64):
        super(Mix2InceptionBlockC, self).__init__()
        
        # Path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Path 2: 3x3 convolution
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        return torch.cat([path1_out, path2_out], dim=1)

class MixModel2(nn.Module):
    """
    MixModel2 combining Space-to-Depth convolutions, Inception blocks with dilated convolutions,
    and Squeeze-and-Excitation attention for 32x32 images.
    
    This model features:
    1. Initial downsampling with Space-to-Depth
    2. Multi-scale feature extraction with Inception blocks
    3. Skip connections in feature stages for better gradient flow
    4. SE attention for channel-wise feature refinement
    5. Progressive downsampling with Space-to-Depth operations
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MixModel2, self).__init__()
        
        # Initial Space-to-Depth convolution (32x32x3 → 16x16x12)
        self.spd_initial = SPDConv(in_channels=3, out_channels=12, scale=2, kernel_size=1, padding=0)
        
        # Stage A: First inception stage at 16x16 resolution
        self.inception_a1 = Mix2InceptionBlockA(in_channels=12, filters=16)  # 16x16x12 → 16x16x64
        self.se_a1 = Mix2SEBlock(channels=64)  # SE attention on inception output
        
        self.inception_a2 = Mix2InceptionBlockA(in_channels=64, filters=16)  # 16x16x64 → 16x16x64
        self.se_a2 = Mix2SEBlock(channels=64)  # SE attention on inception output
        
        # Downsampling to Stage B (16x16x64 → 8x8x256)
        self.spd_b = SPDConv(in_channels=64, out_channels=256, scale=2, kernel_size=1, padding=0)
        
        # Stage B: Second inception stage at 8x8 resolution
        self.inception_b1 = Mix2InceptionBlockA(in_channels=256, filters=32)  # 8x8x256 → 8x8x128
        self.se_b1 = Mix2SEBlock(channels=128)  # SE attention on inception output
        
        self.inception_b2 = Mix2InceptionBlockA(in_channels=128, filters=32)  # 8x8x128 → 8x8x128
        self.se_b2 = Mix2SEBlock(channels=128)  # SE attention on inception output
        
        # Downsampling to Stage C (8x8x128 → 4x4x512)
        self.spd_c = SPDConv(in_channels=128, out_channels=512, scale=2, kernel_size=1, padding=0)
        
        # Stage C: Final inception stage at 4x4 resolution
        self.inception_c = Mix2InceptionBlockC(in_channels=512, filters=64)  # 4x4x512 → 4x4x128
        self.se_c = Mix2SEBlock(channels=128)  # SE attention on inception output
        
        # Global Average Pooling and Classification
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 4x4x128 → 1x1x128
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_classes)  # 128 → num_classes
        
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
        # Initial Space-to-Depth
        x = self.spd_initial(x)  # 3x32x32 → 12x16x16
        
        # Stage A (16x16 resolution)
        x = self.inception_a1(x)  # 12x16x16 → 64x16x16
        x = self.se_a1(x)
        
        # Skip connection for Stage A's second block
        x_a2_input = x
        x = self.inception_a2(x)  # 64x16x16 → 64x16x16
        x = x + x_a2_input  # Skip connection
        x = self.se_a2(x)
        
        # Downsample to Stage B
        x = self.spd_b(x)  # 64x16x16 → 256x8x8
        
        # Stage B (8x8 resolution)
        x = self.inception_b1(x)  # 256x8x8 → 128x8x8
        x = self.se_b1(x)
        
        # Skip connection for Stage B's second block
        x_b2_input = x
        x = self.inception_b2(x)  # 128x8x8 → 128x8x8
        x = x + x_b2_input  # Skip connection
        x = self.se_b2(x)
        
        # Downsample to Stage C
        x = self.spd_c(x)  # 128x8x8 → 512x4x4
        
        # Stage C (4x4 resolution)
        x = self.inception_c(x)  # 512x4x4 → 128x4x4
        x = self.se_c(x)
        
        # Global Average Pooling and Classification
        x = self.gap(x)  # 128x4x4 → 128x1x1
        x = torch.flatten(x, 1)  # 128x1x1 → 128
        x = self.dropout(x)
        x = self.fc(x)  # 128 → num_classes
        
        return x
