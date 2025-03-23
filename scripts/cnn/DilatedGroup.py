import torch
import torch.nn as nn
from scripts.multi_branch.SPDDualBranch import SPDConv  # Import SPDConv for downsampling

# ### Dilated Group Convolution Network ###
class DilatedGroupConvBlock(nn.Module):
    """
    Custom block implementing dilated group convolutions with 7x7 kernels.
    
    Features:
    - Uses grouped convolutions to reduce parameter count
    - Employs dilated convolutions to increase receptive field without pooling
    - Maintains spatial information with residual connections
    """
    def __init__(self, in_channels, out_channels, dilation=1, stride=1, groups=4, reduction_ratio=4):
        super(DilatedGroupConvBlock, self).__init__()
        
        # Calculate reduced dimensions for bottleneck
        reduced_channels = max(out_channels // reduction_ratio, 8)
        
        # Ensure in_channels and out_channels are divisible by groups
        assert in_channels % groups == 0, f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        assert out_channels % groups == 0, f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        
        # Input projection with 1x1 convolution (no dilation needed here)
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dilated grouped convolution with 7x7 kernel
        self.grouped_conv = nn.Sequential(
            nn.Conv2d(
                reduced_channels, 
                reduced_channels, 
                kernel_size=7, 
                stride=1,  # Changed from stride parameter to always 1
                padding=3 * dilation,  # Maintain spatial size with proper padding
                dilation=dilation,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(reduced_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection (residual path)
        self.skip = nn.Identity()
        
        # Use SPDConv for downsampling in skip connection if stride > 1
        if stride > 1:
            self.skip = nn.Sequential(
                SPDConv(in_channels, out_channels, kernel_size=1, scale=stride, padding=0)
            )
        elif in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        # Add SPDConv for downsampling in main path if stride > 1
        self.downsample = None
        if stride > 1:
            self.downsample = SPDConv(
                reduced_channels,
                reduced_channels,
                kernel_size=3,
                scale=stride,
                padding=1
            )
            
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        # Main branch
        out = self.input_proj(x)
        out = self.grouped_conv(out)
        
        # Apply downsampling if needed
        if self.downsample is not None:
            out = self.downsample(out)
            
        out = self.output_proj(out)
        
        # Residual connection
        out += self.skip(identity)
        out = self.relu(out)
        
        return out


class DilatedGroupConvNet(nn.Module):
    """
    Neural network using dilated group convolutions with 7x7 kernels.
    
    This architecture:
    1. Uses dilated convolutions to capture wide context without pooling
    2. Employs group convolutions to reduce parameter count
    3. Replaces pooling operations with SPDConv for downsampling
    4. Maintains spatial information flow via residual connections
    """
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(DilatedGroupConvNet, self).__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: Regular and dilated convolutions at same scale (32x32)
        self.stage1 = nn.Sequential(
            DilatedGroupConvBlock(32, 64, dilation=1, stride=1),
            DilatedGroupConvBlock(64, 64, dilation=2, stride=1),
            DilatedGroupConvBlock(64, 64, dilation=4, stride=1)
        )
        
        # Transition 1: SPDConv instead of strided convolution (32x32 → 16x16)
        self.transition1 = DilatedGroupConvBlock(64, 128, dilation=1, stride=2)
        
        # Stage 2: Medium-scale features (16x16)
        self.stage2 = nn.Sequential(
            DilatedGroupConvBlock(128, 128, dilation=1, stride=1),
            DilatedGroupConvBlock(128, 128, dilation=2, stride=1),
            DilatedGroupConvBlock(128, 128, dilation=4, stride=1)
        )
        
        # Transition 2: SPDConv instead of strided convolution (16x16 → 8x8)
        self.transition2 = DilatedGroupConvBlock(128, 256, dilation=1, stride=2)
        
        # Stage 3: Deep features with increased dilation (8x8)
        self.stage3 = nn.Sequential(
            DilatedGroupConvBlock(256, 256, dilation=1, stride=1),
            DilatedGroupConvBlock(256, 256, dilation=2, stride=1)
        )
        
        # Global feature extraction using SPDConv instead of strided convolutions
        self.global_features = nn.Sequential(
            SPDConv(256, 512, kernel_size=3, scale=2, padding=1),  # 8x8 → 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SPDConv(512, 512, kernel_size=3, scale=2, padding=1),  # 4x4 → 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Changed kernel_size from 2 to 1 to fix size mismatch error
            SPDConv(512, 512, kernel_size=1, scale=2, padding=0),  # 2x2 → 1x1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
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
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.stem(x)
        
        # Feature extraction stages with dilated convolutions
        x = self.stage1(x)
        x = self.transition1(x)
        x = self.stage2(x)
        x = self.transition2(x)
        x = self.stage3(x)
        
        # Global feature extraction without pooling
        x = self.global_features(x)
        
        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
