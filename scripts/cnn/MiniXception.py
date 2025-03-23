import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.multi_branch.SPDDualBranch import SPDConv  # Import the existing Space-to-Depth implementation

class SeparableConv2d(nn.Module):
    """
    Depthwise separable convolution that factorizes a standard convolution into
    depthwise (per-channel) and pointwise (1x1) convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        # Depthwise convolution (one filter per input channel)
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Depthwise: one filter per input channel
            bias=bias
        )
        # Pointwise convolution (1x1 conv to change channel dimensions)
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1,  # 1x1 kernel
            bias=bias
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class XceptionBlock(nn.Module):
    """
    Basic block in Xception architecture with separable convolutions and residual connection.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(XceptionBlock, self).__init__()
        
        # Main branch with 3 separable convolutions
        self.sep_conv1 = nn.Sequential(
            # Change inplace=True to inplace=False for all ReLU activations
            nn.ReLU(inplace=False),
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.sep_conv2 = nn.Sequential(
            nn.ReLU(inplace=False),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.sep_conv3 = nn.Sequential(
            nn.ReLU(inplace=False),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.sep_conv1(x)
        out = self.sep_conv2(out)
        out = self.sep_conv3(out)
        
        return out + residual

class DownsampleBlock(nn.Module):
    """
    Block for spatial downsampling using Space-to-Depth instead of pooling.
    """
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        
        # Main branch: Space-to-Depth downsampling
        self.main_branch = nn.Sequential(
            # Change inplace=True to inplace=False for all ReLU activations
            nn.ReLU(inplace=False),
            SeparableConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )
        
        # Space-to-Depth downsampling (replaces pooling)
        self.spd_downsample = SPDConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            scale=2,  # Scale factor for downsampling
            padding=1
        )
        
        # Skip connection with Space-to-Depth to match dimensions
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # Use SPDConv for skip connection downsampling
            SPDConv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                scale=2,
                padding=0
            )
        )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.main_branch(x)
        out = self.spd_downsample(out)
        
        return out + residual

class MiniXception(nn.Module):
    """
    Xception-inspired model for 32x32x3 inputs with Space-to-Depth downsampling.
    
    This model:
    1. Uses depthwise separable convolutions for efficiency
    2. Replaces pooling with Space-to-Depth operations to preserve information
    3. Uses residual connections throughout for better gradient flow
    4. Optimized for small 32x32 input size
    """
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(MiniXception, self).__init__()
        
        # Entry flow
        # Initial convolution (32x32 -> 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            # Change inplace=True to inplace=False for all ReLU activations
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        
        # First downsampling block (32x32 -> 16x16)
        self.down1 = DownsampleBlock(64, 128)
        
        # XceptionBlock after first downsampling
        self.block1 = XceptionBlock(128, 128)
        
        # Second downsampling block (16x16 -> 8x8)
        self.down2 = DownsampleBlock(128, 256)
        
        # Middle flow - repeated Xception blocks
        self.middle_flow = nn.Sequential(
            XceptionBlock(256, 256),
            XceptionBlock(256, 256),
            XceptionBlock(256, 256)
        )
        
        # Final downsampling to 4x4 resolution
        self.down3 = DownsampleBlock(256, 512)
        
        # Exit flow
        self.exit_flow = nn.Sequential(
            SeparableConv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            SeparableConv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False)
        )
        
        # Global average pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(512, num_classes)
        
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
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Entry flow
        x = self.stem(x)          # 32x32x64
        x = self.down1(x)         # 16x16x128
        x = self.block1(x)        # 16x16x128
        x = self.down2(x)         # 8x8x256
        
        # Middle flow
        x = self.middle_flow(x)   # 8x8x256
        
        # Exit flow
        x = self.down3(x)         # 4x4x512
        x = self.exit_flow(x)     # 4x4x512
        
        # Global pooling and classification
        x = self.global_pool(x)   # 1x1x512
        x = torch.flatten(x, 1)   # 512
        x = self.dropout(x)
        x = self.classifier(x)    # num_classes
        
        return x
