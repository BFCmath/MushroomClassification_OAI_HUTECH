import torch
import torch.nn as nn
from cnn.SmallResNet import *
# ### Space-to-Depth Convolution for Information Preservation ###
class SpaceToDepthConv(nn.Module):
    """
    Space-to-Depth Convolution that rearranges spatial information into channel dimension
    instead of losing it through downsampling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, block_size=2, stride=1, padding=1):
        super(SpaceToDepthConv, self).__init__()
        self.block_size = block_size
        # Regular convolution
        self.conv = nn.Conv2d(
            in_channels * (block_size ** 2),  # Input channels increased by block_size^2
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Get dimensions
        batch_size, channels, height, width = x.size()
        
        # Check if dimensions are compatible with block_size
        if height % self.block_size != 0 or width % self.block_size != 0:
            # Add padding if needed
            pad_h = self.block_size - (height % self.block_size)
            pad_w = self.block_size - (width % self.block_size)
            if pad_h < self.block_size or pad_w < self.block_size:
                x = nn.functional.pad(x, (0, pad_w if pad_w < self.block_size else 0, 
                                         0, pad_h if pad_h < self.block_size else 0))
                # Update dimensions after padding
                batch_size, channels, height, width = x.size()
                
        # Space-to-depth transformation: rearrange spatial dims into channel dim
        x = x.view(
            batch_size,
            channels,
            height // self.block_size,
            self.block_size,
            width // self.block_size,
            self.block_size
        )
        # Permute and reshape to get all spatial blocks as channels
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch_size, channels * (self.block_size ** 2), 
                   height // self.block_size, width // self.block_size)
        
        # Apply convolution
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

# ### Dual-Branch Network Architecture ###
class DualBranchNetwork(nn.Module):
    """
    Dual-Branch Network with one branch focusing on global features
    and another branch focusing on local details, with a common feature subspace.
    """
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(DualBranchNetwork, self).__init__()
        
        # Branch 1: Global feature extraction with larger kernels
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 64, stride=2),  # 16x16
            ResidualBlock(64, 128, stride=2),  # 8x8
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Branch 2: Local feature extraction with SPD-Conv to preserve information
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SpaceToDepthConv(32, 64, block_size=2),  # 16x16
            SpaceToDepthConv(64, 128, block_size=2),  # 8x8
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Common feature subspace mapping
        self.feature_fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Use xavier_uniform for linear layers - better for fusion networks
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Process through both branches
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        
        # Concatenate branch outputs
        x_concat = torch.cat((x1, x2), dim=1)
        
        # Map to common feature subspace
        features = self.feature_fusion(x_concat)
        
        # Classification
        out = self.classifier(features)
        
        return out
