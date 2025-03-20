import torch
import torch.nn as nn
import torch.nn.functional as F

class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution that rearranges spatial information into channel dimension
    instead of losing it through downsampling.
    Used specifically for downsampling operations.
    """
    def __init__(self, in_channels, out_channels, scale=2, kernel_size=3, padding=1):
        super(SPDConv, self).__init__()
        self.scale = scale
        # Convolution layer: input channels are scaled by scale^2 due to space-to-depth
        self.conv = nn.Conv2d(in_channels * scale**2, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Space-to-depth: rearranges spatial data into channels
        batch_size, channels, height, width = x.size()
        
        # Check if dimensions are compatible with scale factor
        if height % self.scale != 0 or width % self.scale != 0:
            # Add padding if needed
            pad_h = self.scale - (height % self.scale)
            pad_w = self.scale - (width % self.scale)
            if pad_h < self.scale or pad_w < self.scale:
                x = F.pad(x, (0, pad_w if pad_w < self.scale else 0, 
                             0, pad_h if pad_h < self.scale else 0))
                # Update dimensions after padding
                batch_size, channels, height, width = x.size()
                
        # Space-to-depth transformation: rearrange spatial dims into channel dim
        x = x.view(
            batch_size,
            channels,
            height // self.scale,
            self.scale,
            width // self.scale,
            self.scale
        )
        # Permute and reshape to get all spatial blocks as channels
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch_size, channels * (self.scale ** 2), 
                  height // self.scale, width // self.scale)
        
        # Apply convolution
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class ConvBlock(nn.Module):
    """Standard convolutional block for feature extraction without downsampling."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SPDDualBranchNetwork(nn.Module):
    """
    Dual-Branch Network where both branches use SPDConv exclusively for downsampling operations.
    
    - Branch 1: Focuses on global features using larger kernels
    - Branch 2: Focuses on local details using smaller kernels
    - Both branches use SPDConv for downsampling to preserve spatial information
    """
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(SPDDualBranchNetwork, self).__init__()
        
        # Branch 1: Global feature extraction with larger kernels
        self.branch1_init = ConvBlock(3, 32, kernel_size=7, padding=3)
        self.branch1_down1 = SPDConv(32, 64, scale=2)  # 32x32 -> 16x16
        self.branch1_block1 = ConvBlock(64, 64, kernel_size=5, padding=2)
        self.branch1_down2 = SPDConv(64, 128, scale=2)  # 16x16 -> 8x8
        self.branch1_block2 = ConvBlock(128, 128, kernel_size=5, padding=2)
        
        # Branch 2: Local feature extraction with smaller kernels
        self.branch2_init = ConvBlock(3, 32, kernel_size=3, padding=1)
        self.branch2_down1 = SPDConv(32, 64, scale=2)  # 32x32 -> 16x16
        self.branch2_block1 = ConvBlock(64, 64, kernel_size=3, padding=1)
        self.branch2_down2 = SPDConv(64, 128, scale=2)  # 16x16 -> 8x8
        self.branch2_block2 = ConvBlock(128, 128, kernel_size=3, padding=1)
        
        # Additional processing for both branches
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers for each branch
        self.branch1_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.branch2_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Feature fusion and classifier
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Linear(256, num_classes)
        
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
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Branch 1: Global feature path
        b1 = self.branch1_init(x)
        b1 = self.branch1_down1(b1)  # SPDConv downsampling
        b1 = self.branch1_block1(b1)
        b1 = self.branch1_down2(b1)  # SPDConv downsampling
        b1 = self.branch1_block2(b1)
        b1 = self.avgpool(b1)
        b1 = self.branch1_fc(b1)
        
        # Branch 2: Local detail path
        b2 = self.branch2_init(x)
        b2 = self.branch2_down1(b2)  # SPDConv downsampling
        b2 = self.branch2_block1(b2)
        b2 = self.branch2_down2(b2)  # SPDConv downsampling
        b2 = self.branch2_block2(b2)
        b2 = self.avgpool(b2)
        b2 = self.branch2_fc(b2)
        
        # Concatenate features from both branches
        combined = torch.cat([b1, b2], dim=1)
        
        # Feature fusion
        fused = self.fusion(combined)
        
        # Classification
        out = self.classifier(fused)
        
        return out
