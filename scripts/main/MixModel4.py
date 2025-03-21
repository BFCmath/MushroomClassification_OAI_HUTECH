import torch
import torch.nn as nn
import torch.nn.functional as F
from SPDDualBranch import SPDConv  # Import the existing SPDConv implementation

class Mix4MultiBranchModule(nn.Module):
    """
    Multi-branch module with three paths for learning different spatial features:
    - Branch 1: Local features (1x1 conv)
    - Branch 2: Medium-scale features (3x3, 5x5 dilated=1, 7x7 dilated=2 convs)
    - Branch 3: Global features (5x5, 7x7 dilated=1, 9x9 dilated=2 convs)
    
    Branches process features in parallel to efficiently capture different receptive fields.
    
    Resolution flow:
    - Input: HxW with C channels
    - Branch 1 (Local features): HxW preserved (1x1 kernels don't affect spatial dimensions)
    - Branch 2 (Medium features): HxW preserved for all convs with proper padding
      * 3x3 conv (pad=1): Receptive field = 3x3
      * 5x5 conv (pad=2): Receptive field = 5x5
      * 7x7 conv with dilation=2 (pad=6): Effective receptive field = 13x13
    - Branch 3 (Global features): HxW preserved for all convs with proper padding
      * 5x5 conv (pad=2): Receptive field = 5x5
      * 7x7 conv (pad=3): Receptive field = 7x7
      * 9x9 conv with dilation=2 (pad=8): Effective receptive field = 17x17
    - Output: HxW with out_channels (spatial dimensions maintained)
    """
    def __init__(self, in_channels, out_channels, use_residual=True):
        super(Mix4MultiBranchModule, self).__init__()
        
        # Calculate channels per branch to maintain reasonable parameter count
        self.branch1_channels = out_channels // 4
        self.branch2_channels = out_channels // 4
        self.branch3_channels = out_channels // 2
        
        # Adjust if the division isn't exact
        total_channels = self.branch1_channels + self.branch2_channels + self.branch3_channels
        if total_channels != out_channels:
            self.branch3_channels += (out_channels - total_channels)
        
        # Branch 1: Local features with 1x1 conv (smallest receptive field)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, self.branch1_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.branch1_channels),
            nn.ReLU(inplace=False)
        )
        
        # Branch 2: Medium-scale features with specific convolutions
        branch2_channels_per_conv = self.branch2_channels // 3
        remainder = self.branch2_channels - (branch2_channels_per_conv * 3)
        
        self.branch2_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch2_channels_per_conv, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(branch2_channels_per_conv),
            nn.ReLU(inplace=False)
        )
        
        self.branch2_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch2_channels_per_conv, kernel_size=5, 
                     stride=1, padding=2, dilation=1, bias=False),
            nn.BatchNorm2d(branch2_channels_per_conv),
            nn.ReLU(inplace=False)
        )
        
        self.branch2_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, branch2_channels_per_conv + remainder, kernel_size=7, 
                     stride=1, padding=6, dilation=2, bias=False),
            nn.BatchNorm2d(branch2_channels_per_conv + remainder),
            nn.ReLU(inplace=False)
        )
        
        # Branch 3: Global features with larger kernels and dilations
        branch3_channels_per_conv = self.branch3_channels // 3
        remainder = self.branch3_channels - (branch3_channels_per_conv * 3)
        
        self.branch3_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch3_channels_per_conv, kernel_size=5, 
                     stride=1, padding=2, dilation=1, bias=False),
            nn.BatchNorm2d(branch3_channels_per_conv),
            nn.ReLU(inplace=False)
        )
        
        self.branch3_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, branch3_channels_per_conv, kernel_size=7, 
                     stride=1, padding=3, dilation=1, bias=False),
            nn.BatchNorm2d(branch3_channels_per_conv),
            nn.ReLU(inplace=False)
        )
        
        self.branch3_9x9 = nn.Sequential(
            nn.Conv2d(in_channels, branch3_channels_per_conv + remainder, kernel_size=9, 
                     stride=1, padding=8, dilation=2, bias=False),
            nn.BatchNorm2d(branch3_channels_per_conv + remainder),
            nn.ReLU(inplace=False)
        )
        
        # Residual connection setup
        self.use_residual = use_residual
        self.residual = nn.Identity()
        if use_residual and in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        # Input resolution: HxW with in_channels
        residual = x if not self.use_residual else self.residual(x)
        
        # Branch 1 - Local features with 1x1 conv (preserves spatial resolution)
        # Resolution: HxW → HxW with branch1_channels
        b1 = self.branch1(x)
        
        # Branch 2 - Medium features with specified convs (all preserve spatial resolution)
        # 3x3 conv - Resolution: HxW → HxW with branch2_channels/3
        b2_3x3 = self.branch2_3x3(x)
        
        # 5x5 conv - Resolution: HxW → HxW with branch2_channels/3 
        b2_5x5 = self.branch2_5x5(x)
        
        # 7x7 dilated conv - Resolution: HxW → HxW with branch2_channels/3
        # Effective receptive field: 13x13 due to dilation=2
        b2_7x7 = self.branch2_7x7(x)
        
        # Branch 3 - Global features with larger kernels (all preserve spatial resolution)
        # 5x5 conv - Resolution: HxW → HxW with branch3_channels/3
        b3_5x5 = self.branch3_5x5(x)
        
        # 7x7 conv - Resolution: HxW → HxW with branch3_channels/3
        b3_7x7 = self.branch3_7x7(x)
        
        # 9x9 dilated conv - Resolution: HxW → HxW with branch3_channels/3
        # Effective receptive field: 17x17 due to dilation=2
        b3_9x9 = self.branch3_9x9(x)
        
        # Concatenate branch outputs along channel dimension
        # Resolution: HxW, Channels = sum of all branch channels (equals out_channels)
        out = torch.cat([
            b1, 
            b2_3x3, b2_5x5, b2_7x7,
            b3_5x5, b3_7x7, b3_9x9
        ], dim=1)
        
        # Add residual connection if enabled (preserves spatial resolution)
        if self.use_residual:
            out = out + residual
            out = self.relu(out)
        
        # Output resolution: HxW with out_channels (spatial dimensions maintained)
        return out

class Mix4SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """
    def __init__(self, channels, reduction=16):
        super(Mix4SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 8)  # Ensure minimum size
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MixModel4(nn.Module):
    """
    MixModel4: A CNN with multi-branch modules capturing different spatial scales.
    
    Key features:
    1. Parallel branches for different receptive fields:
       - Branch 1: Local features with 1x1 conv
       - Branch 2: Medium features with parallel 3x3, 5x5, 7x7 convolutions
       - Branch 3: Global features with parallel 5x5, 7x7, 9x9 convolutions
    2. Minimal downsampling using SPDConv when needed
    3. Residual connections throughout the network
    4. Multi-scale feature aggregation for final classification
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MixModel4, self).__init__()
        
        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        
        # Stage 1: Multi-branch modules with SE attention (32x32 spatial size)
        self.stage1 = nn.Sequential(
            Mix4MultiBranchModule(64, 96, use_residual=True),
            Mix4SEBlock(96),
            Mix4MultiBranchModule(96, 128, use_residual=True),
            Mix4SEBlock(128)
        )
        
        # Optional downsampling with SPDConv (32x32 → 16x16)
        self.downsample1 = SPDConv(128, 192, scale=2)
        
        # Stage 2: Multi-branch modules with SE attention (16x16 spatial size)
        self.stage2 = nn.Sequential(
            Mix4MultiBranchModule(192, 256, use_residual=True),
            Mix4SEBlock(256),
            Mix4MultiBranchModule(256, 384, use_residual=True),
            Mix4SEBlock(384)
        )
        
        # Multi-scale pooling for feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global features
        self.mid_pool = nn.AdaptiveAvgPool2d((2, 2))     # Medium-scale features
        self.local_pool = nn.AdaptiveAvgPool2d((4, 4))   # Local features
        
        # Calculate feature dimensions
        global_features = 384           # 1x1x384
        mid_features = 384 * 2 * 2      # 2x2x384
        local_features = 384 * 4 * 4    # 4x4x384
        total_features = global_features + mid_features + local_features
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate/2)
        )
        
        # Classification layer
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: 3x32x32
        
        # Initial feature extraction
        x = self.stem(x)  # 3x32x32 → 64x32x32 (maintains spatial resolution)
        
        # Stage 1: Multi-branch modules at 32x32 resolution
        x = self.stage1(x)  # 64x32x32 → 128x32x32 (maintains spatial resolution)
        
        # Downsampling (only once in the network)
        x = self.downsample1(x)  # 128x32x32 → 192x16x16 (spatial reduction by factor of 2)
        
        # Stage 2: Multi-branch modules at 16x16 resolution
        x = self.stage2(x)  # 192x16x16 → 384x16x16 (maintains 16x16 resolution)
        
        # Multi-scale feature aggregation
        global_features = self.global_pool(x)  # 384x16x16 → 384x1x1 (global pooling)
        mid_features = self.mid_pool(x)        # 384x16x16 → 384x2x2 (medium pooling)
        local_features = self.local_pool(x)    # 384x16x16 → 384x4x4 (local pooling)
        
        # Flatten and concatenate features
        global_features = torch.flatten(global_features, 1)  # 384x1x1 → 384 (flattened)
        mid_features = torch.flatten(mid_features, 1)        # 384x2x2 → 1536 (flattened)
        local_features = torch.flatten(local_features, 1)    # 384x4x4 → 6144 (flattened)
        concat_features = torch.cat([global_features, mid_features, local_features], dim=1)
        # Concatenated features: 384 + 1536 + 6144 = 8064 dimensions
        
        # Feature fusion
        fused = self.fusion(concat_features)  # 8064 → 256 (with intermediate 512)
        
        # Classification
        out = self.classifier(fused)  # 256 → num_classes
        
        return out
