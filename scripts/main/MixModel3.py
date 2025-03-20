import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
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

class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important spatial locations.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        # Calculate spatial attention map
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(attention)
        attention = torch.sigmoid(attention)
        
        # Apply attention
        return x * attention

class DilatedMultiScaleBlock(nn.Module):
    """
    Multi-scale feature extraction block using dilated convolutions without spatial reduction.
    Includes residual connection and attention mechanisms.
    """
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4], use_attention=True):
        super(DilatedMultiScaleBlock, self).__init__()
        self.use_attention = use_attention
        
        # Calculate channels per path to maintain reasonable parameter count
        self.path_channels = out_channels // 4  # 4 paths
        
        # Path 1: 1x1 convolution only (point-wise)
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, self.path_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.path_channels),
            nn.ReLU(inplace=False)
        )
        
        # Path 2-4: Different dilated convolutions
        self.path_modules = nn.ModuleList()
        for dilation in dilations:
            self.path_modules.append(nn.Sequential(
                nn.Conv2d(in_channels, self.path_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.path_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    self.path_channels, 
                    self.path_channels, 
                    kernel_size=3, 
                    padding=dilation, 
                    dilation=dilation,
                    bias=False
                ),
                nn.BatchNorm2d(self.path_channels),
                nn.ReLU(inplace=False)
            ))
        
        # Projection to combine all paths back to out_channels
        self.project = nn.Sequential(
            nn.Conv2d(self.path_channels * (1 + len(dilations)), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Residual connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Attention mechanisms
        if use_attention:
            self.channel_attn = SEBlock(out_channels)
            self.spatial_attn = SpatialAttention()
            
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        residual = self.skip(x)
        
        # Multi-scale feature extraction
        path1_out = self.path1(x)
        other_paths = [module(x) for module in self.path_modules]
        
        # Concatenate all paths
        concat_features = torch.cat([path1_out] + other_paths, dim=1)
        
        # Project back to output channels
        out = self.project(concat_features)
        
        # Add residual connection
        out = out + residual
        out = self.relu(out)
        
        # Apply attention if enabled
        if self.use_attention:
            out = self.channel_attn(out)
            out = self.spatial_attn(out)
            
        return out

class MixModel3(nn.Module):
    """
    MixModel3: A CNN that maintains 32x32 spatial resolution throughout the network.
    
    Key features:
    1. No spatial downsampling - preserves all spatial information
    2. Uses dilated convolutions to increase receptive field without resolution loss
    3. Combines channel and spatial attention mechanisms
    4. Efficient parameter usage with multi-scale feature extraction
    5. Only reduces spatial dimensions at the final classification stage
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MixModel3, self).__init__()
        
        # Stage 1: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        
        # Stage 2: Multi-scale feature extraction blocks (maintains 32x32 resolution)
        self.stage2 = nn.Sequential(
            DilatedMultiScaleBlock(64, 96, dilations=[1, 2, 4]),
            DilatedMultiScaleBlock(96, 128, dilations=[1, 3, 5])
        )
        
        # Stage 3: Deeper feature extraction with increased dilations (maintains 32x32 resolution)
        self.stage3 = nn.Sequential(
            DilatedMultiScaleBlock(128, 192, dilations=[1, 3, 6]),
            DilatedMultiScaleBlock(192, 256, dilations=[1, 4, 8])
        )
        
        # Stage 4: Final feature refinement (maintains 32x32 resolution)
        self.stage4 = DilatedMultiScaleBlock(256, 384, dilations=[1, 2, 5, 9], use_attention=True)
        
        # Stage 5: Efficient feature aggregation
        # Multi-scale spatial pooling to capture different levels of detail
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 32x32 → 1x1
        self.mid_pool = nn.AdaptiveAvgPool2d((2, 2))     # 32x32 → 2x2
        self.local_pool = nn.AdaptiveAvgPool2d((4, 4))   # 32x32 → 4x4
        
        # Flatten and combine pooled features
        global_features = 384  # From global_pool
        mid_features = 384 * 4  # From mid_pool (2x2)
        local_features = 384 * 16  # From local_pool (4x4)
        total_features = global_features + mid_features + local_features
        
        # Feature fusion layers
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
        # Input: 32x32x3
        
        # Stage 1: Initial feature extraction
        x = self.stem(x)  # 32x32x64
        
        # Stage 2: Multi-scale feature extraction
        x = self.stage2(x)  # 32x32x128
        
        # Stage 3: Deeper feature extraction
        x = self.stage3(x)  # 32x32x256
        
        # Stage 4: Final feature refinement
        x = self.stage4(x)  # 32x32x384
        
        # Stage 5: Multi-scale feature aggregation
        global_features = self.global_pool(x)  # 1x1x384
        mid_features = self.mid_pool(x)        # 2x2x384
        local_features = self.local_pool(x)    # 4x4x384
        
        # Flatten and concatenate
        global_features = torch.flatten(global_features, 1)  # 384
        mid_features = torch.flatten(mid_features, 1)        # 1536 (384*2*2)
        local_features = torch.flatten(local_features, 1)    # 6144 (384*4*4)
        
        concat_features = torch.cat([global_features, mid_features, local_features], dim=1)
        
        # Feature fusion
        fused = self.fusion(concat_features)
        
        # Classification
        out = self.classifier(fused)
        
        return out
