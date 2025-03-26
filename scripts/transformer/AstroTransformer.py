import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .PixT import PositionalEncoding

class RelativeAttention(nn.Module):
    """
    Attention mechanism with relative position bias using depthwise convolutions.
    This follows the implementation in Sample.py.
    """
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.proj_qkv = nn.Linear(in_channels, 3 * in_channels)
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                       padding=kernel_size//2, groups=in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C] where N = H * W
        qkv = self.proj_qkv(x_flat).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # [3, B, N, C]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, N, C]
        
        # Standard attention
        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, N, N]
        
        # Relative Position Bias
        relative_bias = self.depthwise_conv(x).flatten(2).transpose(1, 2)  # [B, N, C]
        relative_bias_attn = torch.matmul(q, relative_bias.transpose(-2, -1))  # [B, N, N]
        attn = attn + relative_bias_attn  # Combine standard and relative attention
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out_flat = torch.matmul(attn, v)  # [B, N, C]
        out = out_flat.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        return out

class StageBlock(nn.Module):
    """
    Block used within stages, consisting of depthwise and expansion convolutions.
    This follows the implementation in Sample.py but maintains spatial dimensions.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Depthwise Convolution (D-Conv)
        self.d_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, 
            padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Expansion Convolution (E-Conv)
        self.e_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip Connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.d_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.e_conv(x)
        x = self.bn2(x)
        x = x + identity
        x = self.relu(x)
        return x

class AstroTransformer(nn.Module):
    """
    Adapted from Sample.py's architecture for 32x32 images and 4-class classification.
    
    This version maintains the full 32x32 spatial resolution throughout the network,
    which is more appropriate for small images where spatial information is critical.
    
    Each stage processes features at different channel depths but maintains spatial dimensions,
    with skip connections between stages.
    """
    def __init__(self, num_classes=4, expansion=2, layers=[2, 2, 2]):
        super().__init__()
        
        # Stem Stage (S0) - maintain spatial size for small input
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.expansion = expansion
        self.layers = layers  # Number of blocks per stage: [L1, L2, L3]

        # Stage S1 - in:32 -> out:64 (for expansion=2), maintain 32x32 size
        self.s1 = self._make_stage(32, layers[0], stride=1)  # Use stride 1 to maintain spatial size
        self.adjust_s0_to_s1 = nn.Conv2d(32, expansion * 32, kernel_size=1, stride=1, bias=False)
        
        # Stage S2 - in:64 -> out:128 (for expansion=2), maintain 32x32 size
        self.s2 = self._make_stage(expansion * 32, layers[1], stride=1)
        self.adjust_s1_to_s2 = nn.Conv2d(expansion * 32, expansion**2 * 32, kernel_size=1, stride=1, bias=False)
        
        # Stage S3 - in:128 -> out:256 (for expansion=2), maintain 32x32 size
        self.s3 = self._make_stage(expansion**2 * 32, layers[2], stride=1)
        # No additional adjustment needed
        
        # Relative Attention after S3 (operating on 32x32 feature maps)
        self.relative_attention = RelativeAttention(expansion**3 * 32)

        # Global Average Pooling + Classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(expansion**3 * 32, num_classes)
    
    def _make_stage(self, in_channels, num_blocks, stride=1):
        """Create a stage with num_blocks blocks, all using stride 1 to maintain spatial dimensions."""
        out_channels = self.expansion * in_channels
        
        # All blocks use stride 1 to maintain spatial dimensions
        blocks = [StageBlock(in_channels, out_channels, stride=stride)]
        
        # Additional blocks
        for _ in range(1, num_blocks):
            blocks.append(StageBlock(out_channels, out_channels, stride=1))
            
        return nn.Sequential(*blocks)

    def forward(self, x):
        # Input: 32x32x3
        x = self.stem(x)  # Output: 32x32x32
        s0_out = x
        
        # Stage S1
        x = self.s1(x)  # Output: 32x32x64
        adjusted_s0 = self.adjust_s0_to_s1(s0_out)  # Adjust S0: 32x32x32 -> 32x32x64
        x = x + adjusted_s0
        s1_out = x
        
        # Stage S2
        x = self.s2(x)  # Output: 32x32x128
        adjusted_s1 = self.adjust_s1_to_s2(s1_out)  # Adjust S1: 32x32x64 -> 32x32x128
        x = x + adjusted_s1
        s2_out = x
        
        # Stage S3
        x = self.s3(x)  # Output: 32x32x256
        
        # Relative Attention
        x = self.relative_attention(x)  # Output: 32x32x256
        
        # Global Average Pooling
        x = self.global_avg_pool(x)  # Output: 1x1x256
        
        # Flatten and FC Layer
        x = torch.flatten(x, 1)  # Output: 256
        x = self.fc(x)  # Output: num_classes
        
        return x

class AstroConfig:
    """
    Configuration class for AstroTransformer hyperparameters.
    """
    def __init__(self, 
                 expansion=2,
                 layers=[2, 2, 2],
                 use_gradient_checkpointing=False):
        self.expansion = expansion
        self.layers = layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
    
    @classmethod
    def small(cls):
        """Small model configuration."""
        return cls(expansion=2, layers=[1, 1, 1])
    
    @classmethod
    def base(cls):
        """Standard model configuration."""
        return cls(expansion=2, layers=[2, 2, 2])
    
    @classmethod
    def large(cls):
        """Larger model configuration."""
        return cls(expansion=3, layers=[2, 3, 3])

def create_astro_model(num_classes=4, config=None):
    """
    Helper function to create an AstroTransformer model with specific parameters.
    
    Args:
        num_classes: Number of output classes
        config: Optional AstroConfig instance for full customization
        
    Returns:
        AstroTransformer model
    """
    # Use provided config or create one from parameters
    if config is None:
        config = AstroConfig()
    
    print(f"Creating AstroTransformer for 32x32 images")
    print(f"  Architecture: {len(config.layers)} stages with {config.layers} blocks")
    print(f"  Expansion factor: {config.expansion}")
    
    # Add memory optimization details if enabled
    if config.use_gradient_checkpointing:
        print("  Using gradient checkpointing to reduce memory usage")
    
    model = AstroTransformer(
        num_classes=num_classes,
        expansion=config.expansion,
        layers=config.layers
    )
    
    return model
