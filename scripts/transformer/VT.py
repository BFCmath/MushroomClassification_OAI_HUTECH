import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scripts.transformer.PixT import TransformerConfig, PositionalEncoding, TransformerEncoderBlock

class EnhancedResidualBlock(nn.Module):
    """
    Enhanced residual block with more complex feature extraction but
    without spatial downsampling, optimized for small images.
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        
        # Second conv with dilation for larger receptive field
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        
        # Third conv for deeper feature extraction
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Skip connection if dimensions change
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu_out = nn.ReLU(inplace=False)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu_out(out)
        
        return out

class SPDInspiredBlock(nn.Module):
    """
    Block inspired by SPDResNet but without downsampling spatial dimensions.
    Uses grouped convolutions and branch design for complex feature extraction.
    """
    def __init__(self, in_channels, out_channels, groups=4):
        super(SPDInspiredBlock, self).__init__()
        mid_channels = out_channels // 2
        
        # Branch 1: Basic pathway
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, 
                     groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )
        
        # Branch 2: Dilated pathway for larger receptive field
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                     padding=2, dilation=2, groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )
        
        # Combine branches
        self.combine = nn.Sequential(
            nn.Conv2d(mid_channels*2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        
        # Concatenate branches along channel dimension
        combined = torch.cat([branch1, branch2], dim=1)
        out = self.combine(combined)
        
        out += identity
        out = self.relu(out)
        
        return out

class EnhancedSPDBackend(nn.Module):
    """
    Enhanced backend inspired by SPDResNet but preserving spatial dimensions
    for small 32x32 images. Combines multiple types of residual blocks.
    """
    def __init__(self, in_channels=3, out_channels=192):
        super(EnhancedSPDBackend, self).__init__()
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        
        # Layer 1: Standard enhanced residual blocks
        self.layer1 = nn.Sequential(
            EnhancedResidualBlock(64, 96),
            EnhancedResidualBlock(96, 96, dilation=2)
        )
        
        # Layer 2: SPD-inspired blocks
        self.layer2 = nn.Sequential(
            SPDInspiredBlock(96, 128),
            SPDInspiredBlock(128, 160)
        )
        
        # Layer 3: Final feature refinement
        self.layer3 = EnhancedResidualBlock(160, out_channels)
        
        # Final feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        
        # Attention mechanism (SE-like) for channel calibration
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid()
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
    
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.refinement(x)
        
        # Apply channel attention
        attention = self.channel_attention(x)
        x = x * attention
        
        return x

class FilterTokenizer(nn.Module):
    """
    Filter-based tokenizer that converts feature maps to a smaller set of tokens.
    Uses learned attention weights to aggregate spatial information.
    """
    def __init__(self, C, L):
        super(FilterTokenizer, self).__init__()
        """
        Args:
            C (int): Number of input channels in the feature map.
            L (int): Number of visual tokens to produce (token length).
        """
        self.C = C  # Channel dimension
        self.L = L  # Token length
        
        # Attention projection matrix - computes importance of each spatial position for each token
        self.attention = nn.Conv2d(C, L, kernel_size=1)
        
        # Value projection to compute token features from input features
        self.value_proj = nn.Conv2d(C, C, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map of shape [batch_size, C, H, W]
            
        Returns:
            torch.Tensor: Tokens of shape [batch_size, L, C]
        """
        batch_size, C, H, W = x.shape
        
        # Project input features to values
        values = self.value_proj(x)  # [batch_size, C, H, W]
        
        # Compute attention weights
        attn_weights = self.attention(x)  # [batch_size, L, H, W]
        
        # Reshape and apply softmax across spatial dimensions
        attn_weights = attn_weights.view(batch_size, self.L, -1)  # [batch_size, L, H*W]
        attn_weights = F.softmax(attn_weights, dim=2)  # Softmax across spatial dim
        
        # Reshape values for matrix multiplication
        values = values.view(batch_size, C, -1)  # [batch_size, C, H*W]
        values = values.permute(0, 2, 1)  # [batch_size, H*W, C]
        
        # Compute token features using attention
        tokens = torch.bmm(attn_weights, values)  # [batch_size, L, C]
        
        return tokens

class VisualTransformer(nn.Module):
    """
    Visual Transformer (VT) model optimized for small 32x32 images.
    
    Features:
    1. Enhanced SPD-inspired backend without downsampling to preserve spatial information
    2. Filter-based tokenizer to reduce sequence length to manageable size
    3. Transformer encoder with self-attention for global feature extraction
    4. Uses classification token for final prediction
    """
    def __init__(self, num_classes=10, backend_channels=192, token_length=48, 
                 d_model=192, nhead=8, num_layers=6, dim_feedforward=768, 
                 dropout=0.1, img_size=32, use_gradient_checkpointing=False):
        super(VisualTransformer, self).__init__()
        
        self.img_size = img_size
        self.d_model = d_model
        self.token_length = token_length
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Enhanced SPD-inspired backend without downsampling
        self.backbone = EnhancedSPDBackend(in_channels=3, out_channels=backend_channels)
        
        # Project backend channels to transformer dimension if different
        self.channel_proj = None
        if backend_channels != d_model:
            self.channel_proj = nn.Linear(backend_channels, d_model)
        
        # Filter tokenizer to reduce sequence length
        self.tokenizer = FilterTokenizer(C=backend_channels, L=token_length)
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=token_length + 1,  # +1 for CLS token
            dropout=dropout
        )
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize transformer layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize classification token
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Input: [batch_size, channels, height, width]
        batch_size = x.size(0)
        
        # 1. Process through ResNet backend
        features = self.backbone(x)  # [batch_size, backend_channels, height, width]
        
        # 2. Convert features to tokens with filter tokenizer
        tokens = self.tokenizer(features)  # [batch_size, token_length, backend_channels]
        
        # 3. Project token dimension if needed
        if self.channel_proj is not None:
            tokens = self.channel_proj(tokens)  # [batch_size, token_length, d_model]
        
        # 4. Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_model]
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # [batch_size, 1+token_length, d_model]
        
        # 5. Add positional encoding
        tokens = self.positional_encoding(tokens)  # [batch_size, 1+token_length, d_model]
        
        # 6. Process through transformer blocks
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens)
            else:
                tokens = block(tokens)
        
        # 7. Extract classification token
        cls_token = tokens[:, 0]  # [batch_size, d_model]
        
        # 8. Apply classification head
        cls_token = self.norm(cls_token)
        output = self.classifier(cls_token)
        
        return output

def create_vt_model(num_classes=10, img_size=32, config=None):
    """
    Create a Visual Transformer model with the given configuration.
    
    Args:
        num_classes: Number of output classes
        img_size: Size of input images (assumes square images)
        config: Optional TransformerConfig instance for customization
        
    Returns:
        VisualTransformer model
    """
    # Use provided config or create a default one
    if config is None:
        config = TransformerConfig(
            img_size=img_size,
            d_model=192,
            nhead=8,
            num_layers=6,
            dropout=0.1,
            use_gradient_checkpointing=False
        )
    
    # Determine token length based on image size
    # For 32x32 images, use 48 tokens
    # For larger images, scale accordingly
    token_length = 48
    if img_size > 32:
        # Scale tokens proportionally, but cap at reasonable values
        token_length = min(int(48 * (img_size / 32)), 256)
    
    # Use higher backend channels with the enhanced backbone
    backend_channels = config.d_model
    backend_channels = max(128, backend_channels)  # Ensure at least 128 channels
    
    print(f"Creating Enhanced Visual Transformer for {img_size}x{img_size} images")
    print(f"  Backend: SPD-inspired architecture with {backend_channels} channels")
    print(f"  Tokenizer: Converting to {token_length} tokens")
    print(f"  Transformer: d_model={config.d_model}, heads={config.nhead}, layers={config.num_layers}")
    
    if hasattr(config, 'use_gradient_checkpointing') and config.use_gradient_checkpointing:
        print("  Using gradient checkpointing to reduce memory usage")
    
    model = VisualTransformer(
        num_classes=num_classes,
        backend_channels=backend_channels,
        token_length=token_length,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        img_size=img_size,
        use_gradient_checkpointing=getattr(config, 'use_gradient_checkpointing', False)
    )
    
    return model

# Example factory functions similar to PixT for easy model creation
def create_vt_tiny(num_classes=10, img_size=32):
    """Create a tiny Visual Transformer model."""
    config = TransformerConfig(
        img_size=img_size,
        d_model=96,
        nhead=4,
        num_layers=4,
        dropout=0.1
    )
    return create_vt_model(num_classes, img_size, config)

def create_vt_small(num_classes=10, img_size=32):
    """Create a small Visual Transformer model."""
    config = TransformerConfig(
        img_size=img_size,
        d_model=192,
        nhead=6,
        num_layers=6,
        dropout=0.1
    )
    return create_vt_model(num_classes, img_size, config)

def create_vt_base(num_classes=10, img_size=32):
    """Create a base Visual Transformer model."""
    config = TransformerConfig(
        img_size=img_size,
        d_model=256,
        nhead=8,
        num_layers=8,
        dropout=0.1
    )
    return create_vt_model(num_classes, img_size, config)

def create_vt_memory_efficient(num_classes=10, img_size=32):
    """Create a memory-efficient Visual Transformer model."""
    config = TransformerConfig(
        img_size=img_size,
        d_model=192,
        nhead=6,
        num_layers=6,
        dropout=0.1,
        use_gradient_checkpointing=True
    )
    return create_vt_model(num_classes, img_size, config)
