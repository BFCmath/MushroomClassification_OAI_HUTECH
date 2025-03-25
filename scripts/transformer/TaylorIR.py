import torch
import torch.nn as nn
import torch.nn.functional as F
from .PixT import PositionalEncoding

class TaylorWindowAttention(nn.Module):
    """
    Implements Direct-TaylorShift attention which uses a Taylor series approximation
    instead of the standard softmax for more efficient attention calculation.
    """
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(head_dim)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, num_heads, N, head_dim]
        
        # Apply scaling to query
        q = q * self.scale
        
        # Compute attention scores: A = QK^T / sqrt(d)
        A = q @ k.transpose(-2, -1)  # [B, num_heads, N, N]
        
        # Taylor-Softmax: (1 + A + 0.5 * A^2) / sum(1 + A + 0.5 * A^2)
        # More efficient than standard softmax for relatively small attention scores
        A2 = A ** 2
        numerator = 1 + A + 0.5 * A2
        denominator = numerator.sum(dim=-1, keepdim=True)  # Sum over last dimension
        attn = numerator / denominator  # [B, num_heads, N, N]
        attn = self.attn_drop(attn)
        
        # Apply attention weights to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TaylorTransformerBlock(nn.Module):
    """
    Transformer block using TaylorWindowAttention with pre-norm architecture.
    """
    def __init__(self, dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Taylor window attention
        self.attn = TaylorWindowAttention(dim, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TaylorIRClassifier(nn.Module):
    """
    Classification model using TaylorWindowAttention for 32x32x3 images.
    This model uses a convolutional feature extractor followed by Taylor-approximated
    transformer blocks for efficient global reasoning.
    """
    def __init__(self, num_classes=4, img_size=32, embed_dim=192, num_heads=6, 
                 num_layers=6, dim_feedforward=768, dropout=0.1, 
                 use_gradient_checkpointing=False):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Feature extraction with a simple stem (preserves spatial dimensions)
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # Image to sequence conversion (flatten spatial dimensions)
        self.num_patches = img_size * img_size
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            embed_dim, 
            max_len=self.num_patches + 1,  # +1 for CLS token
            dropout=dropout
        )
        
        # Transformer blocks with Taylor window attention
        self.transformer_blocks = nn.ModuleList([
            TaylorTransformerBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize linear layers and convolutions
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize classification token
        torch.nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Input: [batch_size, 3, H, W]
        batch_size = x.shape[0]
        
        # Feature extraction
        x = self.stem(x)  # [batch_size, embed_dim, H, W]
        
        # Reshape to sequence format
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [batch_size, H*W, embed_dim]
        
        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1+H*W, embed_dim]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        # Extract classification token
        x = x[:, 0]  # [batch_size, embed_dim]
        
        # Layer normalization and classification
        x = self.norm(x)
        x = self.classifier(x)
        
        return x

class TaylorConfig:
    """
    Configuration class for TaylorIR hyperparameters.
    """
    def __init__(self, 
                 img_size=32,
                 embed_dim=192, 
                 num_heads=6,
                 num_layers=6,
                 dim_feedforward=768,
                 dropout=0.1,
                 use_gradient_checkpointing=False):
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.use_gradient_checkpointing = use_gradient_checkpointing
    
    @classmethod
    def small(cls, img_size=32):
        """Small model configuration."""
        return cls(img_size=img_size, embed_dim=128, num_layers=4, num_heads=4)
    
    @classmethod
    def base(cls, img_size=32):
        """Standard model configuration."""
        return cls(img_size=img_size, embed_dim=192, num_layers=6, num_heads=6)
    
    @classmethod
    def large(cls, img_size=32):
        """Larger model configuration."""
        return cls(img_size=img_size, embed_dim=256, num_layers=8, num_heads=8)

def create_taylorir_model(num_classes=4, img_size=32, config=None):
    """
    Helper function to create a TaylorIR model with specific parameters.
    
    Args:
        num_classes: Number of output classes
        img_size: Size of input images (assumes square images)
        config: Optional TaylorConfig instance for full customization
        
    Returns:
        TaylorIRClassifier model
    """
    # Use provided config or create one from parameters
    if config is None:
        config = TaylorConfig(img_size=img_size)
    
    print(f"Creating TaylorIR classifier for {config.img_size}x{config.img_size} images")
    print(f"  Model: embed_dim={config.embed_dim}, heads={config.num_heads}, layers={config.num_layers}")
    
    # Add memory optimization details if enabled
    if config.use_gradient_checkpointing:
        print("  Using gradient checkpointing to reduce memory usage")
    
    model = TaylorIRClassifier(
        num_classes=num_classes,
        img_size=config.img_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        use_gradient_checkpointing=config.use_gradient_checkpointing
    )
    
    return model
