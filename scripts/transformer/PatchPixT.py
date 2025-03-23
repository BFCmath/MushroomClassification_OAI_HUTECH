import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PixT import (
    TransformerEncoderBlock,
    PositionalEncoding,
    TransformerConfig,
    SequencePooling
)

class PatchPixTConfig(TransformerConfig):
    """
    Configuration class for PatchPixT hyperparameters.
    Extends TransformerConfig with patch-specific settings.
    """
    def __init__(self, 
                 img_size=32,
                 patch_size=4,  # Default patch size
                 d_model=128, 
                 nhead=None,
                 num_layers=6,
                 dim_feedforward=None,
                 dropout=0.1,
                 activation="gelu",
                 use_gradient_checkpointing=False,
                 sequence_reduction_factor=1,
                 share_layer_params=False,
                 use_sequence_downsampling=False):
        super().__init__(
            img_size=img_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            use_gradient_checkpointing=use_gradient_checkpointing,
            sequence_reduction_factor=sequence_reduction_factor,
            share_layer_params=share_layer_params,
            use_sequence_downsampling=use_sequence_downsampling
        )
        self.patch_size = patch_size
    
    @classmethod
    def tiny_2x2(cls, img_size=32):
        """Small transformer with 2x2 patches."""
        return cls(img_size=img_size, d_model=96, num_layers=4, dropout=0.1, patch_size=2)
    
    @classmethod
    def small_4x4(cls, img_size=32):
        """Balanced transformer with 4x4 patches."""
        return cls(img_size=img_size, d_model=192, num_layers=6, dropout=0.1, patch_size=4)
    
    @classmethod
    def base_8x8(cls, img_size=32):
        """Larger transformer with 8x8 patches."""
        return cls(img_size=img_size, d_model=256, num_layers=8, dropout=0.1, patch_size=8)

class PatchEmbedding(nn.Module):
    """
    Patch embedding layer that converts image patches to embeddings.
    
    For example, with a patch size of 4, a 32x32 image would be divided into 
    64 patches of size 4x4, each embedded into a vector of dimension d_model.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, d_model=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Make sure img_size is divisible by patch_size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        
        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional layer for patch embedding
        # This projects each patch to d_model dimensions
        self.proj = nn.Conv2d(
            in_channels, 
            d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        # Apply patch projection
        x = self.proj(x)  # [batch_size, d_model, height/patch_size, width/patch_size]
        
        # Reshape to sequence format
        batch_size, d_model, h, w = x.shape
        x = x.flatten(2)  # [batch_size, d_model, height*width/patch_size²]
        x = x.transpose(1, 2)  # [batch_size, height*width/patch_size², d_model]
        
        return x

class PatchPixT(nn.Module):
    """
    Patch Pixel Transformer (PatchPixT) that treats patches of pixels as tokens.
    
    This model divides an image into patches (e.g., 2x2, 4x4, or 8x8) and processes
    each patch as a token, reducing sequence length compared to PixT while
    maintaining spatial information within patches.
    
    Flowchart:
    Input (32x32x3)
    → Patch Embedding (e.g., 64 patches for 4x4 patch size)
    → Positional Encoding
    → Transformer Blocks
    → Classification Token
    → MLP Head
    → Output (num_classes)
    """
    def __init__(self, num_classes=10, img_size=32, patch_size=4, d_model=128, 
                 nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1,
                 use_gradient_checkpointing=False,
                 sequence_reduction_factor=1,
                 share_layer_params=False,
                 use_sequence_downsampling=False):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.share_layer_params = share_layer_params
        
        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            d_model=d_model
        )
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Sequence reduction if enabled
        self.sequence_reduction = None
        if use_sequence_downsampling and sequence_reduction_factor > 1:
            # Apply sequence reduction after adding CLS token and positional encoding
            self.sequence_reduction = SequencePooling(
                d_model, 
                reduction_factor=sequence_reduction_factor,
                mode='mean'
            )
            # Calculate effective sequence length after downsampling
            effective_seq_len = (self.num_patches // sequence_reduction_factor) + 1  # +1 for CLS token
        else:
            effective_seq_len = self.num_patches + 1  # Original length +1 for CLS token
            
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model, 
            max_len=effective_seq_len, 
            dropout=dropout
        )
        
        # Create transformer blocks with parameter sharing if enabled
        if share_layer_params:
            # Create a single transformer block to be reused
            shared_block = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            # Create a list that refers to the same block multiple times
            self.transformer_blocks = nn.ModuleList([shared_block] * num_layers)
        else:
            # Create separate transformer blocks as usual
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
        self._init_weights()
        
    def _init_weights(self):
        # Initialize linear layers
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
        # Input: [batch_size, channels, height, width]
        batch_size = x.size(0)
        
        # 1. Apply patch embedding
        x = self.patch_embedding(x)  # [batch_size, num_patches, d_model]
        
        # 2. Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1+num_patches, d_model]
        
        # 3. Add positional encoding
        x = self.positional_encoding(x)  # [batch_size, 1+num_patches, d_model]
        
        # 4. Apply sequence reduction if enabled
        if self.sequence_reduction is not None:
            x = self.sequence_reduction(x)
        
        # 5. Pass through transformer blocks with or without gradient checkpointing
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        # 6. Extract classification token
        x = x[:, 0]  # [batch_size, d_model]
        
        # 7. Classification
        x = self.norm(x)
        x = self.classifier(x)
        
        return x

def create_patchpixt_model(
    num_classes=10, 
    img_size=32, 
    patch_size=4, 
    d_model=128, 
    dropout_rate=0.1, 
    config=None
):
    """
    Helper function to create a PatchPixT model with specific parameters.
    
    Args:
        num_classes: Number of output classes
        img_size: Size of input images (assumes square images)
        patch_size: Size of patches (2, 4, or 8 typically)
        d_model: Size of embedding dimension
        dropout_rate: Dropout rate for regularization
        config: Optional PatchPixTConfig instance for full customization
        
    Returns:
        PatchPixT model
    """
    # Use provided config or create one from parameters
    if config is None:
        config = PatchPixTConfig(
            img_size=img_size,
            patch_size=patch_size,
            d_model=d_model,
            dropout=dropout_rate
        )
    
    print(f"Creating PatchPixT with {config.patch_size}x{config.patch_size} patches for {config.img_size}x{config.img_size} images")
    print(f"  Resulting in {(config.img_size // config.patch_size) ** 2} tokens (plus 1 cls token)")
    print(f"  Model: d_model={config.d_model}, heads={config.nhead}, layers={config.num_layers}")
    
    # Add memory optimization details if enabled
    if config.use_gradient_checkpointing:
        print("  Using gradient checkpointing to reduce memory usage")
    if config.sequence_reduction_factor > 1:
        print(f"  Using sequence reduction with factor {config.sequence_reduction_factor}")
    if config.share_layer_params:
        print("  Using parameter sharing between transformer layers")
        
    model = PatchPixT(
        num_classes=num_classes,
        img_size=config.img_size,
        patch_size=config.patch_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        sequence_reduction_factor=config.sequence_reduction_factor,
        share_layer_params=config.share_layer_params,
        use_sequence_downsampling=config.use_sequence_downsampling
    )
    
    return model
