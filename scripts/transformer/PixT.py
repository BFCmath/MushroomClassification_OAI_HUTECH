import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerConfig:
    """
    Configuration class for PixelTransformer hyperparameters.
    
    This allows easy customization of transformer architecture without 
    changing the model implementation.
    """
    def __init__(self, 
                 img_size=32,
                 d_model=128, 
                 nhead=None,  # Will be auto-calculated if None
                 num_layers=6,
                 dim_feedforward=None,  # Will be auto-calculated if None
                 dropout=0.1,
                 activation="gelu",
                 # Add new memory efficiency parameters
                 use_gradient_checkpointing=False,
                 sequence_reduction_factor=1,
                 share_layer_params=False,
                 use_sequence_downsampling=False):
        self.img_size = img_size
        self.d_model = d_model
        # Auto-calculate number of heads if not specified
        self.nhead = nhead if nhead is not None else max(4, d_model // 32)
        self.num_layers = num_layers
        # Auto-calculate feedforward dimension if not specified (typical transformer uses 4x d_model)
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else d_model * 4
        self.dropout = dropout
        self.activation = activation
        
        # Memory optimization parameters
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.sequence_reduction_factor = sequence_reduction_factor
        self.share_layer_params = share_layer_params
        self.use_sequence_downsampling = use_sequence_downsampling
    
    @classmethod
    def tiny(cls, img_size=32):
        """Small and efficient transformer configuration."""
        return cls(img_size=img_size, d_model=96, num_layers=4, dropout=0.1)
    
    @classmethod
    def small(cls, img_size=32):
        """Balanced transformer configuration for small images."""
        return cls(img_size=img_size, d_model=192, num_layers=6, dropout=0.1)
    
    @classmethod
    def base(cls, img_size=32):
        """Larger transformer configuration with more capacity."""
        return cls(img_size=img_size, d_model=256, num_layers=8, dropout=0.1)
    
    @classmethod
    def memory_efficient(cls, img_size=32):
        """Memory-efficient configuration with optimizations for reduced VRAM usage."""
        return cls(
            img_size=img_size, 
            d_model=192,              # Still decent model capacity
            num_layers=6,             # Keep layer depth for capacity
            dropout=0.1,
            use_gradient_checkpointing=True,  # Enable checkpointing
            share_layer_params=True,          # Share parameters between layers
            use_sequence_downsampling=True    # Use sequence reduction
        )

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer sequence.
    Works with flattened pixels from images, treating each pixel as a token position.
    """
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create standard 1D positional encoding matrix
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with multi-head self-attention and feed-forward networks.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection and layer norm
        # x shape: [batch_size, seq_len, d_model]
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)
        
        return x

# Add a sequence reduction module
class SequencePooling(nn.Module):
    """Reduces sequence length by pooling nearby tokens."""
    def __init__(self, d_model, reduction_factor=2, mode='mean'):
        super(SequencePooling, self).__init__()
        self.reduction_factor = reduction_factor
        self.mode = mode
        if mode == 'learned':
            self.projection = nn.Linear(d_model * reduction_factor, d_model)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # Check if the sequence length is divisible by reduction factor
        if seq_len % self.reduction_factor != 0:
            # Add padding tokens if needed
            pad_len = self.reduction_factor - (seq_len % self.reduction_factor)
            padding = torch.zeros(batch_size, pad_len, d_model, device=x.device)
            x = torch.cat([x, padding], dim=1)
            seq_len = x.shape[1]
        
        # Reshape to group tokens
        x = x.reshape(batch_size, seq_len // self.reduction_factor, self.reduction_factor, d_model)
        
        # Pool tokens
        if self.mode == 'mean':
            # Mean pooling (most efficient)
            x = torch.mean(x, dim=2)  # [batch_size, seq_len//reduction_factor, d_model]
        elif self.mode == 'max':
            # Max pooling
            x = torch.max(x, dim=2)[0]  # [batch_size, seq_len//reduction_factor, d_model]
        elif self.mode == 'learned':
            # Learned pooling
            x = x.reshape(batch_size, seq_len // self.reduction_factor, self.reduction_factor * d_model)
            x = self.projection(x)  # [batch_size, seq_len//reduction_factor, d_model]
        
        return x

class PixelTransformer(nn.Module):
    """
    Pixel Transformer (PixT) that treats each pixel as a token for 32x32 images.
    Memory-optimized version with gradient checkpointing and other efficiency options.
    
    Unlike Vision Transformer (ViT) which divides images into patches,
    PixT works directly with individual pixels as tokens, making it suitable
    for already-small images like 32x32.
    
    Flowchart:
    Input (32x32x3)
    → Pixel Embedding (1024x128)
    → Positional Encoding
    → Transformer Blocks
    → Classification Token
    → MLP Head
    → Output (num_classes)
    """
    def __init__(self, num_classes=10, d_model=128, nhead=8, num_layers=6, 
                dim_feedforward=512, dropout=0.1, img_size=32,
                use_gradient_checkpointing=False,
                sequence_reduction_factor=1,
                share_layer_params=False,
                use_sequence_downsampling=False):
        super(PixelTransformer, self).__init__()
        
        self.img_size = img_size
        self.num_pixels = img_size * img_size
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.share_layer_params = share_layer_params
        
        # Pixel embedding - projects each RGB pixel to d_model dimensions
        self.pixel_embedding = nn.Linear(3, d_model)
        
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
            effective_seq_len = (self.num_pixels // sequence_reduction_factor) + 1  # +1 for CLS token
        else:
            effective_seq_len = self.num_pixels + 1  # Original length +1 for CLS token
            
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
        
        # Initialize classification token
        torch.nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Input: [batch_size, channels, height, width]
        batch_size = x.size(0)
        
        # 1. Reshape to [batch_size, height*width, channels]
        x = x.permute(0, 2, 3, 1)  # [batch_size, height, width, channels]
        x = x.reshape(batch_size, self.num_pixels, 3)  # [batch_size, height*width, channels]
        
        # 2. Project RGB values to embedding dimension
        x = self.pixel_embedding(x)  # [batch_size, height*width, d_model]
        
        # 3. Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1+num_pixels, d_model]
        
        # 4. Add positional encoding
        x = self.positional_encoding(x)  # [batch_size, 1+num_pixels, d_model]
        
        # 5. Apply sequence reduction if enabled
        if self.sequence_reduction is not None:
            x = self.sequence_reduction(x)
        
        # 6. Pass through transformer blocks with or without gradient checkpointing
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        # 7. Extract classification token
        x = x[:, 0]  # [batch_size, d_model]
        
        # 8. Classification
        x = self.norm(x)
        x = self.classifier(x)
        
        return x

# Memory-efficient PixelTransformer variant
class MemoryEfficientPixT(nn.Module):
    """
    Memory-efficient variant of PixelTransformer with early spatial downsampling.
    This model drastically reduces memory usage by downsampling the spatial dimensions
    before processing with the transformer, reducing sequence length from 1024 to 256.
    """
    def __init__(self, num_classes=10, d_model=192, nhead=6, num_layers=6, 
                 dim_feedforward=768, dropout=0.1, img_size=32):
        super(MemoryEfficientPixT, self).__init__()
        
        self.img_size = img_size
        self.d_model = d_model
        
        # Initial convolutional layers to reduce spatial dimensions (32x32 -> 16x16)
        self.spatial_reduction = nn.Sequential(
            nn.Conv2d(3, d_model//2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model//2),
            nn.GELU(),
            nn.Conv2d(d_model//2, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        
        # Now we have 16x16 feature map with d_model channels
        # Total sequence length: 16x16 = 256 tokens
        self.reduced_pixels = (img_size // 2) ** 2
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding for reduced sequence length
        self.positional_encoding = PositionalEncoding(d_model, max_len=self.reduced_pixels+1, dropout=dropout)
        
        # Transformer encoder blocks with gradient checkpointing
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize convolution layers
        for m in self.spatial_reduction.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Initialize transformer and linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        # Initialize classification token
        torch.nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Input: [batch_size, channels, height, width]
        batch_size = x.size(0)
        
        # 1. Apply spatial reduction with convolutions
        x = self.spatial_reduction(x)  # [batch_size, d_model, height/2, width/2]
        
        # 2. Reshape to sequence format
        x = x.permute(0, 2, 3, 1)  # [batch_size, height/2, width/2, d_model]
        x = x.reshape(batch_size, self.reduced_pixels, self.d_model)  # [batch_size, reduced_pixels, d_model]
        
        # 3. Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1+reduced_pixels, d_model]
        
        # 4. Add positional encoding
        x = self.positional_encoding(x)
        
        # 5. Apply transformer blocks with gradient checkpointing
        for block in self.transformer_blocks:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        # 6. Extract classification token
        x = x[:, 0]  # [batch_size, d_model]
        
        # 7. Apply classification head
        x = self.norm(x)
        x = self.classifier(x)
        
        return x

def create_pixt_model(num_classes=10, img_size=32, d_model=128, dropout_rate=0.1, config=None):
    """
    Helper function to create a PixT model with specific parameters.
    
    Args:
        num_classes: Number of output classes
        img_size: Size of input images (assumes square images)
        d_model: Size of embedding dimension
        dropout_rate: Dropout rate for regularization
        config: Optional TransformerConfig instance for full customization
        
    Returns:
        PixelTransformer model
    """
    # Use provided config or create one from parameters
    if config is None:
        config = TransformerConfig(
            img_size=img_size,
            d_model=d_model,
            dropout=dropout_rate
        )
    
    # Use the more memory-efficient model if requested
    if hasattr(config, 'use_sequence_downsampling') and config.use_sequence_downsampling and img_size >= 32:
        print(f"Creating Memory-Efficient PixelTransformer for {config.img_size}x{config.img_size} images")
        print(f"  Using spatial downsampling to reduce sequence length")
        print(f"  Model: d_model={config.d_model}, heads={config.nhead}, layers={config.num_layers}")
        
        model = MemoryEfficientPixT(
            num_classes=num_classes,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            img_size=config.img_size
        )
    else:
        print(f"Creating PixelTransformer for {config.img_size}x{config.img_size} images with {config.d_model}-dim embeddings")
        print(f"  Layers: {config.num_layers}, Heads: {config.nhead}, FF dim: {config.dim_feedforward}")
        
        # Add memory optimization details if enabled
        if hasattr(config, 'use_gradient_checkpointing') and config.use_gradient_checkpointing:
            print("  Using gradient checkpointing to reduce memory usage")
        if hasattr(config, 'sequence_reduction_factor') and config.sequence_reduction_factor > 1:
            print(f"  Using sequence reduction with factor {config.sequence_reduction_factor}")
        if hasattr(config, 'share_layer_params') and config.share_layer_params:
            print("  Using parameter sharing between transformer layers")
            
        model = PixelTransformer(
            num_classes=num_classes,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            img_size=config.img_size,
            use_gradient_checkpointing=getattr(config, 'use_gradient_checkpointing', False),
            sequence_reduction_factor=getattr(config, 'sequence_reduction_factor', 1),
            share_layer_params=getattr(config, 'share_layer_params', False),
            use_sequence_downsampling=getattr(config, 'use_sequence_downsampling', False)
        )
    
    return model
