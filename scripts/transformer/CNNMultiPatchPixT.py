import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .PixT import TransformerEncoderBlock, PositionalEncoding, SequencePooling
from .PatchPixT import PatchEmbedding, PatchPixTConfig

class CNNMultiPatchPixTConfig:
    """
    Configuration class for CNNMultiPatchPixT hyperparameters.
    
    This model combines a CNN backbone with multiple patch-size transformers.
    """
    def __init__(self, 
                 img_size=32,
                 patch_sizes=[1, 2, 4],  # Default patch sizes
                 d_model=128, 
                 nhead=None,  # Will be auto-calculated if None
                 num_layers=6,
                 dim_feedforward=None,  # Will be auto-calculated if None
                 dropout=0.1,
                 activation="gelu",
                 fusion_type="concat",  # How to fuse features: "concat", "weighted_sum", "attention"
                 growth_rate=12,  # Growth rate for DenseNet layers
                 use_gradient_checkpointing=False,
                 share_layer_params=False,
                 cnn_dropout=0.1):  # Dropout for CNN layers
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.d_model = d_model
        # Auto-calculate number of heads if not specified
        self.nhead = nhead if nhead is not None else max(4, d_model // 32)
        self.num_layers = num_layers
        # Auto-calculate feedforward dimension if not specified
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else d_model * 4
        self.dropout = dropout
        self.activation = activation
        self.fusion_type = fusion_type
        self.growth_rate = growth_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.share_layer_params = share_layer_params
        self.cnn_dropout = cnn_dropout
    
    @classmethod
    def small(cls, img_size=32):
        """Small model configuration."""
        return cls(img_size=img_size, d_model=128, num_layers=4, growth_rate=12)
    
    @classmethod
    def base(cls, img_size=32):
        """Standard model configuration."""
        return cls(img_size=img_size, d_model=192, num_layers=6, growth_rate=24)
    
    @classmethod
    def large(cls, img_size=32):
        """Larger model configuration."""
        return cls(img_size=img_size, d_model=256, num_layers=8, growth_rate=32)

class DenseLayer(nn.Module):
    """
    Basic building block for DenseNet.
    """
    def __init__(self, in_channels, growth_rate, dropout=0.1):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    """
    DenseNet block consisting of multiple DenseLayers.
    """
    def __init__(self, in_channels, growth_rate, num_layers, dropout=0.1):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, dropout))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class TransitionBlock(nn.Module):
    """
    Transition layer between DenseNet blocks.
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x

class CNNBackbone(nn.Module):
    """
    DenseNet-like CNN backbone that provides features at multiple scales.
    """
    def __init__(self, growth_rate=12, dropout=0.1):
        super(CNNBackbone, self).__init__()
        
        # Initial convolution to create feature maps
        self.initial_conv = nn.Conv2d(3, 2 * growth_rate, kernel_size=3, padding=1, bias=False)
        
        # Block 1: 32x32 -> 32x32 (maintain spatial dimensions)
        self.block1 = DenseBlock(2 * growth_rate, growth_rate, num_layers=4, dropout=dropout)
        in_channels1 = 2 * growth_rate + 4 * growth_rate
        self.trans1 = TransitionBlock(in_channels1, in_channels1 // 2, dropout=dropout)
        
        # Block 2: 16x16 -> 16x16
        self.block2 = DenseBlock(in_channels1 // 2, growth_rate, num_layers=6, dropout=dropout)
        in_channels2 = in_channels1 // 2 + 6 * growth_rate
        self.trans2 = TransitionBlock(in_channels2, in_channels2 // 2, dropout=dropout)
        
        # Block 3: 8x8 -> 8x8
        self.block3 = DenseBlock(in_channels2 // 2, growth_rate, num_layers=8, dropout=dropout)
        
        # Store output channels for reference
        self.out_channels1 = in_channels1
        self.out_channels2 = in_channels2
        self.out_channels3 = in_channels2 // 2 + 8 * growth_rate
    
    def forward(self, x):
        # Initial convolution: (B, 3, 32, 32) -> (B, 2*growth_rate, 32, 32)
        x = self.initial_conv(x)
        
        # Block 1
        x1 = self.block1(x)  # (B, out_channels1, 32, 32)
        x = self.trans1(x1)  # (B, out_channels1//2, 16, 16)
        
        # Block 2
        x2 = self.block2(x)  # (B, out_channels2, 16, 16)
        x = self.trans2(x2)  # (B, out_channels2//2, 8, 8)
        
        # Block 3
        x3 = self.block3(x)  # (B, out_channels3, 8, 8)
        
        return x1, x2, x3

class CNNMultiPatchPixT(nn.Module):
    """
    CNN-backed Multi-Scale Patch Transformer with simplified architecture.
    
    This model uses a DenseNet-like CNN backbone to extract features at multiple 
    scales, applies uniform non-overlapping patching to each feature map, and
    uses mean pooling to extract features instead of CLS tokens.
    """
    def __init__(self, num_classes=10, img_size=32, patch_sizes=[1, 2, 4], 
                 d_model=128, nhead=8, num_layers=6, dim_feedforward=512, 
                 dropout=0.1, fusion_type="concat", growth_rate=12, 
                 use_gradient_checkpointing=False, share_layer_params=False,
                 cnn_dropout=0.1):
        super(CNNMultiPatchPixT, self).__init__()
        
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.d_model = d_model
        self.fusion_type = fusion_type
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # 1. CNN Backbone for extracting multi-scale features
        self.backbone = CNNBackbone(growth_rate=growth_rate, dropout=cnn_dropout)
        
        # 2. Patch embeddings for each branch - will extract non-overlapping patches
        # and project them to d_model dimension
        self.patch_embeddings = nn.ModuleList([
            # Branch 1: 4×4 patches from 32×32 feature map -> 64 tokens
            nn.Conv2d(self.backbone.out_channels1, d_model, kernel_size=4, stride=4, padding=0),
            
            # Branch 2: 2×2 patches from 16×16 feature map -> 64 tokens
            nn.Conv2d(self.backbone.out_channels2, d_model, kernel_size=2, stride=2, padding=0),
            
            # Branch 3: 1×1 patches from 8×8 feature map -> 64 tokens
            nn.Conv2d(self.backbone.out_channels3, d_model, kernel_size=1, stride=1, padding=0)
        ])
        
        # Each branch will have 64 tokens
        self.seq_length = 64
        
        # 3. Positional encodings for all branches (all have same sequence length)
        self.positional_encoding = PositionalEncoding(d_model, max_len=self.seq_length, dropout=dropout)
        
        # 4. Feature fusion mechanism
        if fusion_type == "concat":
            # Concatenate features from all branches
            fusion_dim = d_model * 3  # 3 branches
            self.fusion_layer = nn.Linear(fusion_dim, d_model)
        elif fusion_type == "attention":
            # Use attention to weight different branches
            self.fusion_layer = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
        elif fusion_type == "weighted_sum":
            # Weighted sum with learnable weights
            self.fusion_weights = nn.Parameter(torch.ones(3))  # 3 branches
            fusion_dim = d_model
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # 5. Transformer layers for fused features
        if share_layer_params:
            shared_block = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            self.transformer_blocks = nn.ModuleList([shared_block] * num_layers)
        else:
            self.transformer_blocks = nn.ModuleList([
                TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])
        
        # 6. Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # 7. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
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
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Extract multi-scale features from CNN backbone
        x1, x2, x3 = self.backbone(x)
        
        # List of features with corresponding patch embeddings
        features = [x1, x2, x3]
        
        branch_outputs = []
        
        # 2. Process each branch without transformer layers
        for i, (feature, patch_embedding) in enumerate(
            zip(features, self.patch_embeddings)
        ):
            # Apply patch embedding - this creates 64 tokens for each branch
            # through non-overlapping patches (4×4, 2×2, and 1×1)
            embedded = patch_embedding(feature)  # [batch_size, d_model, 8, 8]
            
            # Reshape to sequence format
            embedded = embedded.flatten(2).transpose(1, 2)  # [batch_size, 64, d_model]
            
            # Add positional encoding
            embedded = self.positional_encoding(embedded)
            
            # Extract features using mean pooling instead of CLS token
            branch_repr = embedded.mean(dim=1)  # [batch_size, d_model]
            branch_outputs.append(branch_repr)
        
        # 3. Fuse features from all branches
        if self.fusion_type == "concat":
            # Concatenate branch outputs
            fused = torch.cat(branch_outputs, dim=1)  # [batch_size, 3*d_model]
            fused = self.fusion_layer(fused)  # [batch_size, d_model]
        elif self.fusion_type == "attention":
            # Use first branch output as query
            query = branch_outputs[0].unsqueeze(1)  # [batch_size, 1, d_model]
            # Concatenate other branch outputs as keys and values
            keys = torch.stack(branch_outputs[1:], dim=1)  # [batch_size, 2, d_model]
            fused, _ = self.fusion_layer(query, keys, keys)
            fused = fused.squeeze(1)  # [batch_size, d_model]
        elif self.fusion_type == "weighted_sum":
            # Weighted sum with learnable weights
            weighted_outputs = [output * weight for output, weight in zip(branch_outputs, self.fusion_weights)]
            fused = sum(weighted_outputs)  # [batch_size, d_model]
        
        # 4. Process the fused representation
        fused = fused.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 5. Process through transformer blocks
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                fused = torch.utils.checkpoint.checkpoint(block, fused)
            else:
                fused = block(fused)
        
        # 6. Extract final representation
        fused = fused.squeeze(1)  # [batch_size, d_model]
        fused = self.norm(fused)
        
        # 7. Classification
        output = self.classifier(fused)
        
        return output

def create_cnn_multipatch_pixt_model(
    num_classes=10,
    img_size=32,
    patch_sizes=[1, 2, 4],
    d_model=128,
    dropout_rate=0.1,
    config=None
):
    """
    Helper function to create a CNNMultiPatchPixT model with specific parameters.
    
    Args:
        num_classes: Number of output classes
        img_size: Size of input images (assumes square images)
        patch_sizes: List of patch sizes for different branches
        d_model: Size of embedding dimension
        dropout_rate: Dropout rate for regularization
        config: Optional CNNMultiPatchPixTConfig instance for full customization
        
    Returns:
        CNNMultiPatchPixT model
    """
    # Use provided config or create one from parameters
    if config is None:
        config = CNNMultiPatchPixTConfig(
            img_size=img_size,
            patch_sizes=patch_sizes,
            d_model=d_model,
            dropout=dropout_rate
        )
    
    print(f"Creating CNNMultiPatchPixT model with CNN backbone and patch sizes {config.patch_sizes}")
    print(f"  CNN backbone with growth rate: {config.growth_rate}")
    print(f"  Transformer: d_model={config.d_model}, heads={config.nhead}, layers={config.num_layers}")
    print(f"  Using fusion type: {config.fusion_type}")
    
    # Print memory optimization details if enabled
    if config.use_gradient_checkpointing:
        print("  Using gradient checkpointing to reduce memory usage")
    if config.share_layer_params:
        print("  Using parameter sharing between transformer layers")
    
    model = CNNMultiPatchPixT(
        num_classes=num_classes,
        img_size=config.img_size,
        patch_sizes=config.patch_sizes,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        fusion_type=config.fusion_type,
        growth_rate=config.growth_rate,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        share_layer_params=config.share_layer_params,
        cnn_dropout=config.cnn_dropout
    )
    
    return model
