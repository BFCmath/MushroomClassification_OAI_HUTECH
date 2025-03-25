# CNNMultiPatchPixT (Mean Pooling Fusion)

This model combines a CNN backbone with multi-scale feature extraction. CNN features at different resolutions are patched to create uniform token sequences and mean-pooled before fusion.

## Detailed Architecture Flowchart

```mermaid
graph TD
    Input["Input: 32×32×3 RGB Image"] --> CNN["CNN Backbone (DenseNet-style)"]
    
    CNN --> B1["Block 1 Features<br>(32×32×68)"]
    CNN --> B2["Block 2 Features<br>(16×16×90)"]
    CNN --> B3["Block 3 Features<br>(8×8×112)"]
    
    B1 --> Patch1["4×4 Patching<br>Conv2d k=4,s=4<br>(batch, d_model, 8, 8)"]
    B2 --> Patch2["2×2 Patching<br>Conv2d k=2,s=2<br>(batch, d_model, 8, 8)"]
    B3 --> Patch3["1×1 Patching<br>Conv2d k=1,s=1<br>(batch, d_model, 8, 8)"]
    
    Patch1 --> Reshape1["Reshape<br>(batch, 64, d_model)"]
    Patch2 --> Reshape2["Reshape<br>(batch, 64, d_model)"]
    Patch3 --> Reshape3["Reshape<br>(batch, 64, d_model)"]
    
    Reshape1 --> POS1["Positional Encoding<br>(batch, 64, d_model)"]
    Reshape2 --> POS2["Positional Encoding<br>(batch, 64, d_model)"]
    Reshape3 --> POS3["Positional Encoding<br>(batch, 64, d_model)"]
    
    POS1 --> MP1["Mean Pooling<br>(batch, d_model)"]
    POS2 --> MP2["Mean Pooling<br>(batch, d_model)"]
    POS3 --> MP3["Mean Pooling<br>(batch, d_model)"]
    
    MP1 --> |Branch 1| Fusion["Feature Fusion<br>(Concat/Attention/Weighted Sum)"]
    MP2 --> |Branch 2| Fusion
    MP3 --> |Branch 3| Fusion
    
    Fusion --> FusedRepr["Fused Representation<br>(batch, d_model)"]
    FusedRepr --> TF["Transformer Layers<br>(all layers)<br>(batch, d_model)"]
    TF --> Norm["Layer Normalization<br>(batch, d_model)"]
    Norm --> Head["Classification Head<br>Linear → GELU → Linear<br>(batch, num_classes)"]
    Head --> Output["Output: Class Probabilities"]
```

## High-Level Architecture

```css
Input: 32x32x3 RGB Image
    ↓
[CNN BACKBONE - DENSENET STYLE]
    ↓
    ├────────────────────┬────────────────────┤
    │                    │                    │                    
    ↓                    ↓                    ↓                    
[BLOCK 1 OUTPUT]    [BLOCK 2 OUTPUT]    [BLOCK 3 OUTPUT]
(32x32, C1)        (16x16, C2)         (8x8, C3)
    │                    │                    │
    ↓                    ↓                    ↓
[4×4 PATCHING]      [2×2 PATCHING]      [1×1 PATCHING]
Non-overlapping     Non-overlapping     Non-overlapping
    │                    │                    │
    ↓                    ↓                    ↓
[PATCH EMBEDDING]    [PATCH EMBEDDING]   [PATCH EMBEDDING]
(batch, 64, d_model) (batch, 64, d_model) (batch, 64, d_model)
    │                    │                    │
    ↓                    ↓                    ↓
[POSITIONAL ENC]     [POSITIONAL ENC]    [POSITIONAL ENC]
    │                    │                    │
    ↓                    ↓                    ↓
[MEAN POOLING]       [MEAN POOLING]      [MEAN POOLING]
(batch, d_model)     (batch, d_model)    (batch, d_model)
    │                    │                    │
    └────────────────────┴────────────────────┘
                         │
                         ↓
                    [FEATURE FUSION]
                  Concat/Attention/Weighted Sum
                         │
                         ↓
                 [TRANSFORMER LAYERS]
                    (all layers)
                         │
                         ↓
                 [CLASSIFICATION HEAD]
                  Linear → GELU → Linear
                         │
                         ↓
                  Output: Class Probabilities
```

## Architecture Details

### CNN Backbone
- **DenseNet-style** architecture with 3 blocks
- Each block contains dense connections for better feature propagation
- Features are extracted at 3 scales: 32x32, 16x16, and 8x8

### Uniform Patching
- **Branch 1**: 4×4 patches from 32×32 feature map → 8×8 grid → 64 tokens
- **Branch 2**: 2×2 patches from 16×16 feature map → 8×8 grid → 64 tokens
- **Branch 3**: 1×1 patches from 8×8 feature map → 8×8 grid → 64 tokens
- Each branch has **exactly the same number of tokens** (64)

### Mean Pooling Features
- Instead of using CLS tokens, each branch representation is obtained by **taking the mean** across all tokens
- This simplifies the model and ensures all spatial information contributes equally to the branch representation

### Fusion Mechanisms
- **Concat**: Concatenates mean-pooled features from all branches, then projects to d_model
- **Attention**: Uses attention to dynamically weight the importance of each scale
- **Weighted Sum**: Learns fixed weights for each scale

### Advantages
1. **Simplified architecture**: Removes CLS token overhead
2. **Better spatial representation**: Mean pooling captures information from all spatial positions
3. **Computational efficiency**: Fewer parameters and reduced sequence length
4. **Multi-scale perception**: Maintains the ability to capture features at different scales
