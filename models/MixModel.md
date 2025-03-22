# MixModel
## MixModel1
```css
Input (32x32x3)
    ↓
[INITIAL SPACE-TO-DEPTH]
    SPDConv (scale=2, kernel=3x3)
    (16x16x12)
    ↓
[CHANNEL ADJUSTMENT]
    Conv2d (1x1) → BatchNorm2d → ReLU
    (16x16x64)
    ↓
[STAGE A - 16x16 RESOLUTION]
    Block A1: Residual Inception Block
    ┌─────────────────────────────────────────────┐
    │ Path 1: 1x1 Conv (64→16)                    │
    │ Path 2: 1x1 Conv → 3x3 Conv (dilation=1)    │
    │ Path 3: 1x1 Conv → 3x3 Conv (dilation=2)    │
    │ Path 4: 1x1 Conv → 3x3 Conv (dilation=3)    │
    │ Concat → 1x1 Conv → SE Attention → Residual │
    └─────────────────────────────────────────────┘
    (16x16x64)
    ↓
    Block A2: Residual Inception Block (same structure)
    (16x16x64)
    ↓
[DOWNSAMPLING 1]
    SPDConv (scale=2)
    (8x8x256)
    ↓
[CHANNEL ADJUSTMENT]
    Conv2d (1x1) → BatchNorm2d → ReLU
    (8x8x128)
    ↓
[STAGE B - 8x8 RESOLUTION]
    Block B1: Residual Inception Block
    ┌─────────────────────────────────────────────┐
    │ Path 1: 1x1 Conv (128→32)                   │
    │ Path 2: 1x1 Conv → 3x3 Conv (dilation=1)    │
    │ Path 3: 1x1 Conv → 3x3 Conv (dilation=2)    │
    │ Path 4: 1x1 Conv → 3x3 Conv (dilation=3)    │
    │ Concat → 1x1 Conv → SE Attention → Residual │
    └─────────────────────────────────────────────┘
    (8x8x128)
    ↓
    Block B2: Residual Inception Block (same structure)
    (8x8x128)
    ↓
[DOWNSAMPLING 2]
    SPDConv (scale=2)
    (4x4x512)
    ↓
[CHANNEL ADJUSTMENT]
    Conv2d (1x1) → BatchNorm2d → ReLU
    (4x4x256)
    ↓
[STAGE C - 4x4 RESOLUTION]
    Block C1: Residual Inception Block
    ┌─────────────────────────────────────────────┐
    │ Path 1: 1x1 Conv (256→64)                   │
    │ Path 2: 1x1 Conv → 3x3 Conv (dilation=1)    │
    │ Path 3: 1x1 Conv → 3x3 Conv (dilation=2)    │
    │ Path 4: 1x1 Conv → 3x3 Conv (dilation=3)    │
    │ Concat → 1x1 Conv → SE Attention → Residual │
    └─────────────────────────────────────────────┘
    (4x4x256)
    ↓
[GLOBAL POOLING]
    AdaptiveAvgPool2d((1,1))
    (1x1x256)
    ↓
    Flatten
    (256)
    ↓
[CLASSIFICATION]
    Dropout
    Linear (256→num_classes)
    ↓
Output (num_classes)
```

## MixModel2
```css
Input (32x32x3)
    ↓
[INITIAL DOWNSAMPLING]
    SPDConv (scale=2, kernel=1x1)
    (16x16x12)
    ↓
[STAGE A - 16x16 RESOLUTION]
    Mix2InceptionBlockA:
        Path 1: 1x1 Conv (16 channels)
        Path 2: 1x1 Conv → 3x3 Conv (dilation=1) (16 channels)
        Path 3: 1x1 Conv → 3x3 Conv (dilation=2) (16 channels)
        Path 4: 1x1 Conv → 3x3 Conv (dilation=3) (16 channels)
        Concatenate
    (16x16x64)
    ↓
    SE Attention A1
    (16x16x64)
    ↓
    Store for Skip Connection ───────┐
    ↓                                │
    Mix2InceptionBlockA (same as above)
    (16x16x64)                       │
    ↓                                │
    Add Skip Connection ←────────────┘
    ↓
    SE Attention A2
    (16x16x64)
    ↓
[DOWNSAMPLING TO STAGE B]
    SPDConv (scale=2, kernel=1x1)
    (8x8x256)
    ↓
[STAGE B - 8x8 RESOLUTION]
    Mix2InceptionBlockA (with 32 filters per path)
    (8x8x128)
    ↓
    SE Attention B1
    (8x8x128)
    ↓
    Store for Skip Connection ───────┐
    ↓                                │
    Mix2InceptionBlockA (with 32 filters per path)
    (8x8x128)                        │
    ↓                                │
    Add Skip Connection ←────────────┘
    ↓
    SE Attention B2
    (8x8x128)
    ↓
[DOWNSAMPLING TO STAGE C]
    SPDConv (scale=2, kernel=1x1)
    (4x4x512)
    ↓
[STAGE C - 4x4 RESOLUTION]
    Mix2InceptionBlockC:
        Path 1: 1x1 Conv (64 channels)
        Path 2: 3x3 Conv (64 channels)
        Concatenate
    (4x4x128)
    ↓
    SE Attention C
    (4x4x128)
    ↓
[GLOBAL POOLING]
    AdaptiveAvgPool2d
    (1x1x128)
    ↓
    Flatten
    (128)
    ↓
    Dropout
    ↓
    Fully Connected
    (num_classes)
    ↓
Output
```

## MixModel3
```css
Input (32x32x3)
    ↓
[STEM]
    Conv2d (3→32, kernel=3x3, pad=1) → BatchNorm2d → ReLU
    Conv2d (32→64, kernel=3x3, pad=1) → BatchNorm2d → ReLU
    (32x32x64)
    ↓
[STAGE 2] - Multi-scale feature extraction with dilated convolutions
    Mix3DilatedMultiScaleBlock (64→96):
        Path 1: 1x1 Conv                              (32x32x24)
        Path 2: 1x1 Conv → 3x3 Conv (dilation=1)      (32x32x24)
        Path 3: 1x1 Conv → 3x3 Conv (dilation=2)      (32x32x24)
        Path 4: 1x1 Conv → 3x3 Conv (dilation=4)      (32x32x24)
        Concatenate → Project → Add Skip → SE+Spatial Attention
    (32x32x96)
    ↓
    Mix3DilatedMultiScaleBlock (96→128):
        Path 1: 1x1 Conv                              (32x32x32)
        Path 2: 1x1 Conv → 3x3 Conv (dilation=1)      (32x32x32)
        Path 3: 1x1 Conv → 3x3 Conv (dilation=3)      (32x32x32)
        Path 4: 1x1 Conv → 3x3 Conv (dilation=5)      (32x32x32)
        Concatenate → Project → Add Skip → SE+Spatial Attention
    (32x32x128)
    ↓
[STAGE 3] - Deeper feature extraction with increased receptive fields
    Mix3DilatedMultiScaleBlock (128→192):
        Path 1: 1x1 Conv                              (32x32x48)
        Path 2: 1x1 Conv → 3x3 Conv (dilation=1)      (32x32x48)
        Path 3: 1x1 Conv → 3x3 Conv (dilation=3)      (32x32x48)
        Path 4: 1x1 Conv → 3x3 Conv (dilation=6)      (32x32x48)
        Concatenate → Project → Add Skip → SE+Spatial Attention
    (32x32x192)
    ↓
    Mix3DilatedMultiScaleBlock (192→256):
        Path 1: 1x1 Conv                              (32x32x64)
        Path 2: 1x1 Conv → 3x3 Conv (dilation=1)      (32x32x64)
        Path 3: 1x1 Conv → 3x3 Conv (dilation=4)      (32x32x64)
        Path 4: 1x1 Conv → 3x3 Conv (dilation=8)      (32x32x64)
        Concatenate → Project → Add Skip → SE+Spatial Attention
    (32x32x256)
    ↓
[STAGE 4] - Final feature refinement with enhanced attention
    Mix3DilatedMultiScaleBlock (256→384):
        Path 1: 1x1 Conv                              (32x32x96)
        Path 2: 1x1 Conv → 3x3 Conv (dilation=1)      (32x32x96)
        Path 3: 1x1 Conv → 3x3 Conv (dilation=2)      (32x32x96)
        Path 4: 1x1 Conv → 3x3 Conv (dilation=5)      (32x32x96)
        Path 5: 1x1 Conv → 3x3 Conv (dilation=9)      (32x32x96)
        Concatenate → Project → Add Skip → SE+Spatial Attention
    (32x32x384)
    ↓
[MULTI-SCALE FEATURE AGGREGATION]
    ┌─────────────────┼─────────────────┐
    ↓                 ↓                 ↓
Global Pooling    Mid Pooling       Local Pooling
(1x1x384)         (2x2x384)         (4x4x384)
    ↓                 ↓                 ↓
Flatten (384)     Flatten (1536)    Flatten (6144)
    ↓                 ↓                 ↓
    └─────────────────┼─────────────────┘
                      ↓
               Concatenate (8064)
                      ↓
[FEATURE FUSION]
    Linear (8064→512) → ReLU → Dropout
    Linear (512→256) → ReLU → Dropout
    (256)
    ↓
[CLASSIFICATION]
    Linear (256→num_classes)
    ↓
Output (num_classes)
```

## MixModel4
```css
Input (32x32x3)
    ↓
[STEM]
    Conv2d (3→32, kernel=3x3) → BatchNorm2d → ReLU
    Conv2d (32→64, kernel=3x3) → BatchNorm2d → ReLU
    (32x32x64)
    ↓
[STAGE 1] - Multi-branch spatial feature extraction
    ┌─────────────────────────────────────────────────┐
    │ Mix4MultiBranchModule (64→96)                   │
    │   Branch 1: 1x1 Conv (24 channels)              │
    │   Branch 2: Medium features (24 channels)       │
    │     - 3x3 Conv                                  │
    │     - 5x5 Conv                                  │
    │     - 7x7 Conv with dilation=2                  │
    │   Branch 3: Global features (48 channels)       │
    │     - 5x5 Conv                                  │
    │     - 7x7 Conv                                  │
    │     - 9x9 Conv with dilation=2                  │
    │   + Residual connection                         │
    └─────────────────────────────────────────────────┘
    (32x32x96)
    ↓
    Mix4SEBlock - Channel-wise attention
    (32x32x96)
    ↓
    ┌─────────────────────────────────────────────────┐
    │ Mix4MultiBranchModule (96→128)                  │
    │   Similar structure as above with 7 paths       │
    │   + Residual connection                         │
    └─────────────────────────────────────────────────┘
    (32x32x128)
    ↓
    Mix4SEBlock - Channel-wise attention
    (32x32x128)
    ↓
[DOWNSAMPLING]
    SPDConv (scale=2, kernel=3x3)
    Space-to-Depth operation with convolution
    (16x16x192)
    ↓
[STAGE 2] - Multi-branch spatial feature extraction at reduced resolution
    ┌─────────────────────────────────────────────────┐
    │ Mix4MultiBranchModule (192→256)                 │
    │   Same 7-pathway structure as Stage 1           │
    │   + Residual connection                         │
    └─────────────────────────────────────────────────┘
    (16x16x256)
    ↓
    Mix4SEBlock - Channel-wise attention
    (16x16x256)
    ↓
    ┌─────────────────────────────────────────────────┐
    │ Mix4MultiBranchModule (256→384)                 │
    │   Same 7-pathway structure as Stage 1           │
    │   + Residual connection                         │
    └─────────────────────────────────────────────────┘
    (16x16x384)
    ↓
    Mix4SEBlock - Channel-wise attention
    (16x16x384)
    ↓
[MULTI-SCALE FEATURE AGGREGATION]
                    ┌─────────────┼─────────────┐
                    ↓             ↓             ↓
          Global Pool         Mid Pool      Local Pool
            (1x1x384)        (2x2x384)      (4x4x384)
                    ↓             ↓             ↓
              Flatten (384)   Flatten (1536) Flatten (6144)
                    ↓             ↓             ↓
                    └─────────────┼─────────────┘
                                  ↓
                           Concatenate
                              (8064)
                                  ↓
[FEATURE FUSION]
    Linear (8064→512) → ReLU → Dropout
    Linear (512→256) → ReLU → Dropout
    (256)
    ↓
[CLASSIFICATION]
    Linear (256→num_classes)
    ↓
Output (num_classes)
```
