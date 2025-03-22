# MiniXception

```css
Input (32x32x3)
    ↓
[STEM]
    Conv2d (3→32, 3x3, pad=1) → BatchNorm2d → ReLU
    Conv2d (32→64, 3x3, pad=1) → BatchNorm2d → ReLU
    (32x32x64)
    ↓
[DOWN1: DOWNSAMPLING BLOCK 1]
    Main Branch:                             Skip Connection:
    ReLU                                     Conv2d (64→128, 1x1)
    SeparableConv2d (64→64, 3x3)             BatchNorm2d
    BatchNorm2d                              SPDConv (128→128, scale=2)
    ReLU                                     (16x16x128)
    SeparableConv2d (64→128, 3x3)
    BatchNorm2d
    ReLU
    SPDConv (128→128, scale=2)
    (16x16x128)
    ↓ + ←───────────────────────────────────┘
    (16x16x128)
    ↓
[BLOCK1: XCEPTION BLOCK]
    Main Branch:                             Skip Connection:
    ReLU                                     Identity
    SeparableConv2d (128→128, 3x3)           (16x16x128)
    BatchNorm2d
    ReLU
    SeparableConv2d (128→128, 3x3)
    BatchNorm2d
    ReLU
    SeparableConv2d (128→128, 3x3)
    BatchNorm2d
    (16x16x128)
    ↓ + ←───────────────────────────────────┘
    (16x16x128)
    ↓
[DOWN2: DOWNSAMPLING BLOCK 2]
    Main Branch:                             Skip Connection:
    ReLU                                     Conv2d (128→256, 1x1)
    SeparableConv2d (128→128, 3x3)           BatchNorm2d
    BatchNorm2d                              SPDConv (256→256, scale=2)
    ReLU                                     (8x8x256)
    SeparableConv2d (128→256, 3x3)
    BatchNorm2d
    ReLU
    SPDConv (256→256, scale=2)
    (8x8x256)
    ↓ + ←───────────────────────────────────┘
    (8x8x256)
    ↓
[MIDDLE FLOW: 3x XCEPTION BLOCKS]
    For each XceptionBlock:
    ReLU → SeparableConv2d → BatchNorm2d
    ReLU → SeparableConv2d → BatchNorm2d
    ReLU → SeparableConv2d → BatchNorm2d
    + Residual Connection
    (8x8x256)
    ↓
[DOWN3: DOWNSAMPLING BLOCK 3]
    Main Branch:                             Skip Connection:
    ReLU                                     Conv2d (256→512, 1x1)
    SeparableConv2d (256→256, 3x3)           BatchNorm2d
    BatchNorm2d                              SPDConv (512→512, scale=2)
    ReLU                                     (4x4x512)
    SeparableConv2d (256→512, 3x3)
    BatchNorm2d
    ReLU
    SPDConv (512→512, scale=2)
    (4x4x512)
    ↓ + ←───────────────────────────────────┘
    (4x4x512)
    ↓
[EXIT FLOW]
    SeparableConv2d (512→512, 3x3) → BatchNorm2d → ReLU
    SeparableConv2d (512→512, 3x3) → BatchNorm2d → ReLU
    (4x4x512)
    ↓
[GLOBAL POOLING]
    AdaptiveAvgPool2d((1,1))
    (1x1x512)
    ↓
[CLASSIFIER]
    Flatten (512)
    Dropout
    Linear (512→num_classes)
    ↓
Output (num_classes)
```