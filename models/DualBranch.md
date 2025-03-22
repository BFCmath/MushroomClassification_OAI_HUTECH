# DualBranch

```css
Input (32x32x3)
    ↓
[DUAL PATHWAY ARCHITECTURE]

Branch 1 (Global features):               Branch 2 (Info-preserving):
    Conv2d (3→32, 7×7, pad=3)                Conv2d (3→32, 3×3, pad=1)
    BatchNorm2d → ReLU                       BatchNorm2d → ReLU
    (32x32x32)                               (32x32x32)
    ↓                                        ↓
    ResidualBlock (32→64, stride=2)          SpaceToDepthConv (32→64, block=2)
    (16x16x64)                               (16x16x64)
    ↓                                        ↓
    ResidualBlock (64→128, stride=2)         SpaceToDepthConv (64→128, block=2)
    (8x8x128)                                (8x8x128)
    ↓                                        ↓
    AdaptiveAvgPool2d(4,4)                   AdaptiveAvgPool2d(4,4)
    (4x4x128)                                (4x4x128)
    ↓                                        ↓
    Flatten (2048)                           Flatten (2048)
    ↓                                        ↓
    Linear (2048→256)                        Linear (2048→256)
    ReLU → Dropout                           ReLU → Dropout
    (256)                                    (256)
    ↓                                        ↓
                    [CONCATENATE]
                        (512)
                        ↓
                [FEATURE FUSION]
                    Linear (512→256)
                    ReLU → Dropout
                    (256)
                        ↓
                [CLASSIFICATION]
                    Linear (256→num_classes)
                        ↓
                    Output (num_classes)
```
