# SPDDualBranch

```css
Input (32x32x3)
    ↓
[DUAL PATHWAY ARCHITECTURE]

Branch 1 (Global features):               Branch 2 (Local details):
    ConvBlock (kernel=7x7)                   ConvBlock (kernel=3x3) 
    3→32, pad=3                              3→32, pad=1
    (32x32x32)                               (32x32x32)
    ↓                                        ↓
    SPDConv (scale=2)                        SPDConv (scale=2)
    32→64                                    32→64
    (16x16x64)                               (16x16x64)
    ↓                                        ↓
    ConvBlock (kernel=5x5)                   ConvBlock (kernel=3x3)
    64→64, pad=2                             64→64, pad=1
    (16x16x64)                               (16x16x64)
    ↓                                        ↓
    SPDConv (scale=2)                        SPDConv (scale=2)
    64→128                                   64→128
    (8x8x128)                                (8x8x128)
    ↓                                        ↓
    ConvBlock (kernel=5x5)                   ConvBlock (kernel=3x3)
    128→128, pad=2                           128→128, pad=1
    (8x8x128)                                (8x8x128)
    ↓                                        ↓
    AdaptiveAvgPool2d(4,4)                   AdaptiveAvgPool2d(4,4)
    (4x4x128)                                (4x4x128)
    ↓                                        ↓
    Flatten                                  Flatten
    (2048)                                   (2048)
    ↓                                        ↓
    Linear 2048→256                          Linear 2048→256
    ReLU → Dropout                           ReLU → Dropout
    (256)                                    (256)
    ↓                                        ↓
                    [CONCATENATE]
                    (512)
                        ↓
                [FEATURE FUSION]
                    Linear 512→256
                    ReLU → Dropout
                    (256)
                        ↓
                [CLASSIFICATION]
                    Linear 256→num_classes
                        ↓
                    Output (num_classes)
```
