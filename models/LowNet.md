# LowNet

```css
Input (32x32x3)
    ↓
[LOW-RESOLUTION FEATURE EXTRACTOR]
    Conv2d (3→32, kernel=3x3, padding=1) 
    VariableReLU(slope=4)
    (32x32x32)
    ↓
    Conv2d (32→64, kernel=3x3, padding=1)
    VariableReLU(slope=2)
    (32x32x64)
    ↓
    Conv2d (64→128, kernel=3x3, padding=1)
    VariableReLU(slope=1)
    (32x32x128)
    ↓
    Dropout
    (32x32x128)
    ↓
[INFORMATION-PRESERVING DOWNSAMPLING]
    SPDConv (128→128, scale=2, kernel=3x3)
    (16x16x128)
    ↓
    SPDConv (128→128, scale=2, kernel=3x3)
    (8x8x128)
    ↓
    Flatten
    (8192) [128*8*8]
    ↓
[CLASSIFIER]
    Linear (8192→256)
    ReLU
    Dropout
    (256)
    ↓
    Linear (256→128)
    ReLU
    Dropout
    (128)
    ↓
    Linear (128→num_classes)
    (num_classes)
    ↓
Output
```

## Estimate performance

< 0.78
