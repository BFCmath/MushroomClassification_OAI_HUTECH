# DilatedGroup

```css
Input (32x32x3)
    ↓
[STEM]
    Conv2d (7x7) → BatchNorm2d → ReLU
    (32x32x32)
    ↓
[STAGE 1] - Same resolution with increasing receptive fields
    DilatedGroupConvBlock (dilation=1, groups=4)
    ↓ Input projection → Grouped dilated conv → Output projection
    (32x32x64)
    ↓
    DilatedGroupConvBlock (dilation=2, groups=4)
    ↓ Input projection → Grouped dilated conv → Output projection
    (32x32x64)
    ↓
    DilatedGroupConvBlock (dilation=4, groups=4)
    ↓ Input projection → Grouped dilated conv → Output projection
    (32x32x64)
    ↓
[TRANSITION 1] - SPDConv downsampling
    DilatedGroupConvBlock (stride=2, dilation=1)
    ↓ Input proj → Grouped conv → SPDConv downsampling → Output proj
    (16x16x128)
    ↓
[STAGE 2] - Medium resolution with increasing receptive fields
    DilatedGroupConvBlock (dilation=1, groups=4)
    ↓ Input projection → Grouped dilated conv → Output projection
    (16x16x128)
    ↓
    DilatedGroupConvBlock (dilation=2, groups=4)
    ↓ Input projection → Grouped dilated conv → Output projection
    (16x16x128)
    ↓
    DilatedGroupConvBlock (dilation=4, groups=4)
    ↓ Input projection → Grouped dilated conv → Output projection
    (16x16x128)
    ↓
[TRANSITION 2] - SPDConv downsampling
    DilatedGroupConvBlock (stride=2, dilation=1)
    ↓ Input proj → Grouped conv → SPDConv downsampling → Output proj
    (8x8x256)
    ↓
[STAGE 3] - Deep features at low resolution
    DilatedGroupConvBlock (dilation=1, groups=4)
    ↓ Input projection → Grouped dilated conv → Output projection
    (8x8x256)
    ↓
    DilatedGroupConvBlock (dilation=2, groups=4)
    ↓ Input projection → Grouped dilated conv → Output projection
    (8x8x256)
    ↓
[GLOBAL FEATURE EXTRACTION] - Cascaded SPDConv
    SPDConv (scale=2) → BatchNorm → ReLU
    (4x4x512)
    ↓
    SPDConv (scale=2) → BatchNorm → ReLU
    (2x2x512)
    ↓
    SPDConv (scale=2) → BatchNorm → ReLU
    (1x1x512)
    ↓
[CLASSIFIER]
    Flatten
    (512)
    ↓
    Dropout (rate=0.3)
    ↓
    Linear
    (num_classes)
    ↓
Output
```
