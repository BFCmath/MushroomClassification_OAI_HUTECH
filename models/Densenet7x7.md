# Densenet7x7

```css
Input (32x32x3)
    ↓
[INITIAL FEATURE EXTRACTION]
    Conv2d (3→64, kernel=7x7, stride=1, padding=3)
    BatchNorm2d → ReLU
    (32x32x64)
    ↓
[DENSE BLOCK 1] (6 layers)
    Each layer:
        BatchNorm2d → ReLU → Conv2d (7x7) → Concatenate
    With dense connections:
        Layer 1: 64 → 32 features → 96 features
        Layer 2: 96 → 32 features → 128 features
        Layer 3: 128 → 32 features → 160 features
        Layer 4: 160 → 32 features → 192 features
        Layer 5: 192 → 32 features → 224 features
        Layer 6: 224 → 32 features → 256 features
    (32x32x256)
    ↓
[TRANSITION 1]
    BatchNorm2d → ReLU → Conv2d (1x1)
    Reduces channels by half
    (32x32x128)
    ↓
[DENSE BLOCK 2] (12 layers)
    Similar structure with dense connections
    Input: 128 features
    Output: 128 + 12*32 = 512 features
    (32x32x512)
    ↓
[TRANSITION 2]
    BatchNorm2d → ReLU → Conv2d (1x1)
    Reduces channels by half
    (32x32x256)
    ↓
[DENSE BLOCK 3] (24 layers)
    Similar structure with dense connections
    Input: 256 features
    Output: 256 + 24*32 = 1024 features
    (32x32x1024)
    ↓
[TRANSITION 3]
    BatchNorm2d → ReLU → Conv2d (1x1)
    Reduces channels by half
    (32x32x512)
    ↓
[DENSE BLOCK 4] (16 layers)
    Similar structure with dense connections
    Input: 512 features
    Output: 512 + 16*32 = 1024 features
    (32x32x1024)
    ↓
[FINAL PROCESSING]
    BatchNorm2d → ReLU
    (32x32x1024)
    ↓
[GLOBAL POOLING]
    AdaptiveAvgPool2d(1,1)
    (1x1x1024)
    ↓
    Flatten
    (1024)
    ↓
    Dropout
    (1024)
    ↓
[CLASSIFICATION]
    Linear (1024→num_classes)
    (num_classes)
    ↓
Output
```
