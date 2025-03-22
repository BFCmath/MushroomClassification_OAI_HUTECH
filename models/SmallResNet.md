# SmallResNet

```css
Input (32x32x3)
    ↓
[INITIAL CONVOLUTION]
    Conv2d (3→32, kernel=7x7, padding=3) → BatchNorm2d → ReLU
    (32x32x32)
    ↓
[RESIDUAL BLOCK 1] - First downsampling block
    Main Branch:                           Skip Connection:
    Conv2d (32→64, kernel=7x7, stride=2)   Conv2d (32→64, kernel=1, stride=2)
    BatchNorm2d → ReLU                     BatchNorm2d
    Conv2d (64→64, kernel=7x7, stride=1)   (16x16x64)
    BatchNorm2d
    (16x16x64)
    ↓ + ←───────────────────────────────┘
    ReLU
    (16x16x64)
    ↓
[RESIDUAL BLOCK 2]
    Main Branch:                           Skip Connection:
    Conv2d (64→64, kernel=7x7, stride=1)   Identity
    BatchNorm2d → ReLU                     (16x16x64)
    Conv2d (64→64, kernel=7x7, stride=1)
    BatchNorm2d
    (16x16x64)
    ↓ + ←───────────────────────────────┘
    ReLU
    (16x16x64)
    ↓
[RESIDUAL BLOCK 3] - Second downsampling block
    Main Branch:                           Skip Connection:
    Conv2d (64→128, kernel=7x7, stride=2)  Conv2d (64→128, kernel=1, stride=2)
    BatchNorm2d → ReLU                     BatchNorm2d
    Conv2d (128→128, kernel=7x7, stride=1) (8x8x128)
    BatchNorm2d
    (8x8x128)
    ↓ + ←───────────────────────────────┘
    ReLU
    (8x8x128)
    ↓
[RESIDUAL BLOCK 4]
    Main Branch:                           Skip Connection:
    Conv2d (128→128, kernel=7x7, stride=1) Identity
    BatchNorm2d → ReLU                     (8x8x128)
    Conv2d (128→128, kernel=7x7, stride=1)
    BatchNorm2d
    (8x8x128)
    ↓ + ←───────────────────────────────┘
    ReLU
    (8x8x128)
    ↓
[RESIDUAL BLOCK 5] - Third downsampling block
    Main Branch:                           Skip Connection:
    Conv2d (128→256, kernel=7x7, stride=2) Conv2d (128→256, kernel=1, stride=2)
    BatchNorm2d → ReLU                     BatchNorm2d
    Conv2d (256→256, kernel=7x7, stride=1) (4x4x256)
    BatchNorm2d
    (4x4x256)
    ↓ + ←───────────────────────────────┘
    ReLU
    (4x4x256)
    ↓
[RESIDUAL BLOCK 6]
    Main Branch:                           Skip Connection:
    Conv2d (256→256, kernel=7x7, stride=1) Identity
    BatchNorm2d → ReLU                     (4x4x256)
    Conv2d (256→256, kernel=7x7, stride=1)
    BatchNorm2d
    (4x4x256)
    ↓ + ←───────────────────────────────┘
    ReLU
    (4x4x256)
    ↓
[GLOBAL AVERAGE POOLING]
    AdaptiveAvgPool2d((1, 1))
    (1x1x256)
    ↓
    Flatten
    (256)
    ↓
[FULLY CONNECTED LAYERS]
    Dropout
    Linear (256→512)
    ReLU
    Dropout
    Linear (512→num_classes)
    ↓
Output (num_classes)
```
