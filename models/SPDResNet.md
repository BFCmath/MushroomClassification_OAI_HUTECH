# SPDResNet

```css
Input (32x32x3)
    ↓
[INITIAL PROCESSING]
    Conv2d (3→64, kernel=3x3, stride=1, padding=1) → BatchNorm2d → ReLU
    (32x32x64)
    ↓
[INITIAL SPD DOWNSAMPLING]
    SpaceToDepthConv (64→64, block_size=2)
    (16x16x64)
    ↓
[LAYER 1] - 2x SPDBasicBlocks without downsampling
    Block 1:                                  Block 2:
    Conv2d (64→64) → BN → ReLU               Conv2d (64→64) → BN → ReLU
    Conv2d (64→64) → BN                      Conv2d (64→64) → BN
    + Skip connection                         + Skip connection
    ReLU                                      ReLU
    (16x16x64)                                (16x16x64)
    ↓
[LAYER 2] - 2x SPDBasicBlocks with first one downsampling
    Block 1 (downsampling):                  Block 2:
    SpaceToDepthConv (64→128)                Conv2d (128→128) → BN → ReLU
    Conv2d (128→128) → BN                    Conv2d (128→128) → BN
    + Skip with SPD and 1x1 Conv             + Skip connection
    ReLU                                      ReLU
    (8x8x128)                                (8x8x128)
    ↓
[LAYER 3] - 2x SPDBasicBlocks with first one downsampling
    Block 1 (downsampling):                  Block 2:
    SpaceToDepthConv (128→256)               Conv2d (256→256) → BN → ReLU
    Conv2d (256→256) → BN                    Conv2d (256→256) → BN
    + Skip with SPD and 1x1 Conv             + Skip connection
    ReLU                                      ReLU
    (4x4x256)                                (4x4x256)
    ↓
[LAYER 4] - 2x SPDBasicBlocks with first one downsampling
    Block 1 (downsampling):                  Block 2:
    SpaceToDepthConv (256→512)               Conv2d (512→512) → BN → ReLU
    Conv2d (512→512) → BN                    Conv2d (512→512) → BN
    + Skip with SPD and 1x1 Conv             + Skip connection
    ReLU                                      ReLU
    (2x2x512)                                (2x2x512)
    ↓
[GLOBAL POOLING]
    AdaptiveAvgPool2d((1,1))
    (1x1x512)
    ↓
[CLASSIFICATION]
    Flatten
    Dropout
    Linear (512→num_classes)
    ↓
Output (num_classes)
```
