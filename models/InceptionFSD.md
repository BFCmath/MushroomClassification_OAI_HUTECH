# InceptionFSD

```css
Input (32x32x3)
    ↓
[INITIAL FEATURE EXTRACTION]
    Conv2d → BatchNorm2d → ReLU
    (32x32x32)
    ↓
[INCEPTION BLOCK 1]
    InceptionModule with SE Attention
    - Branch 1: 1x1 Conv (16 channels)
    - Branch 2: 1x1 Conv → 3x3 Conv (32 channels)
    - Branch 3: 1x1 Conv → 3x3 Conv → 3x3 Conv (16 channels)
    - Branch 4: Pool → 1x1 Conv (16 channels)
    - Concat → SE Attention
    (32x32x80)
    ↓
[INCEPTION BLOCK 1b]
    InceptionModule with SE Attention
    - Branch 1: 1x1 Conv (24 channels)
    - Branch 2: 1x1 Conv → 3x3 Conv (48 channels)
    - Branch 3: 1x1 Conv → 3x3 Conv → 3x3 Conv (24 channels)
    - Branch 4: Pool → 1x1 Conv (20 channels)
    - Concat → SE Attention
    (32x32x116)
    ↓
[REDUCTION BLOCK 1]
    FSDDownsample (Multi-path downsampling)
    - Branch 1: 3x3 Conv, stride=2 (36 channels)
    - Branch 2: 5x5 Conv, stride=2 (36 channels)
    - Branch 3: AvgPool → 1x1 Conv (36 channels)
    - Branch 4: MaxPool → 1x1 Conv (36 channels)
    - Concat
    (16x16x144)
    ↓
[INCEPTION BLOCK 2]
    InceptionModule with SE Attention
    - Branch 1: 1x1 Conv (40 channels)
    - Branch 2: 1x1 Conv → 3x3 Conv (64 channels)
    - Branch 3: 1x1 Conv → 3x3 Conv → 3x3 Conv (32 channels)
    - Branch 4: Pool → 1x1 Conv (32 channels)
    - Concat → SE Attention
    (16x16x168)
    ↓
[REDUCTION BLOCK 2]
    FSDDownsample (Multi-path downsampling)
    - Branch 1: 3x3 Conv, stride=2 (48 channels)
    - Branch 2: 5x5 Conv, stride=2 (48 channels)
    - Branch 3: AvgPool → 1x1 Conv (48 channels)
    - Branch 4: MaxPool → 1x1 Conv (48 channels)
    - Concat
    (8x8x192)
    ↓
[INCEPTION BLOCK 3]
    InceptionModule with SE Attention
    - Branch 1: 1x1 Conv (64 channels)
    - Branch 2: 1x1 Conv → 3x3 Conv (96 channels)
    - Branch 3: 1x1 Conv → 3x3 Conv → 3x3 Conv (48 channels)
    - Branch 4: Pool → 1x1 Conv (48 channels)
    - Concat → SE Attention
    (8x8x256)
    ↓
[MULTI-SCALE FEATURE AGGREGATION]
                                   ┌── inception2_out (16x16x168)
                                   │
    inception3_out (8x8x256) ──────┼───────────┐
       │                           │           │
       ↓                           ↓           ↓
  Global Pool                   Local Pool   Mid Pool
    (1x1x256)                   (4x4x168)    (2x2x256)
       │                           │           │
       ↓                           ↓           ↓
   Flatten (256)             Flatten (2688) Flatten (1024)
       │                           │           │
       └───────────────┬───────────┘           │
                       │                       │
                       └──────────┬────────────┘
                                  │
                                  ↓
                        Concatenate (3968)
                                  │
                                  ↓
[FEATURE FUSION]
    Linear → ReLU → Dropout (768)
    Linear → ReLU → Dropout (384)
    Linear → ReLU → Dropout (256)
                                  │
                                  ↓
[CLASSIFICATION]
    Linear (num_classes)
                                  │
                                  ↓
                               Output
```
