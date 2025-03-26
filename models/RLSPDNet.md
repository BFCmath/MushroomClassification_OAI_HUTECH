# RLSPDNet Architecture

```css
Input (32x32x3)
    ↓
[MK BLOCK SPD 1]
    Layer 1 - Parallel Convolutions:
    ┌───────────────┬───────────────┬───────────────┐
    ↓               ↓               ↓               
    3x3 Conv (3→48) 5x5 Conv (3→24) 7x7 Conv (3→12)
    BatchNorm + ReLU BatchNorm + ReLU BatchNorm + ReLU
    (32x32x48)      (32x32x24)      (32x32x12)
    └───┬───────────┼───────────────┬───────────────┘
        │           │               │
        │      ┌────┴────┐     ┌────┴────┐
        │      ↓         │     ↓         │
    Layer 2 - Feature Combinations:
        │   Concat(48+24)    Concat(24+12)
        │      (32x32x72)      (32x32x36)
        │      ↓               ↓
        │   5x5 Conv (72→36)  7x7 Conv (36→18)
        │   BatchNorm + ReLU   BatchNorm + ReLU
        │   (32x32x36)         (32x32x18)
        │   ↓
        │   3x3 Conv (36→72)
        │   BatchNorm + ReLU
        │   (32x32x72)
    ┌───┴───────────┬───────────────┬───────────────┐
    ↓               ↓               ↓               ↓
    (32x32x48)      (32x32x36)      (32x32x72)      (32x32x18)
    └───────────────┴───────┬───────┴───────────────┘
                            ↓
    Layer 3 - Final Aggregation:
                    Concat(48+36+72+18)
                        (32x32x174)
                            ↓
                    1x1 Conv (174→24)
                    BatchNorm + ReLU
                        (32x32x24)
                            ↓
                Space-to-Depth Conv (24→24, block_size=2)
                        (16x16x24)
    ↓
[MK BLOCK SPD 2] - Same structure but with (24→24) input channels
    Multi-path feature extraction → Feature combinations → Aggregation → Space-to-Depth
    (8x8x24)
    ↓
[MK BLOCK SPD 3] - Same structure but with (24→24) input channels
    Multi-path feature extraction → Feature combinations → Aggregation → Space-to-Depth
    (4x4x24)
    ↓
[FINAL PROCESSING]
    BatchNorm2d
    (4x4x24)
    ↓
    Flatten
    (384)
    ↓
[CLASSIFICATION LAYERS]
    Dropout (0.2)
    ↓
    Linear (384→256) → ReLU
    ↓
    Dropout (0.2)
    ↓
    Linear (256→128) → ReLU
    ↓
    Dropout (0.2)
    ↓
    Linear (128→128) → ReLU
    ↓
    Dropout (0.2)
    ↓
    Linear (128→64) → ReLU
    ↓
    Linear (64→4)
    ↓
Output (4 classes)
```

## MKBlockSPD Detail

Each Multi-Kernel Block with Space-to-Depth (MKBlockSPD) follows this structure:

```css
                         INPUT
                           ↓
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
      3x3 Conv         5x5 Conv         7x7 Conv
      BN + ReLU        BN + ReLU        BN + ReLU
          │                │                │
          │        ┌───────┴───────┐       │
          │        │               │       │
          │   Concat(3x3+5x5)  Concat(5x5+7x7)
          │        │               │
          │    5x5 Conv         7x7 Conv
          │    BN + ReLU        BN + ReLU
          │        │               │
          │    3x3 Conv            │
          │    BN + ReLU           │
          │        │               │
          ↓        ↓               ↓
      path1_1    path3_1        path2_2
          │        │               │
          └────────┼───────────────┘
                   ↓
          Concat(path1_1 + path2_1 + path3_1 + path2_2)
                   ↓
              1x1 Conv (bottleneck)
                BN + ReLU
                   ↓
         Space-to-Depth Conv (block_size=2)
                   ↓
                 OUTPUT
```

## Space-to-Depth vs MaxPooling

```css
MaxPooling (Original RLNet):              Space-to-Depth (RLSPDNet):
┌───┬───┬───┬───┐                        ┌───┬───┬───┬───┐
│ A │ B │ E │ F │                        │ A │ B │ E │ F │
├───┼───┼───┼───┤                        ├───┼───┼───┼───┤
│ C │ D │ G │ H │                        │ C │ D │ G │ H │
├───┼───┼───┼───┤                        ├───┼───┼───┼───┤
│ I │ J │ M │ N │                        │ I │ J │ M │ N │
├───┼───┼───┼───┤          →             ├───┼───┼───┼───┤
│ K │ L │ O │ P │                        │ K │ L │ O │ P │
└───┴───┴───┴───┘                        └───┴───┴───┴───┘
       ↓                                        ↓
┌───┬───┐                               ┌───────────┐
│max│max│                               │ A B C D   │
│   │   │                               │ E F G H   │
├───┼───┤                               │ I J K L   │
│max│max│                               │ M N O P   │
└───┴───┘                               └───────────┘
Information is discarded              All information is preserved
                                     and rearranged in channels
```

This modification helps preserve more spatial information during downsampling, which is beneficial for low-resolution inputs (32x32) and detailed feature extraction.
