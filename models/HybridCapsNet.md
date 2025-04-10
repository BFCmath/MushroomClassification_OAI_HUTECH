# HybridCapsNet Architecture

```css
Input (32x32x3)
    ↓
[FIRST BLOCK - Maintain Resolution]
    [Routing Layer 1]
    Conv2d transforms (3→D) with stride=1
    ↓
    Cluster Routing with Combined Attention:
    ┌───────────────────────────────┐
    │  Spatial + Channel Attention  │
    │  Agreement-based Routing      │
    └───────────────────────────────┘
    LayerNorm
    (32x32xD) * C capsules
    ↓
    [Routing Layer 2]
    Conv2d transforms (D→D) with stride=1
    ↓
    Cluster Routing with Combined Attention
    LayerNorm
    (32x32xD) * C capsules
    ↓
[SECOND BLOCK - Downsample by 2]
    [Routing Layer 3]
    Conv2d transforms (D→D) with stride=2
    ↓
    Cluster Routing with Combined Attention
    LayerNorm
    (16x16xD) * C capsules
    ↓
    [Routing Layer 4]
    Conv2d transforms (D→D) with stride=1
    ↓
    Cluster Routing with Combined Attention
    LayerNorm
    (16x16xD) * C capsules
    ↓
    [Routing Layer 5]
    Conv2d transforms (D→D) with stride=1
    ↓
    Cluster Routing with Combined Attention
    LayerNorm
    (16x16xD) * C capsules
    ↓
[THIRD BLOCK - Downsample by 2]
    [Routing Layer 6]
    Conv2d transforms (D→D) with stride=2
    ↓
    Cluster Routing with Combined Attention
    LayerNorm
    (8x8xD) * C capsules
    ↓
    [Routing Layer 7]
    Conv2d transforms (D→D) with stride=1
    ↓
    Cluster Routing with Combined Attention
    LayerNorm
    (8x8xD) * C capsules
    ↓
    [Routing Layer 8]
    Conv2d transforms (D→D) with stride=1
    ↓
    Cluster Routing with Combined Attention
    LayerNorm
    (8x8xD) * C capsules
    ↓
[FOURTH BLOCK - Downsample by 2]
    [Routing Layer 9]
    Conv2d transforms (D→D) with stride=2
    ↓
    Cluster Routing with Combined Attention
    LayerNorm
    (4x4xD) * C capsules
    ↓
    [Routing Layer 10]
    Conv2d transforms (D→D) with stride=1
    ↓
    Cluster Routing with Combined Attention
    LayerNorm
    (4x4xD) * C capsules
    ↓
    [Routing Layer 11]
    Conv2d transforms (D→D) with stride=1
    ↓
    Cluster Routing with Combined Attention
    LayerNorm
    (4x4xD) * num_classes capsules
    ↓
[CLASSIFICATION]
    Reshape & Concatenate Capsules
    ↓
    Optional Dropout
    ↓
    Linear (D * 4 * 4 → 1) per class
    ↓
Output (num_classes)
```

## RoutingWithCombinedAttention Detail

Each Routing Layer uses a combined attention mechanism for capsule routing:

```css
                    CAPSULES FROM PREVIOUS LAYER
                              ↓
                ┌─────────────┼─────────────┐
                ↓             ↓             ↓
           Conv2d for      Conv2d for    Conv2d for
          Capsule 1       Capsule 2     Capsule C
                │             │             │
                └─────────────┼─────────────┘
                              ↓
                     CLUSTER ROUTING
                              ↓
                 ┌────────────┴────────────┐
                 │                         │
            Channel-wise              Spatial-wise
             Attention                  Attention
                 │                         │
                 └────────────┬────────────┘
                              ↓
                   AGREEMENT COMPUTATION
                              ↓
                  Compute Standard Deviation
                              ↓
                  Take Negative Log (Agreement)
                              ↓
                   Softmax for Weighting
                              ↓
                  Weighted Sum of Capsules
                              ↓
                  LayerNorm Normalization
                              ↓
                    OUTPUT CAPSULES
```

## Combined Attention Mechanism

```css
Channel Attention:                         Spatial Attention:
┌───────────────────────────┐             ┌───────────────────────────┐
│ Input Capsule Features    │             │ Input Capsule Features    │
└─────────────┬─────────────┘             └─────────────┬─────────────┘
              ↓                                         ↓
┌─────────────┴─────────────┐             ┌─────────────┴─────────────┐
│ Global Avg & Max Pooling  │             │   Channel Pooling         │
│  along capsule dimension  │             │ (Max & Avg across channels)│
└─────────────┬─────────────┘             └─────────────┬─────────────┘
              ↓                                         ↓
┌─────────────┴─────────────┐             ┌─────────────┴─────────────┐
│        MLP Network        │             │     3x3 Conv (2→1)        │
│ (Dimensionality reduction)│             └─────────────┬─────────────┘
└─────────────┬─────────────┘                          ↓
              ↓                           ┌─────────────┴─────────────┐
┌─────────────┴─────────────┐             │        Sigmoid           │
│        Sigmoid           │             └─────────────┬─────────────┘
└─────────────┬─────────────┘                          ↓
              ↓                           ┌─────────────┴─────────────┐
  Apply to capsule features              │ Apply to capsule features  │
  (Rescale + Original features)          │ (Rescale + Original features)
```

The HybridCapsNet combines both the dynamic routing mechanism of capsule networks with the attention mechanisms from spatial and channel attention modules. This hybrid approach allows the network to learn more robust representations by focusing on informative parts of both spatial locations and feature channels.

Default parameter values:
- C (caps_channels): 4
- K (caps_kernels): 10
- D (caps_depth): 32
- input_img_dim: 3 (RGB images)
