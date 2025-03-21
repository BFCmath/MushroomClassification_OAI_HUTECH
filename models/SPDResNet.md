# SPDResNet

```css
Input Image (3 channels, 32x32)
         │
         ▼
┌─────────────────────┐
│ Initial Conv (3x3)  │
│ BatchNorm + ReLU    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ SPD Initial         │
│ Space-to-Depth      │
│ 32x32 → 16x16       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Layer 1             │
│ SPDBasic/Bottleneck │
│ 2 blocks            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Layer 2             │
│ SPDBasic/Bottleneck │
│ 2 blocks, downsample│
│ 16x16 → 8x8         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Layer 3             │
│ SPDBasic/Bottleneck │
│ 2 blocks, downsample│
│ 8x8 → 4x4           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Layer 4             │
│ SPDBasic/Bottleneck │
│ 2 blocks, downsample│
│ 4x4 → 2x2           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ AdaptiveAvgPool     │
│ (1x1)               │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Dropout             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Classifier          │
└─────────────────────┘
```

## Estimate performance

0.84

Ensemble: 0.9
