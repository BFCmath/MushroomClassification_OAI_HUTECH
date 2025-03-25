# MultiScalePixT

## Normal

```css
Input Image (32x32x3)
    │
    ▼
┌─────────────────────────────────────┐
│ Extract Non-overlapping Patches     │
│  - Divide into 2×2 pixel patches    │
│  - Reshape and permute dimensions   │
└─────────────────────────────────────┘
    │
    │ [B, 256, 12] (256 patches of 12 values each)
    ▼
┌─────────────────────────────────────┐
│ Patch Embedding                     │
│  - Linear projection to d_model     │
└─────────────────────────────────────┘
    │
    │ [B, 256, 128] tokens
    ▼
┌─────────────────────────────────────┐
│ Add CLS Token + Positional Encoding │
└─────────────────────────────────────┘
    │
    │ [B, 257, 128] tokens
    ▼
┌─────────────────────────────────────┐
│ Transformer Encoder Blocks          │
│  - Multi-head Self Attention        │
│  - Feed Forward Networks            │
└─────────────────────────────────────┘
    │
    │ [B, 257, 128] tokens
    ▼
┌─────────────────────────────────────┐
│ Extract CLS Token + Classification  │
└─────────────────────────────────────┘
    │
    ▼
Output Classes
```

## Nested

```css
Input Image (32x32x3)
    │
    ▼
┌─────────────────────────────────────┐
│ Extract Hierarchical Patches        │
│  - Divide into 4×4 primary patches  │
│  - Further divide into 2×2 sub-patches│
└─────────────────────────────────────┘
    │
    │ [B, 64, 4, 12] (64 primary patches, 
    │                 each with 4 sub-patches)
    ▼
┌─────────────────────────────────────┐
│ Sub-patch Embedding                 │
└─────────────────────────────────────┘
    │
    │ [B, 64, 4, d_model/2]
    ▼
┌─────────────────────────────────────┐
│ Local Transformer (per patch)       │
│  - Process each primary patch's     │
│    sub-patches independently        │
└─────────────────────────────────────┘
    │
    │ [B, 64, d_model] (after projection)
    ▼
┌─────────────────────────────────────┐
│ Add CLS Token + Positional Encoding │
└─────────────────────────────────────┘
    │
    │ [B, 65, d_model]
    ▼
┌─────────────────────────────────────┐
│ Global Transformer                  │
│  - Process all patch representations│
└─────────────────────────────────────┘
    │
    │ [B, 65, d_model]
    ▼
┌─────────────────────────────────────┐
│ Extract CLS Token + Classification  │
└─────────────────────────────────────┘
    │
    ▼
Output Classes
```
