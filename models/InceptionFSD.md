# InceptionFSD

```css
Input Image (3 channels, 32x32)
         │
         ▼
┌─────────────────────┐
│  Conv (3x3, 32)     │
│  BatchNorm + ReLU   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Inception Module 1  │
│ 4 parallel paths    │
│ + SE attention      │
│ 32 → 80 channels    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Inception Module 1b │
│ 4 parallel paths    │
│ + SE attention      │
│ 80 → 116 channels   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ FSDDownsample 1     │
│ 32x32 → 16x16       │
│ 116 → 144 channels  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Inception Module 2  │
│ 4 parallel paths    │
│ + SE attention      │
│ 144 → 168 channels  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ FSDDownsample 2     │
│ 16x16 → 8x8         │
│ 168 → 192 channels  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Inception Module 3  │
│ 4 parallel paths    │
│ + SE attention      │
│ 192 → 256 channels  │
└──────────┬──────────┘
           │
        ┌──┴──┐
        │     │
        ▼     │
┌─────────────┐ │
│GlobalPool   │ │
│ (1x1)       │ │
└───────┬─────┘ │
        │       │
        │  ┌────▼────┐
        │  │MidPool  │
        │  │ (2x2)   │
        │  └────┬────┘
        │       │
        │       │
        │  ┌────▼────┐
        │  │LocalPool│
        │  │ (4x4)   │
        │  └────┬────┘
        │       │
        ▼       ▼
┌─────────────────────┐
│ Concatenate Features│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 3-Layer Feature     │
│ Fusion (FC layers)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Classifier          │
└─────────────────────┘
```
