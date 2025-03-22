# PixT

```css
Input: 32x32x3 RGB Image
    ↓
[RESHAPING]
    Permute dimensions 0,2,3,1 (NCHW → NHWC)
    (batch_size, 3, 32, 32) → (batch_size, 32, 32, 3)
    ↓
    Flatten spatial dimensions
    (batch_size, 32, 32, 3) → (batch_size, 1024, 3)
    ↓
[PIXEL EMBEDDING]
    Linear projection of RGB values
    (batch_size, 1024, 3) → (batch_size, 1024, 128)
    ↓
[CLS TOKEN ADDITION]
    Prepend learnable classification token
    (batch_size, 1024, 128) → (batch_size, 1025, 128)
    ↓
[POSITIONAL ENCODING]
    Add sinusoidal position embeddings
    Maintains (batch_size, 1025, 128)
    ↓
[TRANSFORMER ENCODER - 6 LAYERS]
    For each layer:
        ┌─────────────────────────────────┐
        │ Self-Attention (8 heads)        │
        │   ↓                             │
        │ Residual connection + Dropout   │
        │   ↓                             │
        │ Layer Normalization             │
        │   ↓                             │
        │ Feed-Forward Network            │
        │   (128 → 512 → 128)             │
        │   ↓                             │
        │ Residual connection             │
        │   ↓                             │
        │ Layer Normalization             │
        └─────────────────────────────────┘
    Maintains (batch_size, 1025, 128) throughout
    ↓
[FEATURE EXTRACTION]
    Extract CLS token embedding
    (batch_size, 1025, 128) → (batch_size, 128)
    ↓
    Layer Normalization
    Maintains (batch_size, 128)
    ↓
[CLASSIFICATION HEAD]
    Linear → GELU → Dropout
    (batch_size, 128) → (batch_size, 512)
    ↓
    Linear
    (batch_size, 512) → (batch_size, num_classes)
    ↓
Output: Class Probabilities
```
