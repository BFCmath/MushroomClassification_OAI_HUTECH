# PatchPixT

```css
Input: 32x32x3 RGB Image
    ↓
[PATCH EMBEDDING - CONVOLUTIONAL]
    Use Conv2d with kernel_size=patch_size, stride=patch_size
    For example, with patch_size=4:
    (batch_size, 3, 32, 32) → (batch_size, d_model, 8, 8)
    ↓
    Reshape to sequence format
    (batch_size, d_model, 8, 8) → (batch_size, 64, d_model)
    ↓
[CLS TOKEN ADDITION]
    Prepend learnable classification token
    (batch_size, 64, d_model) → (batch_size, 65, d_model)
    ↓
[POSITIONAL ENCODING]
    Add sinusoidal position embeddings
    Maintains (batch_size, 65, d_model)
    ↓
[OPTIONAL SEQUENCE REDUCTION]
    If enabled, reduce sequence length further
    (batch_size, 65, d_model) → (batch_size, reduced_seq_len, d_model)
    ↓
[TRANSFORMER ENCODER - N LAYERS]
    For each layer:
        ┌─────────────────────────────────┐
        │ Self-Attention (nhead heads)    │
        │   ↓                             │
        │ Residual connection + Dropout   │
        │   ↓                             │
        │ Layer Normalization             │
        │   ↓                             │
        │ Feed-Forward Network            │
        │   (d_model → dim_feedforward → d_model) │
        │   ↓                             │
        │ Residual connection             │
        │   ↓                             │
        │ Layer Normalization             │
        └─────────────────────────────────┘
    Maintains shape throughout
    ↓
[FEATURE EXTRACTION]
    Extract CLS token embedding
    (batch_size, 65, d_model) → (batch_size, d_model)
    ↓
    Layer Normalization
    Maintains (batch_size, d_model)
    ↓
[CLASSIFICATION HEAD]
    Linear → GELU → Dropout
    (batch_size, d_model) → (batch_size, dim_feedforward)
    ↓
    Linear
    (batch_size, dim_feedforward) → (batch_size, num_classes)
    ↓
Output: Class Probabilities
```
