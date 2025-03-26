# AstroTransformer

A CNN-inspired architecture with relative attention and stage-wise processing designed for 32x32 image classification, adapted from Sample.py.

```css
Input: 32x32x3 RGB Image
    ↓
[STEM STAGE (S0)]
    Conv3x3 → BatchNorm → ReLU → Conv3x3 → BatchNorm → ReLU
    (batch_size, 3, 32, 32) → (batch_size, 32, 32, 32)
    ↓
[STAGE S1]
    StageBlocks with Depthwise & Expansion Convolutions
    (batch_size, 32, 32, 32) → (batch_size, 64, 32, 32)
    ↓
    Add adjusted S0 features (projected via Conv1x1)
    ↓
[STAGE S2]
    StageBlocks with Depthwise & Expansion Convolutions
    (batch_size, 64, 32, 32) → (batch_size, 128, 32, 32)
    ↓
    Add adjusted S1 features (projected via Conv1x1)
    ↓
[STAGE S3]
    StageBlocks with Depthwise & Expansion Convolutions
    (batch_size, 128, 32, 32) → (batch_size, 256, 32, 32)
    ↓
[RELATIVE ATTENTION]
    QKV projection + depthwise convolution for position bias
    Maintains (batch_size, 256, 32, 32)
    ↓
[GLOBAL AVERAGE POOLING]
    (batch_size, 256, 32, 32) → (batch_size, 256, 1, 1)
    ↓
    Flatten
    (batch_size, 256, 1, 1) → (batch_size, 256)
    ↓
[FULLY CONNECTED]
    (batch_size, 256) → (batch_size, num_classes)
    ↓
Output: Class Probabilities
```

## Key Features

### Resolution-Preserving Stage-Based Architecture

The model processes images through sequential stages, each maintaining the full spatial resolution:

1. **StageBlocks**: Specialized blocks containing:
   - Depthwise Convolution (D-Conv): spatial filtering while preserving channels
   - Expansion Convolution (E-Conv): channel expansion via 1x1 convolutions
   - Skip Connections: to maintain gradient flow

2. **Stage Adjustments**: Features from earlier stages are projected and added to later stages:
   - This creates rich multi-channel representations
   - Helps with gradient flow during training

3. **Maintained Spatial Resolution**:
   - Full 32×32 resolution is preserved throughout all stages
   - Better preservation of fine spatial details for small images

### Relative Attention Mechanism

Unlike traditional self-attention, this model uses a relative attention mechanism:

- QKV projection creates query, key and value embeddings
- Depthwise convolution captures local context for relative position bias
- Combined standard and relative attention weights are applied to values
- Operates on full 32x32 spatial feature maps rather than downsampled ones

### Model Configurations

- **Small**: Expansion=2, Layers=[1,1,1]
- **Base**: Expansion=2, Layers=[2,2,2]
- **Large**: Expansion=3, Layers=[2,3,3]

## Usage

To use the AstroTransformer model, set the following parameters in your configuration:

```python
config.model_type = 'trans'
config.transformer_type = 'astro'
# Additional configurations are applied automatically
```
