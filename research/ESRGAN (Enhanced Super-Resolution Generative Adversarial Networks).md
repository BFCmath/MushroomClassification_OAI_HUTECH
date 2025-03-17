# ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)

## Source

[Paper](https://arxiv.org/abs/1809.00219)

## IDEAS
### Limitations of SRGAN's Basic Block and Batch Normalization (BN)

```css
Conv - BN - ReLU - Conv - BN - ReLU  +  Skip Connection
```

### Introduction of the Residual-in-Residual Dense Block (RRDB)

### Relativistic average GAN (RaGAN)

ESRGAN adopts the Relativistic average GAN (RaGAN) discriminator to address this limitation. RaGAN, proposed in a separate paper (cited as Jolicoeur-Martineau, A. in ESRGAN paper), changes the discriminator's objective to focus on relative realness.

### Perceptual Loss

## Going to use
+ Dense Block
