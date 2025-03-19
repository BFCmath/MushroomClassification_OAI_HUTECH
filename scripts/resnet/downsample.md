# Model Downsampling Review

After reviewing all the models in the script, here's an analysis of which ones implement downsampling (reducing resolution as the network gets deeper):

## 1. SmallResNet
**Has Downsampling: Yes**
- Uses strided convolutions in residual blocks to progressively reduce spatial dimensions:
  - `self.res_block1 = ResidualBlock(32, 64, stride=2)  # 16x16`
  - `self.res_block3 = ResidualBlock(64, 128, stride=2)  # 8x8`
  - `self.res_block5 = ResidualBlock(128, 256, stride=2)  # 4x4`
- Resolution progression: 32×32 → 16×16 → 8×8 → 4×4
- Finally uses global average pooling to reduce to 1×1

## 2. DenseNet7x7
**Has Downsampling: No**
- Specifically designed to maintain spatial dimensions throughout the network
- From the class description: "DenseNet implementation with 7x7 kernels and **no spatial reduction** for small 32x32 images"
- Transition layers only reduce channels, not spatial dimensions
- Only reduces spatial dimensions at the very end via global average pooling

## 3. DualBranchNetwork
**Has Downsampling: Yes**
- Branch 1 (Global Features):
  - Uses residual blocks with stride=2: 32×32 → 16×16 → 8×8
  - Ends with adaptive pooling to 4×4
- Branch 2 (Local Features):
  - Uses SpaceToDepthConv modules which rearrange spatial information but effectively reduce dimensions
  - Similar progression: 32×32 → 16×16 → 8×8 → 4×4

## 4. InceptionFSD
**Has Downsampling: Yes**
- Uses max pooling in reduction blocks:
  - `self.reduction1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 16x16`
  - `self.reduction2 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 8x8`
- Uses multi-scale pooling at different resolutions to extract features at different scales

## 5. DilatedGroupConvNet
**Has Downsampling: Yes, but different approach**
- Avoids pooling operations but still reduces spatial dimensions via strided convolutions
- Transition blocks with stride=2:
  - `self.transition1 = DilatedGroupConvBlock(64, 128, dilation=1, stride=2)` (32×32 → 16×16)
  - `self.transition2 = DilatedGroupConvBlock(128, 256, dilation=1, stride=2)` (16×16 → 8×8)
- Final global feature extraction uses sequential strided convolutions:
  - 8×8 → 4×4 → 2×2 → 1×1
- Focus on maintaining information while reducing dimensions through dilated convolutions

## Summary

Among all models, only **DenseNet7x7** maintains spatial resolution throughout the main feature extraction pathway. All other models implement some form of spatial dimension reduction as they get deeper, either through strided convolutions, max pooling, or specialized modules like SpaceToDepth.

The DilatedGroupConvNet represents a compromise approach - it does reduce spatial dimensions but uses techniques (dilated convolutions and residual connections) specifically designed to preserve spatial information despite the reduction in resolution.