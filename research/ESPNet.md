# ESPNet

## Key Findings
The primary contribution of the paper "ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation" is the introduction of ESPNet, a fast and efficient convolutional neural network (CNN) designed for semantic segmentation of high-resolution images under resource constraints. This is achieved through the development of a novel convolutional module called the Efficient Spatial Pyramid (ESP), which leverages convolution factorization to decompose standard convolutions into point-wise convolutions and a spatial pyramid of dilated convolutions. This design significantly reduces computational complexity, memory usage, and power consumption while maintaining a large effective receptive field, making ESPNet suitable for deployment on resource-constrained edge devices like those used in self-driving cars, robots, and augmented reality applications.

## How Does It Work?

The ESP module follows a reduce-split-transform-merge strategy:

- **Reduce**: The input feature map (with $M$ channels) is projected into a lower-dimensional space (e.g., $ \frac{N}{K} $ channels, where $N$ is the number of output channels and $K$ is a width divider hyperparameter) using a 1x1 point-wise convolution. This step reduces computational complexity.
- **Split**: The reduced feature maps are divided across $K$ parallel branches.
- **Transform**: Each branch applies an $n \times n$ dilated convolution with a unique dilation rate (e.g., $2^{k-1}$, where $k = 1, \dots, K$). Dilated convolutions increase the effective receptive field without increasing the number of parameters, as they insert "holes" (zeros) between kernel elements.
- **Merge**: The outputs from the $K$ branches are concatenated to produce an $N$-dimensional output feature map.

The effective receptive field of the ESP module is $\left[(n-1) 2^{K-1} + 1 \right]^2$, which is significantly larger than that of a standard $n \times n$ convolution, enabling it to capture multi-scale contextual information efficiently.

## Going to use
- ESP Module
