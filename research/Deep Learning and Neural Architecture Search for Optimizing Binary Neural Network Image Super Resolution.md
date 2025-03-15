# Deep Learning and Neural Architecture Search for Optimizing Binary Neural Network Image Super Resolution

## Source

[Paper](https://www.mdpi.com/2313-7673/9/6/369#:~:text=2024%20%2D%20Methods%20like,being%20input%20into%20the)

## IDEAS

### **Dilated Group Convolution**
Dilated convolutions are useful in SR because they increase the receptive field without increasing the number of parameters or computational cost. A larger receptive field helps the network understand context over a wider area of the low-resolution image, which is crucial for reconstructing high-resolution details. Group convolution further enhances efficiency.

> Dilated Convolutions:  

```css
1   (gap) 3  
(gap)   7   (gap)  
11   (gap) 13  
```

> Group Convolutions:

```css
Channel 1:    Channel 2:    Channel 3:    Channel 4:
1  2  3  4   5  6  7  8   9  10 11 12   13 14 15 16
17 18 19 20  21 22 23 24  25 26 27 28  29 30 31 32
33 34 35 36  37 38 39 40  41 42 43 44  45 46 47 48
49 50 51 52  53 54 55 56  57 58 59 60  61 62 63 64
```

+ Group 1: Channel 1 and Channel 2
+ Group 2: Channel 3 and Channel 4

### 7x7 Binary Group Convolution

The weights of the 7x7 filter are binarized, meaning they are limited to +1 or -1. This makes the convolution operation more efficient because multiplying by +1 or -1 is much cheaper than working with floating-point values.

### No MaxPool or AvgPool

### Libra Parameter Binarization (Libra-PB)

This technique addresses a core challenge in Binary Neural Networks (BNNs): information loss due to the extreme quantization of weights and activations to binary values (-1, +1). Libra Parameter Binarization (Libra-PB) is incorporated into SRBNAS to maximize the retention of information during the forward propagation process in the binary network.

### Error Decay Estimator (EDE)

EDE is a sophisticated gradient approximation technique designed to make the backpropagation process in BNNs more robust and accurate. Its two-stage progressive approach, starting with a broad gradient flow and then refining the derivative shape, addresses the challenges of training BNNs with discontinuous binarization functions. This is particularly important for complex tasks like super-resolution where accurate gradient information is crucial for learning fine details.

## Going to use
+ Dilated Group Convolution
+ 7x7 Binary Group Convolution
+ No MaxPool or AvgPool
