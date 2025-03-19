# No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects
### Key Finding
The authors identify a fundamental flaw in traditional CNN architectures: the use of strided convolution and pooling layers, which are commonly employed to downsample feature maps. While these operations are effective in scenarios with high-quality images and medium-to-large objects, they lead to a loss of fine-grained information in tougher tasks where images are of low resolution or objects are small. This loss degrades the performance of CNNs, as it hampers the learning of effective feature representations under such conditions. The paper argues that this design limitation becomes particularly evident when the assumption of abundant redundant pixel information no longer holds, as is the case with blurry images or tiny objects.

### Contributions
1. **Introduction of SPD-Conv Building Block**:
   - The authors propose a novel CNN building block called SPD-Conv (Space-to-Depth Convolution) to replace strided convolution and pooling layers entirely. SPD-Conv consists of a space-to-depth (SPD) layer followed by a non-strided (vanilla) convolution layer. The SPD layer downsamples feature maps by rearranging spatial information into the channel dimension, preserving all discriminative details without loss, unlike strided convolution or pooling. The subsequent non-strided convolution reduces the increased channel dimensionality using learnable parameters, maintaining feature quality.

2. **General and Unified Approach**:
   - SPD-Conv is presented as a versatile and unified solution that can be seamlessly integrated into most, if not all, existing CNN architectures. It eliminates both strided convolution and pooling operations in a consistent manner, making it broadly applicable across various computer vision tasks, such as object detection and image classification.

3. **Empirical Validation with Enhanced Models**:
   - The authors demonstrate the effectiveness of SPD-Conv by applying it to two widely used CNN architectures: YOLOv5 (for object detection) and ResNet (for image classification), creating YOLOv5-SPD, ResNet18-SPD, and ResNet50-SPD. These modified models are evaluated on benchmark datasets like COCO-2017, Tiny ImageNet, and CIFAR-10. The results show significant improvements over state-of-the-art models, particularly in average precision (AP) for small objects and top-1 accuracy for low-resolution images. For instance, YOLOv5-SPD-n achieves a 13.15% higher AP for small objects compared to YOLOv5n, and ResNet18-SPD achieves a top-1 accuracy of 64.52% on Tiny ImageNet, surpassing ResNet18’s 61.68%.

4. **Open-Source Availability**:
   - To encourage adoption and further research, the authors have made their code publicly available at <https://github.com/LabSAINT/SPD-Conv>. They also note that SPD-Conv can be easily integrated into popular deep learning frameworks like PyTorch and TensorFlow, enhancing its potential impact on the research and practitioner communities.

### Summary
The paper’s key finding is that strided convolution and pooling layers, while efficient in standard scenarios, are detrimental to CNN performance on low-resolution images and small objects due to information loss. The primary contribution is the SPD-Conv building block, which addresses this issue by preserving fine-grained information during downsampling, offering a generalizable and high-performing alternative. Through rigorous experimentation and open-source dissemination, the authors establish SPD-Conv as a significant improvement over conventional CNN designs, especially for challenging visual tasks.
