# Astroformer
## Methodology
### 1. Model Architecture: Astroformer as a CoAtNet Variant

* **Hybrid Approach:** Astroformer is presented as a **hybrid transformer-convolutional architecture**. This means it strategically combines convolutional layers with transformer blocks to leverage the strengths of both approaches.
  * **CNNs for Local Features and Efficiency:** Convolutional layers are known for their efficiency in capturing local patterns and spatial inductive biases, which are beneficial for image tasks, especially with limited data.
  * **Transformers for Global Context:** Transformer blocks, particularly with self-attention, excel at capturing long-range dependencies and global context within the image.

* **Inspiration from CoAtNet:** Astroformer draws significant inspiration from **CoAtNet (Dai et al., 2021)**.  It adopts a similar overall structure but introduces key modifications.

* **Modified Stack Design (C-C-C-T):**  Astroformer employs a **different stack design** compared to standard CoAtNet architectures.
  * **CoAtNet's Original Design (C-C-T-T or C-T-T-T):**  CoAtNet typically uses stacks like Convolution-Convolution-Transformer-Transformer (C-C-T-T) or Convolution-Transformer-Transformer-Transformer (C-T-T-T).
  * **Astroformer's Design (C-C-C-T):** Astroformer uses a **Convolution-Convolution-Convolution-Transformer (C-C-C-T) stack**.  The paper emphasizes that this C-C-C-T design performs **better** than C-C-T-T and C-T-T-T in low-data regimes, offering higher generalization and more stable training.
  * **Stages (S0-S4):**  Similar to CoAtNet, Astroformer uses a multi-stage network structure (S0, S1, S2, S3, S4):
    * **S0 (Stem Stage):**  Simple 2-layer convolutional stem for initial feature extraction.
    * **S1, S2, S3:**  Employ **Inverted Residual blocks** (from MobileNetV2) with squeeze-excitation. These are efficient convolutional blocks.
    * **S4:** Employs a **Transformer block** with the proposed relative attention.  This is where the global context is captured.

### 2. Relative Attention Mechanism

* **Combining Convolutions and Self-Attention:** Astroformer utilizes **relative attention** to effectively combine depthwise convolutions and self-attention within the Transformer block in Stage S4.
* **Depthwise Convolutions for Local Features:** Depthwise convolutions are used to extract local features efficiently. They use a fixed kernel to process each channel independently.
* **Self-Attention for Global Context:** Self-attention allows the model to attend to all parts of the input feature map, capturing global relationships and dependencies.
* **Relative Attention Formula (Equation 1):** The paper presents the formula for their relative attention mechanism:

   ```math
   Yi =  ∑_{j∈G}  [ exp(x_i^T x_j + W_{i-j}) / ∑_{k∈G} exp(x_i^T x_k + W_{i-k}) ] * x_j
   ```

  * **x_i, Y_i:** Input and output vectors at position *i*.
  * **W_{i-j}:** Depthwise convolution kernel, dependent on the *relative* position (i-j).
  * **G:** Global spatial space (all positions in the feature map).
  * **exp(x_i^T x_j):** Standard self-attention term, measuring similarity between input vectors at positions *i* and *j*.
  * **W_{i-j} term:**  The key addition is the depthwise convolution kernel *W_{i-j}*. This kernel is *static* (not learned per input) and provides a fixed, translationally equivariant inductive bias, enhancing the attention mechanism.
  * **Attention Weight (A_{i,j}):** The attention weight between positions *i* and *j* is determined by *both* the self-attention similarity (x_i^T x_j) and the depthwise convolution kernel (W_{i-j}).
  * **Interpretation:** The update to the attention weight is achieved by effectively adding a global static convolution kernel.  The attention is "relative" because the kernel W depends on the *relative* position (i-j), not the absolute positions *i* and *j*.

### 3. Network Construction Process

* **Multi-Stage Architecture:**  The network is built in stages (S0-S4) with gradual downsampling of the feature map.
* **Downsampling:** Feature map size is reduced progressively through the stages, reducing spatial size and increasing channel depth. This is achieved using pooling and strided convolutions.
* **Global Relative Attention in S4:**  Global relative attention is applied in the final stage (S4) after spatial dimensions have been reduced, making the computation more efficient.

### 4. Augmentation and Regularization Techniques

* **Crucial for Low-Data Regimes:** The paper emphasizes that careful selection of augmentation and regularization is **critical** for good performance in low-data settings. They found that strong augmentations are more effective than strong regularization in this context.
* **Augmentations:**
  * **Mixup (Zhang et al., 2017):**  Linearly interpolates between pairs of images and their corresponding labels.
  * **RandAugment (Cubuk et al., 2020):**  Automatically selects a set of augmentation operations and their magnitudes from a predefined set.
* **Regularization:**
  * **Stochastic Depth Regularization:** Randomly drops layers during training to improve generalization.
  * **Weight Decay:** L2 regularization to prevent overfitting by penalizing large weights.
  * **Label Smoothing:**  Softens the target labels to prevent overconfidence and improve calibration.
* **Avoidance of Certain Augmentations (CutMix, DropBlock, Cutout):** The authors found that regional dropout-based augmentations like CutMix, DropBlock, and Cutout were **detrimental** for the Galaxy10 DECals dataset. This is attributed to the sensitivity of galaxy morphology labels. Even small occlusions can change the perceived morphology and thus the ground truth label, negatively impacting model training.

### 5. Reasons for Good Performance in Low-Data Regimes

The paper summarizes the reasons why Astroformer performs well in low-data regimes:

1. **Careful Augmentation and Regularization:**  Judicious selection and tuning of these techniques are paramount, especially in smaller datasets.
2. **Generalizability of Hybrid Model:** The hybrid transformer-convolutional architecture exhibits good generalizability, avoiding overfitting and handling unseen data better.
3. **Stable Training:** The C-C-C-T design and the relative attention mechanism contribute to more stable training compared to deeper transformer-based models (like C-C-T-T or C-T-T-T) in low-data scenarios.
4. **Translational Equivalence:** The inherent translational equivalence of convolutional operations (and the designed relative attention) helps reduce overfitting in low-data regimes, making the model less sensitive to the exact position of features in the input image.

**In essence, the "Methodology" section lays out a well-reasoned and carefully crafted approach for image classification in low-data scenarios. It combines architectural innovations (Astroformer, C-C-C-T stack, relative attention) with tailored training strategies (specific augmentations and regularizations) to achieve state-of-the-art results and address the challenges of data scarcity.**

## Going to use

* Astroformer's Design (C-C-C-T)
* Augmentations: Mixup + RandAugment
* Regularization:Label Smoothing
* Avoidance of Certain Augmentations (CutMix, DropBlock, Cutout)
