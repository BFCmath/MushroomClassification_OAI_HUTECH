# MushroomClassification_OAI_HUTECH

## Introduction

This repository contains my solution for the Olympiad in Artificial Intelligence at Ho Chi Minh City 2025 (OAI HCMC 2025), hosted by HUTECH University. The competition aims to foster academic exchange and promote research in artificial intelligence among students from universities and colleges across Vietnam.

The challenge focuses on the classification of four common mushroom species in Vietnam using machine learning techniques. Participants are provided with a dataset of 1,400 color images (32x32 pixels) of mushrooms, evenly distributed among Pleurotus ostreatus, Coprinus comatus, Ganoderma lucidum, and Agaricus bisporus. The goal is to develop a model that accurately identifies the mushroom type from an image, with performance evaluated based on accuracy on a test set.

This repository includes all code, experiments, and documentation related to my approach for the competition.

## TL;DR

This repository contains my solution for the mushroom classification task in the OAI HCMC 2025 competition. The challenge involved working with a low-resolution (32x32), small dataset of 1,400 images, with no external data allowed and potential misalignment between train and test sets.

To tackle these challenges, I developed ConvNet-based models incorporating:
- **[Mixup Class](#mixup-class)**: A novel augmentation technique to improve robustness by introducing a synthetic fifth class.
- **[Selective Augmentation](#augmentation-strategies)**: Carefully chosen color and spatial augmentations to enhance generalization while preserving critical low-resolution features.
- **[Attack Validation](#attack-validation)**: A robustness assessment method using augmented test sets to evaluate model performance under challenging conditions.

The final model achieved:
- **Public Test Accuracy**: 0.925 (best submission) and 0.90 (average across random seeds).
- **Private Test Ranking**: 23rd out of ~140 teams, with an accuracy of 0.83 and an F1 score of 0.8265.

For more details on the methodology, experiments, and results, refer to [my report](BFC/BFC_report.pdf).

## Dataset

The dataset provided for this competition consists of 1,400 color images of mushrooms, captured at a fixed resolution of 32x32 pixels. The images are evenly distributed among four common mushroom species found in Vietnam:

- **Pleurotus ostreatus** (Nấm bào ngư) – label: 1
- **Coprinus comatus** (Nấm đùi gà) – label: 2
- **Ganoderma lucidum** (Nấm linh chi trắng) – label: 3
- **Agaricus bisporus** (Nấm mỡ) – label: 0

Each class contains 350 images, with 300 images designated for training and 50 images for testing. The images were taken from various angles (top, bottom, cross-section, and close-up) under both natural and artificial lighting conditions to ensure diversity.

All images are manually labeled to ensure high-quality ground truth for model training and evaluation. The dataset is designed to support research in image classification and computer vision, specifically for the task of mushroom species recognition.

**Label mapping:**

| Label | Species (Vietnamese)      | Species (English)        |
|-------|--------------------------|---------------------------|
| 0     | Nấm mỡ                   | Agaricus bisporus         |
| 1     | Nấm bào ngư              | Pleurotus ostreatus       |
| 2     | Nấm đùi gà               | Coprinus comatus          |
| 3     | Nấm linh chi trắng       | Ganoderma lucidum         |

The dataset is split into training and test sets, and the final evaluation is based on the accuracy of predictions on the test set.

## Problem Statement

The goal of this competition is to build a robust model that classifies four mushroom species from low-resolution (32x32) color images. The dataset is small (1,400 images), and the use of external data is prohibited. Due to the lack of suitable pre-trained models for this specific task, solutions must rely on learning effective features directly from the provided data. Additionally, potential distribution shifts between train and test sets and evaluation with varying random seeds require models to generalize well and be robust to initialization.

## Methodology

### Mixup Class

To improve robustness and handle ambiguous inputs, I implemented a "Mixup Class" augmentation. This technique generates a synthetic fifth class (label 4) by averaging the pixel values of randomly selected images from each of the four original classes. These composite images are created offline and added to the training set, transforming the task into a five-class classification problem.

### Augmentation Strategies

Data augmentation was essential for improving the model's generalization and robustness, especially given the low resolution and limited size of the dataset. Two categories of augmentations were applied: **color augmentations** and **spatial augmentations**.

#### Color Augmentations

Color transformations were designed to make the model invariant to variations in lighting, camera sensors, and color casts. The following techniques were applied randomly during training:

- **RandomColorDrop**
- **RandomChannelSwap**
- **RandomGamma**
- **SimulateHSVNoise**

These augmentations significantly improved training stability and model performance by simulating diverse lighting and color conditions.

#### Spatial Augmentations

Spatial transformations were applied conservatively to avoid distorting critical patterns in the low-resolution images. The following techniques were used:

- **MultiScaleTransform**: Randomly rescales the image within a defined range to simulate variations in camera distance.
- **RandomCropAndZoom**: Crops and zooms into random regions of the image to mimic slight variations in framing.

Overly aggressive spatial augmentations, such as Cutout or strong flips, were avoided as they disrupted the low-resolution grid and led to training instability.

These carefully selected augmentations helped the model generalize better while preserving the essential features of the low-resolution dataset.

### Attack Validation

To ensure the robustness of the model, I implemented an "Attack Validation" procedure to evaluate performance under challenging conditions. This approach involved creating augmented versions of the public test set using strong, targeted perturbations. The following augmentations were applied individually to simulate real-world challenges:

- **Change Background**: Alters the background to test the model's focus on the mushroom.
- **Colorize Mushroom**: Applies color transformations to simulate unusual lighting or color casts.
- **Darker Shadow**: Adds shadows to test robustness under poor lighting.
- **Random Blur**: Blurs the image to simulate motion or focus issues.
- **Rotate Zoom**: Rotates and zooms the image to mimic framing variations.
- **Simulate Light**: Adjusts lighting conditions to test adaptability.

The model's accuracy was assessed on each augmented test set, with a specific focus on maintaining a threshold accuracy of ≥ 0.70 on the challenging **Random Blur** attack. These attacks were used exclusively for validation and not during training.

This validation step was critical for identifying and eliminating approaches that performed well on standard test data but failed under more challenging conditions.

## Results

### Public Test Leaderboard

Achieved an accuracy of 0.925 on the public test set for the best submission and an average accuracy of 0.90 across several random seeds.

![leaderboard](image/README_2025-04-28-10-11-40.png)

### Private Test Leaderboard

Ranked 23rd out of approximately 140 teams on the private test set, with an accuracy of 0.83 and an F1 score of 0.8265.

> A little disappointed as the top 20 advance to the final round, but I am proud of this solo achievement among such strong teams from various schools.

![leaderboard](image/README_2025-04-28-10-14-53.png)

## How to run
Please refer to [this folder](BFC)

## File Structure

```css
MushroomClassification_OAI_HUTECH/
│
├── BFC/                     # Folder containing the report and additional documentation
│   ├── BFC_report.pdf       # Detailed report of the methodology and results
│   ├── README.md            # Instructions for running the code
│   └── ...
│
├── models/                  # Folder containing model architecture descriptions and flowcharts
│   └── ...                  # Other model-related files
│
├── augmentation/            # Folder for visualization the augmentation effects
│   ├── advanced_spatial_transforms.py  # Advanced spatial augmentation techniques
│   └── ...  
│
├── simulate_attack/         # Folder for attack simulation scripts
│   ├── change_background.py # Script for simulating background changes
│   └── ...                  # Other attack simulation scripts
│
├── scripts/                 # Folder for various scripts
│   ├── caps/                # Capsule network-related scripts
│   ├── cnn/                 # Convolutional network-related scripts
│   ├── main/                # Main scripts for training and evaluation
│   └── ...                  # Other utility scripts
│
├── research/                # Folder for research notes and references
│   └── ...                  # Research-related markdown files
│
├── templates/               # Folder for visualization templates 
│   ├── image_view.html      # Image viewer template
│   └── ...                  # Other templates
│
├── split_cv/                # Folder for cross-validation scripts
│   └── ...                  
│
├── README.md                # Main README file for the repository
└── IDEA.md                  # Notes and ideas for the project
```

## Future Work

### What I Learned
- **Research Skills**: Improved my ability to find and understand papers in cutting-edge fields.
- **Engineering Skills**: Gained experience in writing scripts, implementing multi-GPU training, and coding directly from research papers and source code instead of relying on APIs.
- **Capsule Networks**: Learned about capsule networks and their equivariance bias, which could be useful for future tasks.

### Areas for Improvement
- **Model Robustness**: Explore more techniques to make models even more robust under challenging conditions.
- **Time Management**: Better balance between research-oriented approaches and engineering/development tasks.
- **Variance Control**: Develop strategies to better control and compare variance between models and changes in architecture.

### Future Plans
- **Research Ideas**: This competition has inspired many ideas for further research, particularly in low-resolution vision tasks and robust model design.
- **Engineering Efficiency**: The engineering skills I gained will help me save time in future competitions and projects, allowing me to focus more on innovation and experimentation.
