# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Import library

# %% [code] {"execution":{"iopub.status.busy":"2025-04-19T04:08:42.162287Z","iopub.execute_input":"2025-04-19T04:08:42.162623Z","iopub.status.idle":"2025-04-19T04:08:42.170959Z","shell.execute_reply.started":"2025-04-19T04:08:42.162596Z","shell.execute_reply":"2025-04-19T04:08:42.170299Z"},"jupyter":{"source_hidden":true}}
### START: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###
SEED = 24  # Số seed (Ban tổ chức sẽ công bố & thay đổi vào lúc chấm)
# Đường dẫn đến thư mục train
# (đúng theo cấu trúc gồm 4 thư mục cho 4 classes của ban tổ chức)
TRAIN_DATA_DIR_PATH = 'data/train'
# Đường dẫn đến thư mục test
TEST_DATA_DIR_PATH = 'data/test'
### END: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###
OUTPUT_DIR_PATH = 'output'
### START: CÁC THƯ VIỆN IMPORT ###
# Lưu ý: các thư viện & phiên bản cài đặt vui lòng để trong requirements.txt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.nn.modules.loss import _Loss

import numpy as np
import pandas as pd
import random
import time
import json
import math
import os
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from sklearn.metrics import confusion_matrix

from PIL import Image, ImageOps, ImageEnhance # this is new 
from torchvision import models, transforms # this is new
import cv2 # this is new 

from scipy.ndimage import map_coordinates, gaussian_filter # this is new 

import warnings

from sklearn.model_selection import StratifiedKFold
### END: CÁC THƯ VIỆN IMPORT ###

### START: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###
# Seeding nhằm đảm bảo kết quả sẽ cố định
# và không ngẫu nhiên ở các lần chạy khác nhau
# Set seed for random
random.seed(SEED)
# Set seed for numpy
np.random.seed(SEED)
# Set seed for torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Set seed for tensorflow
tf.random.set_seed(SEED)
### END: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###

# %% [code] {"execution":{"iopub.status.busy":"2025-04-19T04:08:42.171827Z","iopub.execute_input":"2025-04-19T04:08:42.172027Z","iopub.status.idle":"2025-04-19T04:08:42.188077Z","shell.execute_reply.started":"2025-04-19T04:08:42.172010Z","shell.execute_reply":"2025-04-19T04:08:42.187494Z"},"jupyter":{"outputs_hidden":false}}
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# # SPLIT CV

# %% [code] {"execution":{"iopub.status.busy":"2025-04-19T04:08:42.189045Z","iopub.execute_input":"2025-04-19T04:08:42.189247Z","iopub.status.idle":"2025-04-19T04:08:42.330624Z","shell.execute_reply.started":"2025-04-19T04:08:42.189221Z","shell.execute_reply":"2025-04-19T04:08:42.329804Z"}}
os.makedirs('split_cv', exist_ok=True)

# %% [code] {"jupyter":{"outputs_hidden":true,"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:42.331673Z","iopub.execute_input":"2025-04-19T04:08:42.331897Z","iopub.status.idle":"2025-04-19T04:08:43.323984Z","shell.execute_reply.started":"2025-04-19T04:08:42.331876Z","shell.execute_reply":"2025-04-19T04:08:43.323329Z"}}
# Define paths and configuration
TRAIN_DIR = TRAIN_DATA_DIR_PATH
OUTPUT_CSV = 'split_cv/train_cv.csv'
CV_FOLDS = 6
RANDOM_SEED = SEED
PREFIX = TRAIN_DATA_DIR_PATH
# Define class names
CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)','linh chi trắng']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def create_cv_splits():
    """
    Create cross-validation splits and save to CSV.
    Uses stratified k-fold to ensure class distribution is preserved across folds.
    """
    print(f"Creating {CV_FOLDS}-fold cross-validation splits...")
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    
    # Gather training data
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        prefix_dir = os.path.join(PREFIX, class_name)
        class_idx = CLASS_MAP[class_name]
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue
            
        print(f"Processing class {class_name} ({class_idx})...")
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(prefix_dir, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(img_path)
                labels.append(class_idx)
    
    # Convert to numpy arrays for easier handling
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Initialize fold column with -1 (not assigned)
    fold_assignments = -1 * np.ones(len(labels), dtype=int)
    
    # Assign fold IDs to each sample
    for fold_idx, (_, val_idx) in enumerate(skf.split(image_paths, labels)):
        fold_assignments[val_idx] = fold_idx
    
    # Create DataFrame with all information
    cv_df = pd.DataFrame({
        'image_path': image_paths,
        'class_id': labels, 
        'class_name': [CLASS_NAMES[label] for label in labels],
        'fold': fold_assignments
    })
    
    # Verify that all samples have been assigned a fold
    assert (cv_df['fold'] >= 0).all(), "Error: Some samples were not assigned to any fold"
    
    # Save to CSV
    cv_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Cross-validation splits saved to {OUTPUT_CSV}")
    
    return cv_df

def split():
    """Main function to create and analyze CV splits."""
    # Create CV splits
    cv_df = create_cv_splits()
    
    # Analyze the splits
    # distribution_df = analyze_cv_splits(cv_df)
    
split()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Dataset

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.324686Z","iopub.execute_input":"2025-04-19T04:08:43.324904Z","iopub.status.idle":"2025-04-19T04:08:43.344797Z","shell.execute_reply.started":"2025-04-19T04:08:43.324885Z","shell.execute_reply":"2025-04-19T04:08:43.344222Z"}}
class MushroomDataset(Dataset):
    """Dataset for mushroom classification."""
    
    def __init__(self, csv_path, transform=None, root_dir=None):
        """
        Args:
            csv_path: Path to CSV file with annotations
            transform: Optional transform to be applied
            root_dir: Root directory for the images
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.root_dir = root_dir
        
        # Extract unique classes from the CSV file (assuming 'class' column exists)
        if 'class' in self.data.columns:
            self.classes = sorted(self.data['class'].unique())
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            # Assign dummy classes for inference-only datasets
            self.classes = ['unknown']
            self.class_to_idx = {'unknown': 0}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['path']
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
            
        # Load image from path - with error handling
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image of the expected size with black pixels
            image = Image.new('RGB', (32, 32), color='black')
        
        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)
        
        # Get label if available (class column must exist)
        if 'class' in self.data.columns:
            class_name = self.data.iloc[idx]['class']
            label = self.class_to_idx[class_name]
            return image, label
        else:
            # For inference datasets, return image and placeholder label
            return image, 0

class MixupDataset(Dataset):
    """
    Wrapper dataset that adds mixup class to training data.
    The mixup class is generated by combining images from different classes.
    """
    def __init__(self, base_dataset, mixup_ratio=0.2, mixup_class_name="mixup", strategy="average"):
        """
        Initialize the MixupDataset.
        
        Args:
            base_dataset: The base dataset to wrap
            mixup_ratio: Ratio of mixup samples to original samples
            mixup_class_name: Name of the mixup class
            strategy: How to combine images - "average", "overlay", or "mosaic"
        """
        self.base_dataset = base_dataset
        self.mixup_ratio = mixup_ratio
        self.strategy = strategy
        self.classes = base_dataset.classes.copy() if hasattr(base_dataset, 'classes') else []
        self.classes.append(mixup_class_name)  # Add mixup class name
        
        # Create mapping of class indices to sample indices
        self.class_to_samples = {}
        for i, (_, target) in enumerate(base_dataset):
            if target not in self.class_to_samples:
                self.class_to_samples[target] = []
            self.class_to_samples[target].append(i)
            
        # Calculate number of mixup samples to generate
        num_original = len(base_dataset)
        self.num_mixup = int(num_original * mixup_ratio)
        self.mixup_class_idx = len(self.classes) - 1  # Index of the mixup class
        
        # Store class indices mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.base_classes = base_dataset.classes if hasattr(base_dataset, 'classes') else []
        
        # Count unique classes in the dataset
        self.unique_class_count = len(set(self.class_to_samples.keys()))
        print(f"Dataset has {self.unique_class_count} unique classes")
        
        # Check if we have enough classes for mosaic strategy
        if strategy == "mosaic" and self.unique_class_count < 4:
            print(f"Warning: Only {self.unique_class_count} classes available, but mosaic strategy requires 4 different classes.")
            print("Falling back to 'average' strategy for mixup.")
            self.strategy = "average"
        else:
            self.strategy = strategy
    
    def __len__(self):
        """Return the total size of the dataset including mixup samples."""
        return len(self.base_dataset) + self.num_mixup
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if idx < len(self.base_dataset):
            # Return original sample
            return self.base_dataset[idx]
        
        # Generate a mixup sample
        mixup_idx = idx - len(self.base_dataset)
        
        if self.strategy == "mosaic":
            # Mosaic strategy - combine 4 images from different classes in a grid
            img_tensor = torch.zeros(3, 32, 32)  # Default size, will be adjusted by transform
            
            # Get all available class indices
            class_indices = list(self.class_to_samples.keys())
            
            # Always select 4 different classes for mosaic
            # This is possible because we checked in __init__ that we have at least 4 classes
            selected_classes = random.sample(class_indices, 4)
            
            # Get a random sample from each selected class
            sample_indices = [random.choice(self.class_to_samples[cls]) for cls in selected_classes]
            images = [self.base_dataset[idx][0] for idx in sample_indices]
            
            # Print the classes used for this mosaic (helpful for debugging)
            if random.random() < 0.01:  # Only print occasionally (1% of the time)
                class_names = [self.base_dataset.dataset.classes[cls] if hasattr(self.base_dataset, 'dataset') else str(cls) for cls in selected_classes]
                print(f"Creating mosaic with classes: {class_names}")
            
            # Combine into a mosaic (2x2 grid)
            # Top-left
            h, w = images[0].shape[1] // 2, images[0].shape[2] // 2
            img_tensor[:, :h, :w] = images[0][:, :h, :w]
            # Top-right
            img_tensor[:, :h, w:] = images[1][:, :h, w:]
            # Bottom-left
            img_tensor[:, h:, :w] = images[2][:, h:, :w]
            # Bottom-right
            img_tensor[:, h:, w:] = images[3][:, h:, w:]
        
        elif self.strategy == "overlay":
            # Overlay strategy - overlay two images with transparency
            # Select 2 random classes and samples
            class_indices = list(self.class_to_samples.keys())
            classes = random.sample(class_indices, min(2, len(class_indices)))
            
            if len(classes) < 2:  # Handle case with only one class
                classes = classes * 2
                
            sample_indices = [random.choice(self.class_to_samples[cls]) for cls in classes]
            img1, _ = self.base_dataset[sample_indices[0]]
            img2, _ = self.base_dataset[sample_indices[1]]
            
            # Random alpha for blending
            alpha = random.uniform(0.3, 0.7)
            img_tensor = alpha * img1 + (1 - alpha) * img2
        
        else:  # Default to "average" strategy
            # Average strategy - average multiple images
            # Select 2-4 random classes and samples
            class_indices = list(self.class_to_samples.keys())
            num_samples = random.randint(2, min(4, len(class_indices)))
            classes = random.sample(class_indices, min(num_samples, len(class_indices)))
            
            if len(classes) < 2:  # Handle case with only one class
                classes = classes * 2
                
            sample_indices = [random.choice(self.class_to_samples[cls]) for cls in classes]
            images = [self.base_dataset[idx][0] for idx in sample_indices]
            
            # Simple average
            img_tensor = sum(images) / len(images)
        
        # Return the mixup image with the mixup class label
        target = self.mixup_class_idx
        return img_tensor, target
    
    def get_original_num_classes(self):
        """Return the number of original classes (without mixup)."""
        return len(self.base_classes)

class CustomSubset(Dataset):
    """Custom subset that applies transform to the original dataset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
    
    @property
    def classes(self):
        """Access classes attribute from the underlying dataset."""
        if hasattr(self.subset, 'dataset') and hasattr(self.subset.dataset, 'classes'):
            return self.subset.dataset.classes
        elif hasattr(self.subset, 'classes'):
            return self.subset.classes
        raise AttributeError("Could not find 'classes' attribute in the underlying dataset")
    
    @property
    def dataset(self):
        """Access the original dataset for compatibility."""
        if hasattr(self.subset, 'dataset'):
            return self.subset.dataset
        return self.subset

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Transform

# %% [code] {"execution":{"iopub.status.busy":"2025-04-19T04:08:43.346305Z","iopub.execute_input":"2025-04-19T04:08:43.346511Z","iopub.status.idle":"2025-04-19T04:08:43.429638Z","shell.execute_reply.started":"2025-04-19T04:08:43.346492Z","shell.execute_reply":"2025-04-19T04:08:43.429059Z"},"jupyter":{"source_hidden":true}}
class AdvancedColorTransforms:
    """Collection of advanced color transformation techniques."""
    
    class RandomColorDrop(object):
        """Randomly zero out a color channel with given probability."""
        def __init__(self, p=0.2):
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                img_array = np.array(img)
                channel = random.randint(0, 2)  # Choose R, G, or B channel
                img_array[:, :, channel] = 0
                return Image.fromarray(img_array)
            return img
    
    class RandomChannelSwap(object):
        """Randomly swap color channels."""
        def __init__(self, p=0.2):
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                img_array = np.array(img)
                # Generate a random permutation of [0,1,2]
                perm = list(range(3))
                random.shuffle(perm)
                img_array = img_array[:, :, perm]
                return Image.fromarray(img_array)
            return img
    
    class RandomGamma(object):
        """Apply random gamma correction."""
        def __init__(self, gamma_range=(0.5, 1.5), p=0.3):
            self.gamma_range = gamma_range
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                gamma = random.uniform(*self.gamma_range)
                # Convert to numpy for gamma correction
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = np.power(img_array, gamma)
                img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                return Image.fromarray(img_array)
            return img
    
    class SimulateHSVNoise(object):
        """Simulate HSV space noise by shifting H, S, V channels slightly."""
        def __init__(self, hue_shift=0.05, sat_shift=0.1, val_shift=0.1, p=0.3):
            self.hue_shift = hue_shift
            self.sat_shift = sat_shift
            self.val_shift = val_shift
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                # Convert PIL to cv2 image (RGB to BGR)
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # Convert to HSV
                img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV).astype(np.float32)
                
                # Apply random shifts to each channel
                h_shift = random.uniform(-self.hue_shift, self.hue_shift) * 180
                s_shift = random.uniform(-self.sat_shift, self.sat_shift) * 255
                v_shift = random.uniform(-self.val_shift, self.val_shift) * 255
                
                img_hsv[:, :, 0] = np.mod(img_hsv[:, :, 0] + h_shift, 180)
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] + s_shift, 0, 255)
                img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] + v_shift, 0, 255)
                
                # Convert back to RGB
                img_cv = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img_rgb)
            return img
    
    class SimulateLightingCondition(object):
        """Simulate different lighting conditions."""
        def __init__(self, p=0.3):
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                # Randomly choose a lighting simulation effect
                effect = random.choice(['warm', 'cool', 'bright', 'dark'])
                
                if effect == 'warm':
                    # Add warm/yellowish tint
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(1.2)
                    r, g, b = img.split()
                    r = ImageEnhance.Brightness(r).enhance(1.1)
                    g = ImageEnhance.Brightness(g).enhance(1.0)
                    b = ImageEnhance.Brightness(b).enhance(0.9)
                    return Image.merge('RGB', (r, g, b))
                
                elif effect == 'cool':
                    # Add cool/bluish tint
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(1.2)
                    r, g, b = img.split()
                    r = ImageEnhance.Brightness(r).enhance(0.9)
                    g = ImageEnhance.Brightness(g).enhance(1.0)
                    b = ImageEnhance.Brightness(b).enhance(1.1)
                    return Image.merge('RGB', (r, g, b))
                    
                elif effect == 'bright':
                    # Increase brightness
                    enhancer = ImageEnhance.Brightness(img)
                    return enhancer.enhance(1.2)
                    
                else:  # dark
                    # Decrease brightness
                    enhancer = ImageEnhance.Brightness(img)
                    return enhancer.enhance(0.8)
            return img
    
    class RandomPixelNoise(object):
        """
        Add white, black, or random-colored pixels to the image.
        
        For 32x32 images, this is a gentler alternative to cutout,
        preserving more of the original object information.
        
        Args:
            p: Probability of applying the transform
            percent_pixels: Percentage of total pixels to modify (0.0-1.0)
            percent_range: Range of percentage to use (min_factor, max_factor)
                           Actual percentage will be percent_pixels * random factor in this range
            white_prob: Probability of adding white pixels (default 0.4)
            black_prob: Probability of adding black pixels (default 0.4)
        """
        def __init__(self, p=0.3, percent_pixels=0.05, percent_range=(0.75, 1.25), white_prob=0.4, black_prob=0.4):
            self.p = p
            self.percent_pixels = percent_pixels  # Base percentage (e.g., 0.01 = 1% of pixels)
            self.percent_range = percent_range    # Range multiplier for randomization
            self.white_prob = white_prob
            self.black_prob = black_prob
            self.random_prob = 1.0 - (white_prob + black_prob)
            print(f"Random pixel aug is used with {percent_pixels}!")
            
        def __call__(self, img):
            if random.random() < self.p:
                # Convert to numpy array for pixel manipulation
                img_array = np.array(img)
                height, width, channels = img_array.shape
                
                # Calculate number of pixels to modify based on image size
                total_pixels = height * width
                factor = random.uniform(*self.percent_range)
                num_pixels = max(1, int(total_pixels * self.percent_pixels * factor))
                
                # Randomly choose coordinates
                y_coords = np.random.randint(0, height, num_pixels)
                x_coords = np.random.randint(0, width, num_pixels)
                
                # Determine pixel color type based on probabilities
                pixel_type = random.random()
                
                if pixel_type < self.white_prob:
                    # Add white pixels
                    for y, x in zip(y_coords, x_coords):
                        img_array[y, x, :] = 255
                elif pixel_type < self.white_prob + self.black_prob:
                    # Add black pixels
                    for y, x in zip(y_coords, x_coords):
                        img_array[y, x, :] = 0
                else:
                    # Add random colored pixels
                    for y, x in zip(y_coords, x_coords):
                        img_array[y, x, :] = [random.randint(0, 255) for _ in range(channels)]
                
                return Image.fromarray(img_array)
            return img
    
    class RandomGrayscale(object):
        """Convert image to grayscale with probability p, but keep 3 channels."""
        def __init__(self, p=0.2):
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                img_gray = ImageOps.grayscale(img)
                # Convert back to 3 channels properly
                # The current approach is incorrect - you can't merge the same image object 3 times
                # Using numpy for proper channel stacking instead:
                img_array = np.array(img_gray)
                img_array_3channel = np.stack([img_array, img_array, img_array], axis=-1)
                return Image.fromarray(img_array_3channel)
            return img
        
# Add new custom spatial transforms
class AdvancedSpatialTransforms:
    """Collection of advanced spatial transformation techniques."""
    
    class RandomCropAndZoom(object):
        """
        Randomly crop a portion of the image and resize it back to the original size,
        creating a zooming effect while maintaining original dimensions.
        
        Args:
            crop_scale: The size of the crop relative to original image (0.0-1.0)
                        Lower values = stronger zoom effect
            p: Probability of applying the transform
        """
        def __init__(self, crop_scale=0.9, p=0.5):
            self.crop_scale = crop_scale
            self.p = p
            print(f"Using AdvancedSpatialTransforms with {crop_scale} crop scale")
            
        def __call__(self, img):
            if random.random() < self.p:
                width, height = img.size
                
                # Calculate crop dimensions (e.g., 90% of original size)
                crop_width = int(width * self.crop_scale)
                crop_height = int(height * self.crop_scale)
                
                # Calculate maximum offsets to keep crop within image
                max_x = width - crop_width
                max_y = height - crop_height
                
                # Get random crop position
                x = random.randint(0, max(0, max_x))
                y = random.randint(0, max(0, max_y))
                
                # Perform crop
                cropped_img = img.crop((x, y, x + crop_width, y + crop_height))
                
                # Resize back to original dimensions
                zoomed_img = cropped_img.resize((width, height), Image.BICUBIC)
                
                return zoomed_img
            return img
            
    class RadialDistortion:
        """
        Apply barrel or pincushion distortion to simulate lens effects and natural cap curvature.
        
        Args:
            distortion_type: 'barrel' (outward) or 'pincushion' (inward)
            strength: Distortion strength (0.0-1.0)
            p: Probability of applying transform
        """
        def __init__(self, distortion_type='barrel', strength=0.2, p=0.5):
            self.distortion_type = distortion_type
            self.strength = strength * (1 if distortion_type == 'barrel' else -1)
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                # Convert to numpy array
                img_np = np.array(img).astype(np.float32) / 255.0
                
                h, w = img_np.shape[:2]
                center_x, center_y = w / 2, h / 2
                
                # Create meshgrid for the coordinates
                x = np.arange(w)
                y = np.arange(h)
                X, Y = np.meshgrid(x, y)
                
                # Calculate distance from center and normalize
                dist_x, dist_y = X - center_x, Y - center_y
                r = np.sqrt(dist_x**2 + dist_y**2) / (np.sqrt(center_x**2 + center_y**2))
                
                # Apply distortion
                d = 1.0 + self.strength * (r**2)
                
                # Create new coordinates
                X_distorted = center_x + dist_x * d
                Y_distorted = center_y + dist_y * d
                
                # Clip to valid image coordinates
                X_distorted = np.clip(X_distorted, 0, w - 1).astype(np.float32)
                Y_distorted = np.clip(Y_distorted, 0, h - 1).astype(np.float32)
                
                # Remap image
                result = np.zeros_like(img_np)
                for c in range(img_np.shape[2]):
                    result[:,:,c] = cv2.remap(img_np[:,:,c], X_distorted, Y_distorted, 
                                             interpolation=cv2.INTER_LINEAR, 
                                             borderMode=cv2.BORDER_REFLECT)
                
                # Convert back to PIL Image
                return Image.fromarray((result * 255).astype(np.uint8))
            return img

    class ElasticDeformation:
        """
        Apply elastic deformation to simulate natural mushroom growth variations.
        
        Args:
            alpha: Deformation intensity
            sigma: Smoothness of deformation
            p: Probability of applying transform
        """
        def __init__(self, alpha=5.0, sigma=2.0, p=0.5):
            self.alpha = alpha
            self.sigma = sigma
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                # Convert PIL Image to numpy array
                img_np = np.array(img)
                
                # Get image shape
                shape = img_np.shape
                
                # Create random displacement fields
                dx = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), self.sigma) * self.alpha
                dy = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), self.sigma) * self.alpha
                
                # Create meshgrid for coordinates
                x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                
                # Displace meshgrid indices
                indices_x = np.reshape(x + dx, (-1, 1))
                indices_y = np.reshape(y + dy, (-1, 1))
                
                # Create distorted image for each channel
                result = np.zeros_like(img_np)
                for c in range(shape[2]):
                    result[:, :, c] = map_coordinates(
                        img_np[:, :, c], 
                        [indices_y.flatten(), indices_x.flatten()],
                        order=1
                    ).reshape(shape[:2])
                    
                return Image.fromarray(result)
            return img

    class CentralFocusZoom:
        """
        Apply selective zoom focused on the center region while maintaining periphery.
        Good for emphasizing cap-stem junction.
        
        Args:
            strength: Zoom strength (0.0-1.0)
            center_size: Relative size of center region (0.0-1.0)
            p: Probability of applying transform
        """
        def __init__(self, strength=0.3, center_size=0.5, p=0.5):
            self.strength = strength
            self.center_size = center_size
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                img_np = np.array(img).astype(np.float32) / 255.0
                h, w = img_np.shape[:2]
                
                # Create meshgrid
                x = np.linspace(-1, 1, w)
                y = np.linspace(-1, 1, h)
                X, Y = np.meshgrid(x, y)
                
                # Calculate radius from center (normalized)
                R = np.sqrt(X**2 + Y**2)
                
                # Create zoom function - more zoom in center, less at edges
                zoom_factor = 1 + self.strength * (1 - np.minimum(1, R / self.center_size))
                
                # Apply non-uniform zoom
                X_zoomed = X / zoom_factor
                Y_zoomed = Y / zoom_factor
                
                # Scale back to image coordinates
                X_zoomed = (X_zoomed + 1) * (w - 1) / 2
                Y_zoomed = (Y_zoomed + 1) * (h - 1) / 2
                
                # Remap image
                result = np.zeros_like(img_np)
                for c in range(img_np.shape[2]):
                    result[:,:,c] = cv2.remap(img_np[:,:,c], 
                                             X_zoomed.astype(np.float32), 
                                             Y_zoomed.astype(np.float32), 
                                             interpolation=cv2.INTER_LINEAR, 
                                             borderMode=cv2.BORDER_REFLECT)
                
                return Image.fromarray((result * 255).astype(np.uint8))
            return img

    class AspectRatioVariation:
        """
        Apply subtle changes to height/width ratio to simulate different growth stages.
        
        Args:
            ratio_range: Range of aspect ratio changes (min, max)
            p: Probability of applying transform
        """
        def __init__(self, ratio_range=(0.9, 1.1), p=0.5):
            self.ratio_range = ratio_range
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                # Get original size
                width, height = img.size
                
                # Choose random aspect ratio change
                ratio = random.uniform(*self.ratio_range)
                
                # Apply to width or height randomly
                if random.random() < 0.5:
                    new_width = int(width * ratio)
                    new_height = height
                else:
                    new_width = width
                    new_height = int(height * ratio)
                
                # Resize with new aspect ratio
                resized = img.resize((new_width, new_height), Image.BICUBIC)
                
                # Create new image with original size and paste resized image in center
                result = Image.new('RGB', (width, height))
                paste_x = (width - new_width) // 2
                paste_y = (height - new_height) // 2
                result.paste(resized, (paste_x, paste_y))
                
                return result
            return img

    class GridShuffle:
        """
        Divide image into grid cells and randomly shuffle them.
        Forces model to focus on local patterns rather than global structure.
        
        Args:
            grid_size: Number of grid divisions (2=2x2 grid, 3=3x3 grid, etc.)
            p: Probability of applying transform
        """
        def __init__(self, grid_size=2, p=0.3):
            self.grid_size = grid_size
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                width, height = img.size
                cell_width = width // self.grid_size
                cell_height = height // self.grid_size
                
                # Create grid cells
                cells = []
                for y in range(self.grid_size):
                    for x in range(self.grid_size):
                        box = (x * cell_width, y * cell_height, 
                              (x + 1) * cell_width, (y + 1) * cell_height)
                        cell = img.crop(box)
                        cells.append((cell, box))
                
                # Shuffle the cells (but keep the boxes in original order)
                random.shuffle([c for c, _ in cells])
                
                # Create new image and place shuffled cells
                result = img.copy()
                for (cell, box) in cells:
                    result.paste(cell, box)
                
                return result
            return img

    class PolarTransform:
        """
        Convert image to polar coordinates and apply transformations.
        Particularly effective for circular mushroom caps with radial features.
        
        Args:
            rotation: Range of rotation in polar space
            p: Probability of applying transform
        """
        def __init__(self, rotation=(-15, 15), p=0.3):
            self.rotation_range = rotation
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                # Convert to numpy array
                img_np = np.array(img)
                h, w = img_np.shape[:2]
                
                # Find center
                center_x, center_y = w // 2, h // 2
                
                # Determine maximum radius
                max_radius = int(np.sqrt(center_x**2 + center_y**2))
                
                # Create polar image canvas (radius, angle, channels)
                polar_img = np.zeros((max_radius, 360, img_np.shape[2]), dtype=np.uint8)
                
                # Convert to polar coordinates
                for r in range(max_radius):
                    for theta in range(360):
                        # Convert polar to cartesian
                        x = int(center_x + r * np.cos(np.radians(theta)))
                        y = int(center_y + r * np.sin(np.radians(theta)))
                        
                        # Check if within image bounds
                        if 0 <= x < w and 0 <= y < h:
                            polar_img[r, theta] = img_np[y, x]
                
                # Apply transformation in polar space
                # Rotate the polar image (shift in theta direction)
                rotation = random.randint(*self.rotation_range)
                polar_img = np.roll(polar_img, rotation, axis=1)
                
                # Convert back to cartesian
                result = np.zeros_like(img_np)
                for y in range(h):
                    for x in range(w):
                        # Calculate radius and angle
                        dx = x - center_x
                        dy = y - center_y
                        r = int(np.sqrt(dx**2 + dy**2))
                        theta = int(np.degrees(np.arctan2(dy, dx))) % 360
                        
                        # Check bounds
                        if r < max_radius:
                            result[y, x] = polar_img[r, theta]
                
                return Image.fromarray(result)
            return img

    class ThinPlateSpline:
        """
        Apply Thin-Plate Spline transformation for smooth non-linear warping.
        Simulates realistic shape deformations while preserving continuity.
        
        Args:
            num_control_points: Number of control points to use
            deformation_strength: Range of deformation
            p: Probability of applying transform
        """
        def __init__(self, num_control_points=5, deformation_strength=0.1, p=0.3):
            self.num_control_points = num_control_points
            self.deformation_strength = deformation_strength
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                # Convert to numpy array
                img_np = np.array(img).astype(np.uint8)
                h, w = img_np.shape[:2]
                
                # Create source control points in a grid
                n = self.num_control_points
                source_points = np.zeros((n * n, 2), dtype=np.float32)
                step_x = w // (n + 1)
                step_y = h // (n + 1)
                
                # Initialize grid points
                for i in range(n):
                    for j in range(n):
                        source_points[i * n + j, 0] = (j + 1) * step_x
                        source_points[i * n + j, 1] = (i + 1) * step_y
                
                # Create target points with small random displacements
                max_displacement = min(w, h) * self.deformation_strength
                target_points = source_points.copy()
                for i in range(n * n):
                    dx = random.uniform(-max_displacement, max_displacement)
                    dy = random.uniform(-max_displacement, max_displacement)
                    target_points[i, 0] += dx
                    target_points[i, 1] += dy
                
                # Add corner points to maintain image boundaries
                corner_points = np.array([
                    [0, 0],
                    [0, h-1],
                    [w-1, 0],
                    [w-1, h-1]
                ], dtype=np.float32)
                
                source_points = np.vstack([source_points, corner_points])
                target_points = np.vstack([target_points, corner_points])
                
                # Apply Thin-Plate Spline transformation
                tps = cv2.createThinPlateSplineShapeTransformer()
                matches = []
                for i in range(len(source_points)):
                    matches.append(cv2.DMatch(i, i, 0))
                
                source_points = source_points.reshape(1, -1, 2)
                target_points = target_points.reshape(1, -1, 2)
                
                tps.estimateTransformation(target_points, source_points, matches)
                
                # Apply transformation
                result = tps.warpImage(img_np)
                
                return Image.fromarray(result)
            return img

# Set this to True since we've implemented the transforms directly
ADVANCED_TRANSFORMS_AVAILABLE = True

# ### Enhanced Transformations ###
def get_transforms(image_size=32, aug_strength="standard"):
    """
    Standard transforms with configurable parameters for consistency.
    
    Args:
        image_size: Target image size
        aug_strength: Ignored but included for API consistency
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Now uses parameter
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0.1, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Now uses parameter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Create a wrapper class to make Albumentations compatible with our dataset
class AlbumentationsWrapper:
    """Wrapper for Albumentations transforms to make them compatible with PIL images and PyTorch."""
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, img):
        # Convert PIL Image to numpy array
        if isinstance(img, Image.Image):
            img_np = np.array(img)
            augmented = self.transform(image=img_np)
            return augmented['image']  # Returns a tensor directly
        
        # If it's already a numpy array
        elif isinstance(img, np.ndarray):
            augmented = self.transform(image=img)
            return augmented['image']
        
        # If it's already a tensor, just apply normalization and return
        elif isinstance(img, torch.Tensor):
            return img  # Not ideal, but a fallback
        
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

# Helper class for multi-scale training
class MultiScaleTransform:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list
    
    def __call__(self, img):
        # Randomly select one transformation from the list
        transform = random.choice(self.transforms_list)
        return transform(img)


# ### Enhanced Data Augmentations for Low-Resolution ###
def get_enhanced_transforms(multi_scale=False, image_size=32, pixel_percent = 0.05, crop_scale=0.9, 
                           advanced_spatial_transforms=True, mushroom_transform_params=None):
    # Default parameters for mushroom transforms
    default_params = {
        'radial_strength': 0.15,
        'radial_p': 0.3,
        'elastic_alpha': 2.0,
        'elastic_sigma': 1.5,
        'elastic_p': 0.15,
        'focus_zoom_strength': 0.2,
        'focus_zoom_p': 0.3,
        'aspect_ratio_p': 0.3,
        'grid_shuffle_p': 0.2,
        'polar_p': 0.2,
        'tps_strength': 0.05,
        'tps_p': 0.1
    }
    
    # Update with custom parameters if provided - handle both naming conventions
    if mushroom_transform_params:
        # Map between different naming conventions
        param_mapping = {
            'radial_distortion_strength': 'radial_strength',
            'radial_distortion_p': 'radial_p',
            'elastic_deform_alpha': 'elastic_alpha',
            'elastic_deform_sigma': 'elastic_sigma',
            'elastic_deform_p': 'elastic_p',
            'polar_transform_p': 'polar_p'
        }
        
        # Copy parameters, handling both naming conventions
        for key, value in mushroom_transform_params.items():
            if key in default_params:
                default_params[key] = value
            elif key in param_mapping and param_mapping[key] in default_params:
                print("Using default for", key) 
                default_params[param_mapping[key]] = value
    
    # Base transformations for low-resolution images
    base_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        
        # Add new spatial transformation for crop and zoom
        AdvancedSpatialTransforms.RandomCropAndZoom(crop_scale=crop_scale, p=0.3),
    ]
    
    # Add mushroom-specific spatial transformations if enabled
    if advanced_spatial_transforms:
        mushroom_transforms = [
            AdvancedSpatialTransforms.RadialDistortion(
                distortion_type='barrel', 
                strength=default_params['radial_strength'], 
                p=default_params['radial_p']
            ),
            AdvancedSpatialTransforms.ElasticDeformation(
                alpha=default_params['elastic_alpha'], 
                sigma=default_params['elastic_sigma'], 
                p=default_params['elastic_p']
            ),
            AdvancedSpatialTransforms.CentralFocusZoom(
                strength=default_params['focus_zoom_strength'],
                center_size=0.6, 
                p=default_params['focus_zoom_p']
            ),
            AdvancedSpatialTransforms.AspectRatioVariation(
                ratio_range=(0.95, 1.05), 
                p=default_params['aspect_ratio_p']
            ),
            AdvancedSpatialTransforms.GridShuffle(
                grid_size=2,  # Small grid size for 32x32 images
                p=default_params['grid_shuffle_p']
            ),
            AdvancedSpatialTransforms.PolarTransform(
                rotation=(-10, 10), 
                p=default_params['polar_p']
            ),
            AdvancedSpatialTransforms.ThinPlateSpline(
                num_control_points=3,
                deformation_strength=default_params['tps_strength'], 
                p=default_params['tps_p']
            )
        ]
        base_transforms.extend(mushroom_transforms)
    
    # Add color transforms
    color_transforms = [
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        AdvancedColorTransforms.RandomGrayscale(p=0.1),
        AdvancedColorTransforms.RandomColorDrop(p=0.1),
        AdvancedColorTransforms.RandomChannelSwap(p=0.1),
        AdvancedColorTransforms.RandomGamma(gamma_range=(0.7, 1.3), p=0.2),
        AdvancedColorTransforms.SimulateLightingCondition(p=0.2),
        AdvancedColorTransforms.SimulateHSVNoise(p=0.2),
        AdvancedColorTransforms.RandomPixelNoise(p=0.2, percent_pixels=pixel_percent),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    ]
    base_transforms.extend(color_transforms)
    
    # Add random erasing to simulate occlusions
    post_tensor_transforms = [
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x) if random.random() < 0.3 else x),
    ]
    
    if multi_scale:
        # For multi-scale training, we'll create transforms for different scales
        scales = [0.8, 1.0, 1.2]  # 80%, 100%, and 120% of original size
        train_transforms_list = []
        
        for scale in scales:
            scaled_size = max(8, int(image_size * scale))  # Ensure minimum size of 8
            transforms_for_scale = transforms.Compose([
                transforms.Resize((scaled_size, scaled_size)),
                *base_transforms,
                transforms.Resize((image_size, image_size)),  # Resize back to model's input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                *post_tensor_transforms
            ])
            train_transforms_list.append(transforms_for_scale)
        
        # This will randomly select one of the scales for each batch
        train_transform = MultiScaleTransform(train_transforms_list)
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            *base_transforms,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            *post_tensor_transforms
        ])
    
    # Validation transform remains simpler
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
            
# Add new function to get advanced augmentations with Albumentations
def get_albumentation_transforms(aug_strength="high", image_size=32, multi_scale=False, pixel_percent=0.05, crop_scale=0.9):
    """
    Creates advanced image augmentations using Albumentations library.
    
    Args:
        aug_strength: Strength of augmentation ('low', 'medium', 'high')
        image_size: Target image size
        multi_scale: Whether to use multi-scale transforms when falling back
        pixel_percent: Percentage of pixels to modify in noise transformations
        crop_scale: Scale factor for crop-based augmentations
    """
    if not ALBUMENTATIONS_AVAILABLE:
        print("Warning: Albumentations not available. Falling back to enhanced transforms.")
        # Pass through all parameters properly
        return get_enhanced_transforms(
            multi_scale=multi_scale, 
            image_size=image_size, 
            pixel_percent=pixel_percent, 
            crop_scale=crop_scale
        )
    
    # Configure strength parameters based on aug_strength
    if aug_strength == "low":
        p_general = 0.15
        p_optical = 0.25
        p_affine = 0.4
        distort_limit = 0.5
        p_color = 0.1
    elif aug_strength == "medium":
        p_general = 0.3
        p_optical = 0.4
        p_affine = 0.5
        distort_limit = 0.6
        p_color = 0.15
    else:  # high
        p_general = 0.5
        p_optical = 0.7
        p_affine = 0.85
        distort_limit = 0.7
        p_color = 0.2
    
    # Custom function for color channel drop (no direct equivalent in Albumentations)
    def random_channel_drop(img, **kwargs):
        if np.random.random() < p_color:
            channel = np.random.randint(0, 3)  # Choose R, G, or B channel
            img[:, :, channel] = 0
        return img
    
    # Custom function for simulating different lighting conditions
    def simulate_lighting_condition(img, **kwargs):
        if np.random.random() < p_color:
            effect = np.random.choice(['warm', 'cool', 'bright', 'dark'])
            
            if effect == 'warm':
                # Add warm/yellowish tint
                img = np.copy(img)
                # Increase red, decrease blue slightly
                img[:, :, 0] = np.clip(img[:, :, 0] * 1.1, 0, 255).astype(np.uint8)
                img[:, :, 2] = np.clip(img[:, :, 2] * 0.9, 0, 255).astype(np.uint8)
            elif effect == 'cool':
                # Add cool/bluish tint
                img = np.copy(img)
                # Decrease red, increase blue slightly
                img[:, :, 0] = np.clip(img[:, :, 0] * 0.9, 0, 255).astype(np.uint8)
                img[:, :, 2] = np.clip(img[:, :, 2] * 1.1, 0, 255).astype(np.uint8)
            elif effect == 'bright':
                # Increase brightness
                img = np.clip(img * 1.2, 0, 255).astype(np.uint8)
            else:  # dark
                # Decrease brightness
                img = np.clip(img * 0.8, 0, 255).astype(np.uint8)
        return img
    
    # Equivalent to SimulateHSVNoise from enhanced transforms
    def simulate_hsv_noise(img, **kwargs):
        if np.random.random() < p_color:
            # Convert to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Apply random shifts to each channel
            h_shift = np.random.uniform(-0.05, 0.05) * 180
            s_shift = np.random.uniform(-0.1, 0.1) * 255
            v_shift = np.random.uniform(-0.1, 0.1) * 255
            
            img_hsv[:, :, 0] = np.mod(img_hsv[:, :, 0] + h_shift, 180)
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] + s_shift, 0, 255)
            img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] + v_shift, 0, 255)
            
            # Convert back to RGB
            img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img
    
    # Function to add Gaussian noise (post-processing)
    def add_gaussian_noise(img, **kwargs):
        if np.random.random() < 0.3:  # Same probability as in enhanced transforms
            noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
            img = img + noise
            img = np.clip(img, 0, 1)
        return img
    
    # Full training transforms with both Albumentations and custom transforms
    # Fixed parameter names to match current Albumentations API
    train_transform = A.Compose([
        A.Transpose(p=p_general),
        A.HorizontalFlip(p=p_general),
        A.VerticalFlip(p=p_general),
        A.OneOf([
            # Fixed: 'limit' → 'brightness_limit' and 'contrast_limit'
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.ToGray(p=1.0),
            A.ChannelShuffle(p=1.0),
        ], p=p_color),
        A.Lambda(image=random_channel_drop, p=p_color),
        A.Lambda(image=simulate_lighting_condition, p=p_color),
        A.Lambda(image=simulate_hsv_noise, p=p_color),
        A.OneOf([
            A.MedianBlur(blur_limit=5, p=0.5),
            A.GaussianBlur(blur_limit=5, p=0.5),
            # Fixed: 'var_limit' → 'var'
            A.GaussNoise(var=30.0, p=0.5),
        ], p=p_general),
        A.OneOf([
            A.OpticalDistortion(distortion_scale=distort_limit, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=distort_limit, p=0.5),
            A.ElasticTransform(alpha=3, p=0.5),
        ], p=p_optical),
        A.CLAHE(clip_limit=4.0, p=p_general),
        # Keep ShiftScaleRotate for now even though Affine is recommended
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=p_affine),
        A.Resize(image_size, image_size),
        # Fixed: CoarseDropout parameters
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, min_height=1, min_width=1, p=p_general),
        A.Lambda(image=add_gaussian_noise, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Validation transforms - just resize and normalize
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return AlbumentationsWrapper(train_transform), AlbumentationsWrapper(val_transform)

# ### Helper class for applying transforms to subsets ###
class CustomSubset(Dataset):
    """Custom subset that applies transform to the original dataset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
    
    @property
    def classes(self):
        """Access classes attribute from the underlying dataset."""
        if hasattr(self.subset, 'dataset') and hasattr(self.subset.dataset, 'classes'):
            return self.subset.dataset.classes
        elif hasattr(self.subset, 'classes'):
            return self.subset.classes
        raise AttributeError("Could not find 'classes' attribute in the underlying dataset")
    
    @property
    def dataset(self):
        """Access the original dataset for compatibility."""
        if hasattr(self.subset, 'dataset'):
            return self.subset.dataset
        return self.subset

class KMeansColorAugmentation:
    """
    K-Means Clustering color augmentation that quantizes the image's color space.
    
    This augmentation:
    1. Applies K-Means clustering to find k representative colors in the image
    2. Replaces each pixel with its nearest cluster center
    3. Can blend with the original image for subtle effects
    
    Args:
        k: Number of color clusters (default: 8)
        p: Probability of applying the transform (default: 1.0)
        blend_factor: How much to blend with original image (0.0-1.0, default: 1.0)
                     0.0 = original image, 1.0 = fully clustered image
        max_iter: Maximum K-means iterations (default: 10)
    """
    def __init__(self, k=8, p=1.0, blend_factor=1.0, max_iter=10):
        self.k = k
        self.p = p
        self.blend_factor = blend_factor
        self.max_iter = max_iter
        
    def __call__(self, img):
        if random.random() < self.p:
            # Convert PIL Image to numpy array
            img_array = np.array(img)
            h, w, c = img_array.shape
            
            # Reshape to a list of pixels (Nx3)
            pixels = img_array.reshape(-1, c).astype(np.float32)
            
            # Apply K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iter, 1.0)
            _, labels, centers = cv2.kmeans(pixels, self.k, None, criteria, 
                                             attempts=3, flags=cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to uint8
            centers = np.uint8(centers)
            
            # Map each pixel to its corresponding cluster center
            quantized = centers[labels.flatten()]
            
            # Reshape back to original image dimensions
            quantized_img = quantized.reshape(img_array.shape)
            
            # Blend with original image if blend_factor < 1.0
            if self.blend_factor < 1.0:
                blended_img = cv2.addWeighted(
                    img_array, 1.0 - self.blend_factor, 
                    quantized_img, self.blend_factor, 0
                )
                result_img = blended_img
            else:
                result_img = quantized_img
                
            # Convert back to PIL Image
            return Image.fromarray(result_img)
            
        return img

# Create a standalone transform that applies only KMeans color augmentation
def get_kmeans_transform(image_size=32, k=8, blend_factor=1.0):
    """
    Creates a transform pipeline that only applies KMeans color augmentation.
    
    Args:
        image_size: Target image size
        k: Number of color clusters
        blend_factor: How much to blend with original image (0.0-1.0)
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        KMeansColorAugmentation(k=k, p=1.0, blend_factor=blend_factor),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Simple validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Datasets

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.430481Z","iopub.execute_input":"2025-04-19T04:08:43.430690Z","iopub.status.idle":"2025-04-19T04:08:43.438150Z","shell.execute_reply.started":"2025-04-19T04:08:43.430673Z","shell.execute_reply":"2025-04-19T04:08:43.437587Z"}}
class MushroomDataset(Dataset):
    """Dataset class for loading images with robust error handling."""
    def __init__(self, csv_file, transform=None):
        # Validate CSV file
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file does not exist: {csv_file}")
        self.data = pd.read_csv(csv_file)
        
        # Check required columns and empty data
        required_columns = ['image_path', 'class_name']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError("CSV must contain 'image_path' and 'class_name' columns")
        if len(self.data) == 0:
            raise ValueError("CSV file is empty")

        self.transform = transform
        self.classes = sorted(self.data['class_name'].unique())
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.error_count = 0  # Track error count
        print(f"Loaded dataset with {len(self.data)} samples and {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Try loading with retries
        max_retries = 1
        for attempt in range(max_retries):
            try:
                # Get image path with correct platform handling
                current_idx = (idx + attempt) % len(self.data)
                img_path = str(Path(self.data.iloc[current_idx]['image_path'])).replace('\\', '/')
                
                # Load image and label
                image = Image.open(img_path).convert('RGB')
                label = self.class_to_idx[self.data.iloc[current_idx]['class_name']]
                
                # Apply transformations
                if self.transform:
                    if isinstance(self.transform, AlbumentationsWrapper):
                        image = np.array(image)
                    image = self.transform(image)
                
                return image, label
                
            except Exception as e:
                self.error_count += 1
                print(f"Error loading image at index {current_idx}: {str(e)}")
        
        # If all retries failed, use a placeholder
        print(f"All {max_retries} attempts to load valid image failed, starting at idx {idx}")
        # Create a recognizable pattern with correct image size
        placeholder = torch.ones((3, 32, 32)) * 0.1 if self.transform else Image.new('RGB', (32, 32), color=(25, 25, 25))
        # Use a random valid label to avoid biasing the model
        random_label = random.randint(0, len(self.classes) - 1)
        return placeholder, random_label

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Utils

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.438695Z","iopub.execute_input":"2025-04-19T04:08:43.438878Z","iopub.status.idle":"2025-04-19T04:08:43.455440Z","shell.execute_reply.started":"2025-04-19T04:08:43.438862Z","shell.execute_reply":"2025-04-19T04:08:43.454875Z"}}
# Update main function to use enhanced features but skip distillation
def format_time(seconds):
    """Format seconds into hours, minutes, seconds string."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

def analyze_false_predictions(true_labels, predicted_labels, class_names):
    """
    Analyze and report false predictions per class.
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Model predictions
        class_names: List of class names
        
    Returns:
        Dictionary with false prediction statistics
    """
    try:
        # Convert numpy arrays to lists if needed
        if hasattr(true_labels, 'tolist'):
            true_labels = true_labels.tolist()
        if hasattr(predicted_labels, 'tolist'):
            predicted_labels = predicted_labels.tolist()
        
        # Create confusion matrix - avoiding sklearn import inside function
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Calculate false predictions per class
        false_pred_per_class = {}
        
        # For each true class, count misclassifications
        for true_idx, class_name in enumerate(class_names):
            try:
                # Total samples of this class
                total = int(np.sum(cm[true_idx, :]))
                # Correct predictions (true positives)
                correct = int(cm[true_idx, true_idx])
                # False predictions (should be this class but predicted as something else)
                false = int(total - correct)
                
                # Store the statistics
                false_pred_per_class[class_name] = {
                    'total': int(total),
                    'correct': int(correct),
                    'false': int(false),
                    'accuracy': float(correct / total) if total > 0 else 0.0
                }
                
                # Store which classes this class was confused with
                confused_with = {}
                for pred_idx, pred_class in enumerate(class_names):
                    if pred_idx != true_idx and cm[true_idx, pred_idx] > 0:
                        confused_with[pred_class] = int(cm[true_idx, pred_idx])
                
                false_pred_per_class[class_name]['confused_with'] = confused_with
            except IndexError:
                print(f"WARNING: CLASS INDEX OUT OF RANGE FOR {class_name}. SKIPPING.")
                continue
            except Exception as e:
                print(f"ERROR ANALYZING CLASS {class_name}: {str(e)}")
                continue
        
        # Calculate overall statistics
        total_samples = len(true_labels)
        total_correct = sum(1 for i in range(len(true_labels)) if true_labels[i] == predicted_labels[i])
        overall_accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0
        
        result = {
            'per_class': false_pred_per_class,
            'overall': {
                'total_samples': int(total_samples),
                'correct_predictions': int(total_correct),
                'false_predictions': int(total_samples - total_correct),
                'accuracy': float(overall_accuracy)
            }
        }
        
        return result
    except Exception as e:
        print(f"CRITICAL ERROR IN ANALYZE_FALSE_PREDICTIONS: {str(e)}")
        # Return a minimal valid structure as fallback
        return {
            'per_class': {},
            'overall': {
                'total_samples': len(true_labels) if hasattr(true_labels, '__len__') else 0,
                'correct_predictions': 0,
                'false_predictions': 0,
                'accuracy': 0.0,
                'error': str(e)
            }
        }

def print_false_prediction_report(analysis):
    """Print a human-readable report of false prediction analysis."""
    print("\n=== False Prediction Analysis ===")
    print(f"Overall Accuracy: {analysis['overall']['accuracy']:.4f}")
    print(f"Total Samples: {analysis['overall']['total_samples']}")
    print(f"Correct Predictions: {analysis['overall']['correct_predictions']}")
    print(f"False Predictions: {analysis['overall']['false_predictions']}")
    
    print("\nPer-Class Analysis (sorted by accuracy):")
    
    # Sort classes by accuracy (ascending)
    sorted_classes = sorted(analysis['per_class'].items(), 
                           key=lambda x: x[1]['accuracy'])
    
    for class_name, stats in sorted_classes:
        print(f"\n  Class: {class_name}")
        print(f"    Accuracy: {stats['accuracy']:.4f}")
        print(f"    Total samples: {stats['total']}")
        print(f"    False predictions: {stats['false']}")
        
        if stats['confused_with']:
            print("    Confused with:")
            sorted_confusions = sorted(stats['confused_with'].items(), 
                                     key=lambda x: x[1], reverse=True)
            for confused_class, count in sorted_confusions:
                print(f"      - {confused_class}: {count}")

# Add this helper function to ensure all objects in a dictionary are JSON serializable
def convert_to_json_serializable(obj):
    """Recursively convert a nested dictionary/list with numpy types to Python standard types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # PolyLoss

# %% [code] {"execution":{"iopub.status.busy":"2025-04-19T04:08:43.455963Z","iopub.execute_input":"2025-04-19T04:08:43.456156Z","iopub.status.idle":"2025-04-19T04:08:43.467576Z","shell.execute_reply.started":"2025-04-19T04:08:43.456140Z","shell.execute_reply":"2025-04-19T04:08:43.466987Z"},"jupyter":{"source_hidden":true}}
def to_one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


class PolyLoss(_Loss):
    def __init__(self,
                 softmax: bool = True,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: if target is in one-hot format, its shape should be BNH[WD],
                if it is not one-hot encoded, it should has shape B1H[WD] or BH[WD], where N is the number of classes, 
                It should contain binary values

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        if len(input.shape) - len(target.shape) == 1:
            target = target.unsqueeze(1).long()
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        # target not in one-hot encode format, has shape B1H[WD]
        if n_pred_ch != n_target_ch:
            # squeeze out the channel dimension of size 1 to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.squeeze(target, dim=1).long())
            # convert into one-hot format to calculate ce loss
            target = to_one_hot(target, num_classes=n_pred_ch)
        else:
            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.argmax(target, dim=1))

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        pt = (input * target).sum(dim=1)  # BH[WD]
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD] 
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return polyl


def create_poly_loss(ce_weight=None, epsilon=1.0, reduction='mean'):
    """
    Create a PolyLoss instance with the specified parameters.
    
    Args:
        ce_weight: Optional class weights for cross-entropy component
        epsilon: Weight of the PT term (default: 1.0)
        reduction: Reduction method, one of ["mean", "sum", "none"]
        
    Returns:
        PolyLoss instance
    """
    return PolyLoss(
        softmax=True,
        ce_weight=ce_weight,
        reduction=reduction,
        epsilon=epsilon
    )

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Models

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## SmallResnet

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.468133Z","iopub.execute_input":"2025-04-19T04:08:43.468316Z","iopub.status.idle":"2025-04-19T04:08:43.478899Z","shell.execute_reply.started":"2025-04-19T04:08:43.468299Z","shell.execute_reply":"2025-04-19T04:08:43.478317Z"}}
# ### Model Definition ###
class ResidualBlock(nn.Module):
    """Basic residual block with two 7x7 convolutions and a skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolution and batch norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, 
                              stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution and batch norm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=7, 
                              stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (if dimensions change)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Store input for skip connection
        identity = x
        
        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolution block
        out = self.conv2(out)  # BUG FIX: was using x instead of out!
        out = self.bn2(out)
        
        # Skip connection
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class SmallResNet(nn.Module):
    """Custom ResNet architecture for small 32x32 images with 7x7 kernels."""
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(SmallResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks with increasing channels
        self.res_block1 = ResidualBlock(32, 64, stride=2)  # 16x16
        self.res_block2 = ResidualBlock(64, 64, stride=1)
        self.res_block3 = ResidualBlock(64, 128, stride=2)  # 8x8
        self.res_block4 = ResidualBlock(128, 128, stride=1)
        self.res_block5 = ResidualBlock(128, 256, stride=2)  # 4x4
        self.res_block6 = ResidualBlock(256, 256, stride=1)
        
        # Global average pooling and fully connected layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## DenseNet

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.479561Z","iopub.execute_input":"2025-04-19T04:08:43.479747Z","iopub.status.idle":"2025-04-19T04:08:43.494516Z","shell.execute_reply.started":"2025-04-19T04:08:43.479730Z","shell.execute_reply":"2025-04-19T04:08:43.493960Z"}}
# ### DenseNet with 7x7 Kernels Implementation ###
class DenseLayer(nn.Module):
    """Single layer in a DenseNet block with larger 7x7 kernels."""
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        # BN-ReLU-Conv structure
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)  # Changed from inplace=True
        # Using 7x7 kernels instead of standard 3x3
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=7, 
                             stride=1, padding=3, bias=False)
        
    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)  # This uses the non-inplace ReLU
        out = self.conv(out)
        return torch.cat([x, out], 1)  # Dense connection

class DenseBlock(nn.Module):
    """Block containing multiple densely connected layers with 7x7 kernels."""
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    """Transition layer between DenseBlocks without spatial reduction for small images."""
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)  # Changed from inplace=True
        # Use 1x1 conv to reduce channels but keep spatial dimensions
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)  # This uses the non-inplace ReLU
        x = self.conv(x)
        return x

class DenseNet7x7(nn.Module):
    """DenseNet implementation with 7x7 kernels and no spatial reduction for small 32x32 images."""
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_classes=10, dropout_rate=0.2):
        super(DenseNet7x7, self).__init__()
        
        # Initial convolution without spatial reduction
        self.features = nn.Sequential(
            # Use stride=1 instead of stride=2 to maintain spatial dimensions
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),  # Changed from inplace=True
        )
        
        # Current number of channels
        num_channels = 64
        
        # Add dense blocks and transition layers
        for i, num_layers in enumerate(block_config):
            # Add dense block
            block = DenseBlock(num_channels, growth_rate, num_layers)
            self.features.add_module(f'denseblock{i+1}', block)
            
            # Update number of channels after dense block
            num_channels += num_layers * growth_rate
            
            # Add transition layer except after the last block
            if i != len(block_config) - 1:
                # Reduce number of channels by half in transition
                trans = TransitionLayer(num_channels, num_channels // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_channels = num_channels // 2
        
        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_channels))
        self.features.add_module('relu_final', nn.ReLU(inplace=False))  # Changed from inplace=True
        
        # Keep the global average pooling - still needed for classification
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_channels, num_classes)
        
        # Initialize weights - same as before
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Added proper weight initialization
                if m.bias is not None:  # Check if bias exists before initializing
                    nn.init.constant_(m.bias, 0)
    
    # Forward method remains the same
    def forward(self, x):
        features = self.features(x)
        out = self.avg_pool(features)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Dual branch

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.495103Z","iopub.execute_input":"2025-04-19T04:08:43.495287Z","iopub.status.idle":"2025-04-19T04:08:43.509697Z","shell.execute_reply.started":"2025-04-19T04:08:43.495271Z","shell.execute_reply":"2025-04-19T04:08:43.509128Z"}}
class SpaceToDepthConv(nn.Module):
    """
    Space-to-Depth Convolution that rearranges spatial information into channel dimension
    instead of losing it through downsampling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, block_size=2, stride=1, padding=1):
        super(SpaceToDepthConv, self).__init__()
        self.block_size = block_size
        # Regular convolution
        self.conv = nn.Conv2d(
            in_channels * (block_size ** 2),  # Input channels increased by block_size^2
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Get dimensions
        batch_size, channels, height, width = x.size()
        
        # Check if dimensions are compatible with block_size
        if height % self.block_size != 0 or width % self.block_size != 0:
            # Add padding if needed
            pad_h = self.block_size - (height % self.block_size)
            pad_w = self.block_size - (width % self.block_size)
            if pad_h < self.block_size or pad_w < self.block_size:
                x = nn.functional.pad(x, (0, pad_w if pad_w < self.block_size else 0, 
                                         0, pad_h if pad_h < self.block_size else 0))
                # Update dimensions after padding
                batch_size, channels, height, width = x.size()
                
        # Space-to-depth transformation: rearrange spatial dims into channel dim
        x = x.view(
            batch_size,
            channels,
            height // self.block_size,
            self.block_size,
            width // self.block_size,
            self.block_size
        )
        # Permute and reshape to get all spatial blocks as channels
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch_size, channels * (self.block_size ** 2), 
                   height // self.block_size, width // self.block_size)
        
        # Apply convolution
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

# ### Dual-Branch Network Architecture ###
class DualBranchNetwork(nn.Module):
    """
    Dual-Branch Network with one branch focusing on global features
    and another branch focusing on local details, with a common feature subspace.
    """
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(DualBranchNetwork, self).__init__()
        
        # Branch 1: Global feature extraction with larger kernels
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 64, stride=2),  # 16x16
            ResidualBlock(64, 128, stride=2),  # 8x8
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Branch 2: Local feature extraction with SPD-Conv to preserve information
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SpaceToDepthConv(32, 64, block_size=2),  # 16x16
            SpaceToDepthConv(64, 128, block_size=2),  # 8x8
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Common feature subspace mapping
        self.feature_fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Use xavier_uniform for linear layers - better for fusion networks
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Process through both branches
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        
        # Concatenate branch outputs
        x_concat = torch.cat((x1, x2), dim=1)
        
        # Map to common feature subspace
        features = self.feature_fusion(x_concat)
        
        # Classification
        out = self.classifier(features)
        
        return out

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## SPD Dual Branch

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.511921Z","iopub.execute_input":"2025-04-19T04:08:43.512132Z","iopub.status.idle":"2025-04-19T04:08:43.527731Z","shell.execute_reply.started":"2025-04-19T04:08:43.512115Z","shell.execute_reply":"2025-04-19T04:08:43.527160Z"}}
class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution that rearranges spatial information into channel dimension
    instead of losing it through downsampling.
    Used specifically for downsampling operations.
    """
    def __init__(self, in_channels, out_channels, scale=2, kernel_size=3, padding=1):
        super(SPDConv, self).__init__()
        self.scale = scale
        # Convolution layer: input channels are scaled by scale^2 due to space-to-depth
        self.conv = nn.Conv2d(in_channels * scale**2, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Space-to-depth: rearranges spatial data into channels
        batch_size, channels, height, width = x.size()
        
        # Check if dimensions are compatible with scale factor
        if height % self.scale != 0 or width % self.scale != 0:
            # Add padding if needed
            pad_h = self.scale - (height % self.scale)
            pad_w = self.scale - (width % self.scale)
            if pad_h < self.scale or pad_w < self.scale:
                x = F.pad(x, (0, pad_w if pad_w < self.scale else 0, 
                             0, pad_h if pad_h < self.scale else 0))
                # Update dimensions after padding
                batch_size, channels, height, width = x.size()
                
        # Space-to-depth transformation: rearrange spatial dims into channel dim
        x = x.view(
            batch_size,
            channels,
            height // self.scale,
            self.scale,
            width // self.scale,
            self.scale
        )
        # Permute and reshape to get all spatial blocks as channels
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch_size, channels * (self.scale ** 2), 
                  height // self.scale, width // self.scale)
        
        # Apply convolution
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class ConvBlock(nn.Module):
    """Standard convolutional block for feature extraction without downsampling."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SPDDualBranchNetwork(nn.Module):
    """
    Dual-Branch Network where both branches use SPDConv exclusively for downsampling operations.
    
    - Branch 1: Focuses on global features using larger kernels
    - Branch 2: Focuses on local details using smaller kernels
    - Both branches use SPDConv for downsampling to preserve spatial information
    """
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(SPDDualBranchNetwork, self).__init__()
        
        # Branch 1: Global feature extraction with larger kernels
        self.branch1_init = ConvBlock(3, 32, kernel_size=7, padding=3)
        self.branch1_down1 = SPDConv(32, 64, scale=2)  # 32x32 -> 16x16
        self.branch1_block1 = ConvBlock(64, 64, kernel_size=5, padding=2)
        self.branch1_down2 = SPDConv(64, 128, scale=2)  # 16x16 -> 8x8
        self.branch1_block2 = ConvBlock(128, 128, kernel_size=5, padding=2)
        
        # Branch 2: Local feature extraction with smaller kernels
        self.branch2_init = ConvBlock(3, 32, kernel_size=3, padding=1)
        self.branch2_down1 = SPDConv(32, 64, scale=2)  # 32x32 -> 16x16
        self.branch2_block1 = ConvBlock(64, 64, kernel_size=3, padding=1)
        self.branch2_down2 = SPDConv(64, 128, scale=2)  # 16x16 -> 8x8
        self.branch2_block2 = ConvBlock(128, 128, kernel_size=3, padding=1)
        
        # Additional processing for both branches
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers for each branch
        self.branch1_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.branch2_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Feature fusion and classifier
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Branch 1: Global feature path
        b1 = self.branch1_init(x)
        b1 = self.branch1_down1(b1)  # SPDConv downsampling
        b1 = self.branch1_block1(b1)
        b1 = self.branch1_down2(b1)  # SPDConv downsampling
        b1 = self.branch1_block2(b1)
        b1 = self.avgpool(b1)
        b1 = self.branch1_fc(b1)
        
        # Branch 2: Local detail path
        b2 = self.branch2_init(x)
        b2 = self.branch2_down1(b2)  # SPDConv downsampling
        b2 = self.branch2_block1(b2)
        b2 = self.branch2_down2(b2)  # SPDConv downsampling
        b2 = self.branch2_block2(b2)
        b2 = self.avgpool(b2)
        b2 = self.branch2_fc(b2)
        
        # Concatenate features from both branches
        combined = torch.cat([b1, b2], dim=1)
        
        # Feature fusion
        fused = self.fusion(combined)
        
        # Classification
        out = self.classifier(fused)
        
        return out

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Dilated Group

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.528971Z","iopub.execute_input":"2025-04-19T04:08:43.529169Z","iopub.status.idle":"2025-04-19T04:08:43.546471Z","shell.execute_reply.started":"2025-04-19T04:08:43.529152Z","shell.execute_reply":"2025-04-19T04:08:43.545915Z"}}
# ### Dilated Group Convolution Network ###
class DilatedGroupConvBlock(nn.Module):
    """
    Custom block implementing dilated group convolutions with 7x7 kernels.
    
    Features:
    - Uses grouped convolutions to reduce parameter count
    - Employs dilated convolutions to increase receptive field without pooling
    - Maintains spatial information with residual connections
    """
    def __init__(self, in_channels, out_channels, dilation=1, stride=1, groups=4, reduction_ratio=4):
        super(DilatedGroupConvBlock, self).__init__()
        
        # Calculate reduced dimensions for bottleneck
        reduced_channels = max(out_channels // reduction_ratio, 8)
        
        # Ensure in_channels and out_channels are divisible by groups
        assert in_channels % groups == 0, f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        assert out_channels % groups == 0, f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        
        # Input projection with 1x1 convolution (no dilation needed here)
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dilated grouped convolution with 7x7 kernel
        self.grouped_conv = nn.Sequential(
            nn.Conv2d(
                reduced_channels, 
                reduced_channels, 
                kernel_size=7, 
                stride=1,  # Changed from stride parameter to always 1
                padding=3 * dilation,  # Maintain spatial size with proper padding
                dilation=dilation,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(reduced_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection (residual path)
        self.skip = nn.Identity()
        
        # Use SPDConv for downsampling in skip connection if stride > 1
        if stride > 1:
            self.skip = nn.Sequential(
                SPDConv(in_channels, out_channels, kernel_size=1, scale=stride, padding=0)
            )
        elif in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        # Add SPDConv for downsampling in main path if stride > 1
        self.downsample = None
        if stride > 1:
            self.downsample = SPDConv(
                reduced_channels,
                reduced_channels,
                kernel_size=3,
                scale=stride,
                padding=1
            )
            
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        # Main branch
        out = self.input_proj(x)
        out = self.grouped_conv(out)
        
        # Apply downsampling if needed
        if self.downsample is not None:
            out = self.downsample(out)
            
        out = self.output_proj(out)
        
        # Residual connection
        out += self.skip(identity)
        out = self.relu(out)
        
        return out


class DilatedGroupConvNet(nn.Module):
    """
    Neural network using dilated group convolutions with 7x7 kernels.
    
    This architecture:
    1. Uses dilated convolutions to capture wide context without pooling
    2. Employs group convolutions to reduce parameter count
    3. Replaces pooling operations with SPDConv for downsampling
    4. Maintains spatial information flow via residual connections
    """
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(DilatedGroupConvNet, self).__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: Regular and dilated convolutions at same scale (32x32)
        self.stage1 = nn.Sequential(
            DilatedGroupConvBlock(32, 64, dilation=1, stride=1),
            DilatedGroupConvBlock(64, 64, dilation=2, stride=1),
            DilatedGroupConvBlock(64, 64, dilation=4, stride=1)
        )
        
        # Transition 1: SPDConv instead of strided convolution (32x32 → 16x16)
        self.transition1 = DilatedGroupConvBlock(64, 128, dilation=1, stride=2)
        
        # Stage 2: Medium-scale features (16x16)
        self.stage2 = nn.Sequential(
            DilatedGroupConvBlock(128, 128, dilation=1, stride=1),
            DilatedGroupConvBlock(128, 128, dilation=2, stride=1),
            DilatedGroupConvBlock(128, 128, dilation=4, stride=1)
        )
        
        # Transition 2: SPDConv instead of strided convolution (16x16 → 8x8)
        self.transition2 = DilatedGroupConvBlock(128, 256, dilation=1, stride=2)
        
        # Stage 3: Deep features with increased dilation (8x8)
        self.stage3 = nn.Sequential(
            DilatedGroupConvBlock(256, 256, dilation=1, stride=1),
            DilatedGroupConvBlock(256, 256, dilation=2, stride=1)
        )
        
        # Global feature extraction using SPDConv instead of strided convolutions
        self.global_features = nn.Sequential(
            SPDConv(256, 512, kernel_size=3, scale=2, padding=1),  # 8x8 → 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SPDConv(512, 512, kernel_size=3, scale=2, padding=1),  # 4x4 → 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SPDConv(512, 512, kernel_size=2, scale=2, padding=0),  # 2x2 → 1x1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.stem(x)
        
        # Feature extraction stages with dilated convolutions
        x = self.stage1(x)
        x = self.transition1(x)
        x = self.stage2(x)
        x = self.transition2(x)
        x = self.stage3(x)
        
        # Global feature extraction without pooling
        x = self.global_features(x)
        
        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ##  InceptionFSD

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.547079Z","iopub.execute_input":"2025-04-19T04:08:43.547268Z","iopub.status.idle":"2025-04-19T04:08:43.568337Z","shell.execute_reply.started":"2025-04-19T04:08:43.547251Z","shell.execute_reply":"2025-04-19T04:08:43.567775Z"}}
# Add new FSDDownsample module
class FSDDownsample(nn.Module):
    """
    Feature Scale Detection (FSD) Downsampling Module.
    Performs feature-aware downsampling by learning the most important features
    to preserve while reducing spatial dimensions.
    """
    def __init__(self, in_channels, out_channels):
        super(FSDDownsample, self).__init__()
        
        # Parallel pathways with different receptive fields
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Process through parallel branches that preserve different aspects of features
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)
        
        # Concatenate to combine all feature aspects
        return torch.cat([branch1_output, branch2_output, branch3_output, branch4_output], dim=1)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise feature recalibration.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InceptionModule(nn.Module):
    """
    Inception module with parallel pathways for multi-scale feature extraction.
    Enhanced with additional channels and optional SE attention.
    """
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, use_se=True):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 conv -> 5x5 conv branch (implemented as two 3x3 convs for efficiency)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5red, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
        
        # Max pooling -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
        
        # Optional squeeze-and-excitation attention
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(ch1x1 + ch3x3 + ch5x5 + pool_proj)
    
    def forward(self, x):
        # Process input through each branch
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)
        
        # Concatenate outputs along the channel dimension
        outputs = [branch1_output, branch2_output, branch3_output, branch4_output]
        combined = torch.cat(outputs, 1)
        
        # Apply channel attention if enabled
        if self.use_se:
            combined = self.se(combined)
            
        return combined

class InceptionFSD(nn.Module):
    """
    Enhanced Inception-based Feature Scale Detector with increased capacity.
    
    This model:
    1. Uses deeper Inception modules with more channels
    2. Adds squeeze-and-excitation attention for feature refinement
    3. Uses advanced multi-scale feature aggregation
    4. Enhanced feature fusion network
    """
    def __init__(self, num_classes=10, dropout_rate=0.4):
        super(InceptionFSD, self).__init__()
        
        # Initial feature extraction - maintain 32 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # First inception block - maintain spatial dimensions for small images
        self.inception1 = InceptionModule(
            in_channels=32,
            ch1x1=16,
            ch3x3red=24,
            ch3x3=32,
            ch5x5red=8,
            ch5x5=16,
            pool_proj=16,
            use_se=True
        )  # Output: 80 channels (16+32+16+16)
        
        # Additional inception block at the same scale for more capacity
        self.inception1b = InceptionModule(
            in_channels=80,
            ch1x1=24,
            ch3x3red=32,
            ch3x3=48,
            ch5x5red=12,
            ch5x5=24,
            pool_proj=20,
            use_se=True
        )  # Output: 116 channels (24+48+24+20)
        
        # Reduction block 1 - use FSD downsampling
        self.reduction1 = FSDDownsample(116, 144)  # 116 -> 144 channels, spatial dim: 32x32 -> 16x16
        
        # Second inception block
        self.inception2 = InceptionModule(
            in_channels=144,
            ch1x1=40,
            ch3x3red=48,
            ch3x3=64,
            ch5x5red=16,
            ch5x5=32,
            pool_proj=32,
            use_se=True
        )  # Output: 168 channels (40+64+32+32)
        
        # Reduction block 2 - use FSD downsampling
        self.reduction2 = FSDDownsample(168, 192)  # 168 -> 192 channels, spatial dim: 16x16 -> 8x8
        
        # Third inception block
        self.inception3 = InceptionModule(
            in_channels=192,
            ch1x1=64,
            ch3x3red=64,
            ch3x3=96,
            ch5x5red=24,
            ch5x5=48,
            pool_proj=48,
            use_se=True
        )  # Output: 256 channels (64+96+48+48)
        
        # Multi-scale feature aggregation with additional multi-resolution pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global context
        self.mid_pool = nn.AdaptiveAvgPool2d((2, 2))     # Mid-level context
        self.local_pool = nn.AdaptiveAvgPool2d((4, 4))   # Local context
        
        # Calculate flattened feature dimensions
        global_features = 256  # From inception3
        mid_features = 256 * 4  # 2x2 spatial dimension
        local_features = 168 * 16  # 4x4 spatial dimension
        
        # Enhanced feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(global_features + mid_features + local_features, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(768, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2)  # Reduced dropout in final fusion layer
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:  # Add check for None bias
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:  # Add check for None bias
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)
        
        # Feature extraction at different scales
        inception1_out = self.inception1(x)
        inception1b_out = self.inception1b(inception1_out)
        
        x = self.reduction1(inception1b_out)
        inception2_out = self.inception2(x)
        
        x = self.reduction2(inception2_out)
        inception3_out = self.inception3(x)
        
        # Multi-scale feature aggregation
        global_features = self.global_pool(inception3_out)
        mid_features = self.mid_pool(inception3_out)
        local_features = self.local_pool(inception2_out)
        
        # Flatten features
        global_features = torch.flatten(global_features, 1)
        mid_features = torch.flatten(mid_features, 1)
        local_features = torch.flatten(local_features, 1)
        
        # Concatenate multi-scale features
        concat_features = torch.cat([global_features, mid_features, local_features], dim=1)
        
        # Feature fusion
        fused_features = self.fusion(concat_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## SPD Resnet

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.568869Z","iopub.execute_input":"2025-04-19T04:08:43.569063Z","iopub.status.idle":"2025-04-19T04:08:43.586762Z","shell.execute_reply.started":"2025-04-19T04:08:43.569039Z","shell.execute_reply":"2025-04-19T04:08:43.586186Z"}}
class SPDBasicBlock(nn.Module):
    """
    Basic ResNet block that uses Space-to-Depth for downsampling instead of strided convolutions.
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, downsample=False):
        super(SPDBasicBlock, self).__init__()
        
        # First convolution
        if downsample:
            # Replace stride-2 conv with Space-to-Depth followed by 1x1 conv
            self.conv1 = SpaceToDepthConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                block_size=2,  # Equivalent to stride=2
                padding=1
            )
        else:
            # Regular convolution for non-downsampling blocks
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        # Second convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection/shortcut
        if downsample or in_channels != out_channels:
            if downsample:
                # Use Space-to-Depth for shortcut when downsampling
                self.shortcut = nn.Sequential(
                    nn.PixelUnshuffle(2),  # This is essentially a space-to-depth operation
                    nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                # 1x1 conv for channel matching without downsampling
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class SPDBottleneck(nn.Module):
    """
    Bottleneck ResNet block that uses Space-to-Depth for downsampling instead of strided convolutions.
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, downsample=False):
        super(SPDBottleneck, self).__init__()
        bottleneck_channels = out_channels // self.expansion
        
        # First 1x1 bottleneck convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        
        # Middle 3x3 convolution
        if downsample:
            # Replace stride-2 conv with Space-to-Depth followed by 3x3 conv
            self.conv2 = SpaceToDepthConv(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                kernel_size=3,
                block_size=2,  # Equivalent to stride=2
                padding=1
            )
        else:
            # Regular 3x3 convolution for non-downsampling blocks
            self.conv2 = nn.Sequential(
                nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(inplace=True)
            )
        
        # Final 1x1 expansion convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection/shortcut
        if downsample or in_channels != out_channels:
            if downsample:
                # Use Space-to-Depth for shortcut when downsampling
                self.shortcut = nn.Sequential(
                    nn.PixelUnshuffle(2),  # Space-to-depth operation
                    nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                # 1x1 conv for channel matching without downsampling
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class SPDResNet(nn.Module):
    """
    ResNet architecture that uses Space-to-Depth operations instead of strided convolutions
    for all downsampling operations, preserving spatial information.
    """
    def __init__(self, block, layers, num_classes=10, dropout_rate=0.2):
        super(SPDResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Replace initial pooling with SPD for more information preservation
        self.spd_initial = SpaceToDepthConv(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            block_size=2,  # Equivalent to stride=2 pooling
            padding=1
        )
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], downsample=True)
        self.layer3 = self._make_layer(block, 256, layers[2], downsample=True)
        self.layer4 = self._make_layer(block, 512, layers[3], downsample=True)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, downsample=False):
        layers = []
        # First block may perform downsampling
        layers.append(block(self.in_channels, out_channels, downsample))
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels
        
        # Add remaining blocks (no downsampling)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial processing
        x = self.initial(x)
        x = self.spd_initial(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Pre-defined model configurations
def spdresnet18(num_classes=10, dropout_rate=0.2):
    return SPDResNet(SPDBasicBlock, [2, 2, 2, 2], num_classes=num_classes, dropout_rate=dropout_rate)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Adapted LowNet

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.587359Z","iopub.execute_input":"2025-04-19T04:08:43.587568Z","iopub.status.idle":"2025-04-19T04:08:43.601987Z","shell.execute_reply.started":"2025-04-19T04:08:43.587550Z","shell.execute_reply":"2025-04-19T04:08:43.601427Z"}}
# Custom ReLU with variable slopes
class VariableReLU(nn.Module):
    def __init__(self, slope):
        super(VariableReLU, self).__init__()
        self.slope = slope

    def forward(self, x):
        return torch.max(torch.zeros_like(x), x * self.slope)

class AdaptedLowNet(nn.Module):
    """
    Adaptation of LowNet for main.py framework.
    
    Modifications:
    - Supports RGB input (3 channels) instead of grayscale (1 channel)
    - Allows configurable number of output classes
    - Adds configurable dropout rate
    - Uses padding to maintain spatial dimensions compatibility
    - Uses SPDConv for downsampling instead of pooling to preserve information
    """
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(AdaptedLowNet, self).__init__()
        
        # Low-Resolution Feature Extractor (3 Conv Layers)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = VariableReLU(slope=4)  # Slope = 4 for Layer 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = VariableReLU(slope=2)  # Slope = 2 for Layer 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = VariableReLU(slope=1)  # Slope = 1 for Layer 3
        self.dropout_conv = nn.Dropout(p=dropout_rate)  # Configurable dropout rate
        
        # Replace pooling with SPDConv for downsampling (32x32 -> 8x8)
        # This uses two consecutive SPDConv operations, equivalent to a scale=4 downsampling
        self.downsample = nn.Sequential(
            SPDConv(in_channels=128, out_channels=128, kernel_size=3, scale=2, padding=1),  # 32x32 -> 16x16
            SPDConv(in_channels=128, out_channels=128, kernel_size=3, scale=2, padding=1)   # 16x16 -> 8x8
        )
        
        # Classifier (3 Fully-Connected Layers)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout_fc2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)  # Configurable output classes
        
    def forward(self, x):
        # Feature Extractor
        x = self.relu1(self.conv1(x))    # Conv1 + ReLU(slope=4)
        x = self.relu2(self.conv2(x))    # Conv2 + ReLU(slope=2)
        x = self.relu3(self.conv3(x))    # Conv3 + ReLU(slope=1)
        x = self.downsample(x)           # SPDConv downsampling replaces pooling
        x = self.dropout_conv(x)         # Dropout
        
        # Flatten for fully-connected layers
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 128 * 8 * 8]
        
        # Classifier
        x = F.relu(self.fc1(x))        # FC1 + ReLU
        x = self.dropout_fc1(x)        # Dropout
        x = F.relu(self.fc2(x))        # FC2 + ReLU
        x = self.dropout_fc2(x)        # Dropout
        x = self.fc3(x)                # FC3 (no activation here, Softmax applied in loss)
        
        return x

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Minixception

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.602554Z","iopub.execute_input":"2025-04-19T04:08:43.602740Z","iopub.status.idle":"2025-04-19T04:08:43.620099Z","shell.execute_reply.started":"2025-04-19T04:08:43.602724Z","shell.execute_reply":"2025-04-19T04:08:43.619523Z"}}
class SeparableConv2d(nn.Module):
    """
    Depthwise separable convolution that factorizes a standard convolution into
    depthwise (per-channel) and pointwise (1x1) convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        # Depthwise convolution (one filter per input channel)
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Depthwise: one filter per input channel
            bias=bias
        )
        # Pointwise convolution (1x1 conv to change channel dimensions)
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1,  # 1x1 kernel
            bias=bias
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class XceptionBlock(nn.Module):
    """
    Basic block in Xception architecture with separable convolutions and residual connection.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(XceptionBlock, self).__init__()
        
        # Main branch with 3 separable convolutions
        self.sep_conv1 = nn.Sequential(
            # Change inplace=True to inplace=False for all ReLU activations
            nn.ReLU(inplace=False),
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.sep_conv2 = nn.Sequential(
            nn.ReLU(inplace=False),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.sep_conv3 = nn.Sequential(
            nn.ReLU(inplace=False),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.sep_conv1(x)
        out = self.sep_conv2(out)
        out = self.sep_conv3(out)
        
        return out + residual

class DownsampleBlock(nn.Module):
    """
    Block for spatial downsampling using Space-to-Depth instead of pooling.
    """
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        
        # Main branch: Space-to-Depth downsampling
        self.main_branch = nn.Sequential(
            # Change inplace=True to inplace=False for all ReLU activations
            nn.ReLU(inplace=False),
            SeparableConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )
        
        # Space-to-Depth downsampling (replaces pooling)
        self.spd_downsample = SPDConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            scale=2,  # Scale factor for downsampling
            padding=1
        )
        
        # Skip connection with Space-to-Depth to match dimensions
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # Use SPDConv for skip connection downsampling
            SPDConv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                scale=2,
                padding=0
            )
        )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.main_branch(x)
        out = self.spd_downsample(out)
        
        return out + residual

class MiniXception(nn.Module):
    """
    Xception-inspired model for 32x32x3 inputs with Space-to-Depth downsampling.
    
    This model:
    1. Uses depthwise separable convolutions for efficiency
    2. Replaces pooling with Space-to-Depth operations to preserve information
    3. Uses residual connections throughout for better gradient flow
    4. Optimized for small 32x32 input size
    """
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(MiniXception, self).__init__()
        
        # Entry flow
        # Initial convolution (32x32 -> 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            # Change inplace=True to inplace=False for all ReLU activations
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        
        # First downsampling block (32x32 -> 16x16)
        self.down1 = DownsampleBlock(64, 128)
        
        # XceptionBlock after first downsampling
        self.block1 = XceptionBlock(128, 128)
        
        # Second downsampling block (16x16 -> 8x8)
        self.down2 = DownsampleBlock(128, 256)
        
        # Middle flow - repeated Xception blocks
        self.middle_flow = nn.Sequential(
            XceptionBlock(256, 256),
            XceptionBlock(256, 256),
            XceptionBlock(256, 256)
        )
        
        # Final downsampling to 4x4 resolution
        self.down3 = DownsampleBlock(256, 512)
        
        # Exit flow
        self.exit_flow = nn.Sequential(
            SeparableConv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            SeparableConv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False)
        )
        
        # Global average pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Entry flow
        x = self.stem(x)          # 32x32x64
        x = self.down1(x)         # 16x16x128
        x = self.block1(x)        # 16x16x128
        x = self.down2(x)         # 8x8x256
        
        # Middle flow
        x = self.middle_flow(x)   # 8x8x256
        
        # Exit flow
        x = self.down3(x)         # 4x4x512
        x = self.exit_flow(x)     # 4x4x512
        
        # Global pooling and classification
        x = self.global_pool(x)   # 1x1x512
        x = torch.flatten(x, 1)   # 512
        x = self.dropout(x)
        x = self.classifier(x)    # num_classes
        
        return x

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## LR Net

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.620689Z","iopub.execute_input":"2025-04-19T04:08:43.620880Z","iopub.status.idle":"2025-04-19T04:08:43.636114Z","shell.execute_reply.started":"2025-04-19T04:08:43.620863Z","shell.execute_reply":"2025-04-19T04:08:43.635553Z"}}
class MKBlock(nn.Module):
    """
    PyTorch implementation of the Multi-Kernel (MK) Block from RL-Net.
    Adjusted for 32x32x3 input images with proper channel handling.
    """
    def __init__(self, in_channels):
        super(MKBlock, self).__init__()

        # Layer 1: Parallel Convolutions with different kernel sizes
        # Path 1 (3x3)
        self.conv1_1_3x3 = nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(48)

        # Path 2 (5x5)
        self.conv1_2_5x5 = nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=5, padding=2)
        self.bn1_2 = nn.BatchNorm2d(24)

        # Path 3 (7x7)
        self.conv1_3_7x7 = nn.Conv2d(in_channels=in_channels, out_channels=12, kernel_size=7, padding=3)
        self.bn1_3 = nn.BatchNorm2d(12)

        # Activation
        self.relu = nn.ReLU(inplace=True)

        # Layer 2: Internal Connections and Convolutions
        # Combine 3x3 and 5x5 paths -> Conv 5x5 -> Conv 3x3
        self.conv2_1_5x5 = nn.Conv2d(in_channels=48 + 24, out_channels=36, kernel_size=5, padding=2)
        self.bn2_1 = nn.BatchNorm2d(36)
        self.conv3_1_3x3 = nn.Conv2d(in_channels=36, out_channels=72, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(72)

        # Combine 5x5 and 7x7 paths -> Conv 7x7
        self.conv2_2_7x7 = nn.Conv2d(in_channels=24 + 12, out_channels=18, kernel_size=7, padding=3)
        self.bn2_2 = nn.BatchNorm2d(18)

        # Layer 3: Final Aggregation within block
        # Concatenate specific paths: path1_1 (48) + path2_1 (36) + path3_1 (72) + path2_2 (18) = 174 channels
        final_concat_channels = 48 + 36 + 72 + 18
        self.conv_bottleneck_1x1 = nn.Conv2d(in_channels=final_concat_channels, out_channels=24, kernel_size=1)
        self.bn_bottleneck = nn.BatchNorm2d(24)

        # Max Pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Store output channels for connecting blocks
        self.output_channels = 24

    def forward(self, x):
        # Layer 1 Forward Pass
        path1_1 = self.relu(self.bn1_1(self.conv1_1_3x3(x)))
        path1_2 = self.relu(self.bn1_2(self.conv1_2_5x5(x)))
        path1_3 = self.relu(self.bn1_3(self.conv1_3_7x7(x)))

        # Layer 2 Forward Pass
        # Path originating from 3x3 & 5x5
        concat1 = torch.cat((path1_1, path1_2), dim=1)  # Concatenate along channel dim
        path2_1 = self.relu(self.bn2_1(self.conv2_1_5x5(concat1)))
        path3_1 = self.relu(self.bn3_1(self.conv3_1_3x3(path2_1)))

        # Path originating from 5x5 & 7x7
        concat2 = torch.cat((path1_2, path1_3), dim=1)
        path2_2 = self.relu(self.bn2_2(self.conv2_2_7x7(concat2)))

        # Layer 3 Final Aggregation
        final_concat = torch.cat((path1_1, path2_1, path3_1, path2_2), dim=1)

        # Bottleneck
        bottleneck = self.relu(self.bn_bottleneck(self.conv_bottleneck_1x1(final_concat)))

        # Pooling
        output = self.maxpool(bottleneck)

        return output

class RLNet(nn.Module):
    """
    PyTorch implementation of the adapted RL-Net model.
    Modified for 32x32x3 input images and 4 output classes.
    """
    def __init__(self, num_classes=4, input_channels=3, dropout_rate=0.2):
        super(RLNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Three MK blocks in sequence
        self.mk_block1 = MKBlock(in_channels=input_channels)
        self.mk_block2 = MKBlock(in_channels=self.mk_block1.output_channels)
        self.mk_block3 = MKBlock(in_channels=self.mk_block2.output_channels)

        self.final_bn = nn.BatchNorm2d(self.mk_block3.output_channels)
        self.flatten = nn.Flatten()

        # Calculate the flattened size
        # Initial image size: 32x32
        # After 3 blocks with 2x2 pooling each, size becomes 32/(2^3) = 4x4
        final_size = 4  # 32 // (2*2*2)
        flattened_size = self.mk_block3.output_channels * final_size * final_size

        # Fully Connected Layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(flattened_size, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU(inplace=True)

        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 128)
        self.relu_fc3 = nn.ReLU(inplace=True)

        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(128, 64)
        self.relu_fc4 = nn.ReLU(inplace=True)

        # Output Layer
        self.fc_out = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Multi-Kernel blocks
        x = self.mk_block1(x)
        x = self.mk_block2(x)
        x = self.mk_block3(x)

        # Final processing
        x = self.final_bn(x)
        x = self.flatten(x)

        # Fully connected layers
        x = self.dropout1(x)
        x = self.relu_fc1(self.fc1(x))

        x = self.dropout2(x)
        x = self.relu_fc2(self.fc2(x))

        x = self.dropout3(x)
        x = self.relu_fc3(self.fc3(x))

        x = self.dropout4(x)
        x = self.relu_fc4(self.fc4(x))

        x = self.fc_out(x)  # Logits output
        return x

# Factory function to create the model with default parameters
def create_rlnet(num_classes=4, input_channels=3, dropout_rate=0.2):
    return RLNet(num_classes=num_classes, input_channels=input_channels, dropout_rate=dropout_rate)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## RLSPDNet

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.636669Z","iopub.execute_input":"2025-04-19T04:08:43.636855Z","iopub.status.idle":"2025-04-19T04:08:43.652362Z","shell.execute_reply.started":"2025-04-19T04:08:43.636838Z","shell.execute_reply":"2025-04-19T04:08:43.651767Z"}}
class MKBlockSPD(nn.Module):
    """
    PyTorch implementation of the Multi-Kernel (MK) Block from RL-Net.
    Uses Space-to-Depth instead of MaxPooling for better information preservation.
    """
    def __init__(self, in_channels):
        super(MKBlockSPD, self).__init__()

        # Layer 1: Parallel Convolutions with different kernel sizes
        # Path 1 (3x3)
        self.conv1_1_3x3 = nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(48)

        # Path 2 (5x5)
        self.conv1_2_5x5 = nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=5, padding=2)
        self.bn1_2 = nn.BatchNorm2d(24)

        # Path 3 (7x7)
        self.conv1_3_7x7 = nn.Conv2d(in_channels=in_channels, out_channels=12, kernel_size=7, padding=3)
        self.bn1_3 = nn.BatchNorm2d(12)

        # Activation
        self.relu = nn.ReLU(inplace=True)

        # Layer 2: Internal Connections and Convolutions
        # Combine 3x3 and 5x5 paths -> Conv 5x5 -> Conv 3x3
        self.conv2_1_5x5 = nn.Conv2d(in_channels=48 + 24, out_channels=36, kernel_size=5, padding=2)
        self.bn2_1 = nn.BatchNorm2d(36)
        self.conv3_1_3x3 = nn.Conv2d(in_channels=36, out_channels=72, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(72)

        # Combine 5x5 and 7x7 paths -> Conv 7x7
        self.conv2_2_7x7 = nn.Conv2d(in_channels=24 + 12, out_channels=18, kernel_size=7, padding=3)
        self.bn2_2 = nn.BatchNorm2d(18)

        # Layer 3: Final Aggregation within block
        # Concatenate specific paths: path1_1 (48) + path2_1 (36) + path3_1 (72) + path2_2 (18) = 174 channels
        final_concat_channels = 48 + 36 + 72 + 18
        self.conv_bottleneck_1x1 = nn.Conv2d(in_channels=final_concat_channels, out_channels=24, kernel_size=1)
        self.bn_bottleneck = nn.BatchNorm2d(24)

        # Replace MaxPooling with Space-to-Depth Conv for better information preservation
        self.spd_downsampling = SpaceToDepthConv(
            in_channels=24,
            out_channels=24,
            kernel_size=3,
            block_size=2,  # Similar to stride=2 in MaxPool
            padding=1
        )
        
        # Store output channels for connecting blocks
        self.output_channels = 24

    def forward(self, x):
        # Layer 1 Forward Pass
        path1_1 = self.relu(self.bn1_1(self.conv1_1_3x3(x)))
        path1_2 = self.relu(self.bn1_2(self.conv1_2_5x5(x)))
        path1_3 = self.relu(self.bn1_3(self.conv1_3_7x7(x)))

        # Layer 2 Forward Pass
        # Path originating from 3x3 & 5x5
        concat1 = torch.cat((path1_1, path1_2), dim=1)  # Concatenate along channel dim
        path2_1 = self.relu(self.bn2_1(self.conv2_1_5x5(concat1)))
        path3_1 = self.relu(self.bn3_1(self.conv3_1_3x3(path2_1)))

        # Path originating from 5x5 & 7x7
        concat2 = torch.cat((path1_2, path1_3), dim=1)
        path2_2 = self.relu(self.bn2_2(self.conv2_2_7x7(concat2)))

        # Layer 3 Final Aggregation
        final_concat = torch.cat((path1_1, path2_1, path3_1, path2_2), dim=1)

        # Bottleneck
        bottleneck = self.relu(self.bn_bottleneck(self.conv_bottleneck_1x1(final_concat)))

        # Use SPD instead of MaxPooling for downsampling
        output = self.spd_downsampling(bottleneck)

        return output

class RLSPDNet(nn.Module):
    """
    PyTorch implementation of the RL-Net model with Space-to-Depth downsampling.
    Uses SPD instead of MaxPooling for better information preservation at low resolutions.
    Modified for 32x32x3 input images and 4 output classes.
    """
    def __init__(self, num_classes=4, input_channels=3, dropout_rate=0.2):
        super(RLSPDNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Three MK blocks with SPD instead of MaxPool
        self.mk_block1 = MKBlockSPD(in_channels=input_channels)
        self.mk_block2 = MKBlockSPD(in_channels=self.mk_block1.output_channels)
        self.mk_block3 = MKBlockSPD(in_channels=self.mk_block2.output_channels)

        self.final_bn = nn.BatchNorm2d(self.mk_block3.output_channels)
        self.flatten = nn.Flatten()

        # Calculate the flattened size
        # Initial image size: 32x32
        # After 3 blocks with 2x2 SPD each, size becomes 32/(2^3) = 4x4
        final_size = 4  # 32 // (2*2*2)
        flattened_size = self.mk_block3.output_channels * final_size * final_size

        # Fully Connected Layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(flattened_size, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU(inplace=True)

        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 128)
        self.relu_fc3 = nn.ReLU(inplace=True)

        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(128, 64)
        self.relu_fc4 = nn.ReLU(inplace=True)

        # Output Layer
        self.fc_out = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Multi-Kernel blocks with SPD downsampling
        x = self.mk_block1(x)
        x = self.mk_block2(x)
        x = self.mk_block3(x)

        # Final processing
        x = self.final_bn(x)
        x = self.flatten(x)

        # Fully connected layers
        x = self.dropout1(x)
        x = self.relu_fc1(self.fc1(x))

        x = self.dropout2(x)
        x = self.relu_fc2(self.fc2(x))

        x = self.dropout3(x)
        x = self.relu_fc3(self.fc3(x))

        x = self.dropout4(x)
        x = self.relu_fc4(self.fc4(x))

        x = self.fc_out(x)  # Logits output
        return x

# Factory function to create the model with default parameters
def create_rlspdnet(num_classes=4, input_channels=3, dropout_rate=0.2):
    return RLSPDNet(num_classes=num_classes, input_channels=input_channels, dropout_rate=dropout_rate)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Mixmodel1

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.652983Z","iopub.execute_input":"2025-04-19T04:08:43.653181Z","iopub.status.idle":"2025-04-19T04:08:43.672394Z","shell.execute_reply.started":"2025-04-19T04:08:43.653164Z","shell.execute_reply":"2025-04-19T04:08:43.671822Z"}}
class Mix1SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """
    def __init__(self, channels, reduction=16):
        super(Mix1SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 8)  # Ensure minimum size
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        # Global average pooling
        y = self.avg_pool(x).view(b, c)
        # Channel attention
        y = self.fc(y).view(b, c, 1, 1)
        # Scale the input
        return x * y.expand_as(x)

class Mix1ResidualInceptionBlock(nn.Module):
    """
    Residual Inception Block with dilated convolutions and SE attention.
    """
    def __init__(self, in_channels, out_channels, path_channels, dilations=[1, 2, 3]):
        super(Mix1ResidualInceptionBlock, self).__init__()
        
        # Ensure in_channels match out_channels for residual connection
        self.adjust_channels = None
        if in_channels != out_channels:
            self.adjust_channels = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        # Path 1: 1x1 conv only
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, path_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(path_channels),
            nn.ReLU(inplace=True)
        )
        
        # Paths 2-4: 1x1 conv -> 3x3 dilated conv
        # Dynamically create paths based on dilations parameter
        self.paths = nn.ModuleList()
        for dilation in dilations:
            self.paths.append(nn.Sequential(
                nn.Conv2d(in_channels, path_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(path_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(path_channels, path_channels, kernel_size=3, padding=dilation, 
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(path_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Final 1x1 conv to combine outputs
        self.combine = nn.Sequential(
            nn.Conv2d(path_channels * (1 + len(dilations)), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # SE attention block
        self.se = Mix1SEBlock(out_channels)
        
        # ReLU after residual connection
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Save input for residual connection
        residual = x if self.adjust_channels is None else self.adjust_channels(x)
        
        # Process through path 1
        path1_out = self.path1(x)
        
        # Process through additional paths with dilated convolutions
        path_outputs = [path1_out]
        for path in self.paths:
            path_outputs.append(path(x))
        
        # Concatenate all path outputs
        combined = torch.cat(path_outputs, dim=1)
        
        # Final 1x1 convolution
        out = self.combine(combined)
        
        # Apply SE attention
        out = self.se(out)
        
        # Add residual connection and apply ReLU
        out = self.relu(out + residual)
        
        return out

class Mix1SpaceToDepthConv(nn.Module):
    """
    Space-to-Depth Convolution with custom channel adjustment.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Mix1SpaceToDepthConv, self).__init__()
        # After space-to-depth, channels increase by scale_factor^2
        self.spd = nn.PixelUnshuffle(scale_factor)
        
        # 1x1 conv to adjust channels
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(in_channels * (scale_factor ** 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Apply space-to-depth operation
        x = self.spd(x)
        # Adjust channels
        x = self.channel_adjust(x)
        return x

class MixModel1(nn.Module):
    """
    MixModel1 combining SPDConv, Residual Inception blocks with dilated convolutions,
    and Squeeze-and-Excitation attention for 32x32 images.
    
    Architecture follows the layer-by-layer design specified:
    1. SPDConv initial downsampling
    2. Channel adjustment
    3. Multi-stage feature extraction with residual inception blocks
    4. Progressive downsampling using SPDConv
    5. Global pooling and classification
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MixModel1, self).__init__()
        
        # Initial Layer: SPDConv (32x32x3 → 16x16x12)
        self.initial_spd = SPDConv(in_channels=3, out_channels=12, scale=2, kernel_size=3, padding=1)
        
        # Channel Adjustment: 1x1 Conv (16x16x12 → 16x16x64)
        self.initial_channel_adjust = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual Inception Block A1 (16x16x64 → 16x16x64)
        self.block_a1 = Mix1ResidualInceptionBlock(
            in_channels=64,
            out_channels=64,
            path_channels=16,
            dilations=[1, 2, 3]
        )
        
        # Residual Inception Block A2 (16x16x64 → 16x16x64)
        self.block_a2 = Mix1ResidualInceptionBlock(
            in_channels=64,
            out_channels=64,
            path_channels=16,
            dilations=[1, 2, 3]
        )
        
        # Downsampling: SPDConv (16x16x64 → 8x8x256)
        self.downsample1 = SPDConv(
            in_channels=64,
            out_channels=256,
            scale=2
        )
        
        # Channel Adjustment: 1x1 Conv (8x8x256 → 8x8x128)
        self.channel_adjust1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Residual Inception Block B1 (8x8x128 → 8x8x128)
        self.block_b1 = Mix1ResidualInceptionBlock(
            in_channels=128,
            out_channels=128,
            path_channels=32,
            dilations=[1, 2, 3]
        )
        
        # Residual Inception Block B2 (8x8x128 → 8x8x128)
        self.block_b2 = Mix1ResidualInceptionBlock(
            in_channels=128,
            out_channels=128,
            path_channels=32,
            dilations=[1, 2, 3]
        )
        
        # Downsampling: SPDConv (8x8x128 → 4x4x512)
        self.downsample2 = SPDConv(
            in_channels=128,
            out_channels=512,
            scale=2
        )
        
        # Channel Adjustment: 1x1 Conv (4x4x512 → 4x4x256)
        self.channel_adjust2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual Inception Block C1 (4x4x256 → 4x4x256)
        self.block_c1 = Mix1ResidualInceptionBlock(
            in_channels=256,
            out_channels=256,
            path_channels=64,
            dilations=[1, 2, 3]
        )
        
        # Global Feature Aggregation: Global Average Pooling (4x4x256 → 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Layer (256 → num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial downsampling: SPDConv (32x32x3 → 16x16x12)
        x = self.initial_spd(x)
        
        # Channel adjustment (16x16x12 → 16x16x64)
        x = self.initial_channel_adjust(x)
        
        # Stage A: 16x16 resolution
        x = self.block_a1(x)
        x = self.block_a2(x)
        
        # Downsampling to 8x8
        x = self.downsample1(x)
        x = self.channel_adjust1(x)
        
        # Stage B: 8x8 resolution
        x = self.block_b1(x)
        x = self.block_b2(x)
        
        # Downsampling to 4x4
        x = self.downsample2(x)
        x = self.channel_adjust2(x)
        
        # Stage C: 4x4 resolution
        x = self.block_c1(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Mixmodel2

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.672999Z","iopub.execute_input":"2025-04-19T04:08:43.673198Z","iopub.status.idle":"2025-04-19T04:08:43.691967Z","shell.execute_reply.started":"2025-04-19T04:08:43.673181Z","shell.execute_reply":"2025-04-19T04:08:43.691343Z"}}
class Mix2SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """
    def __init__(self, channels, reduction=16):
        super(Mix2SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 8)  # Ensure minimum size
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Mix2InceptionBlockA(nn.Module):
    """
    Inception block with four parallel paths and dilated convolutions.
    """
    def __init__(self, in_channels, filters=16):
        super(Mix2InceptionBlockA, self).__init__()
        
        # Path 1: 1x1 convolution only
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Path 2: 1x1 → 3x3 conv with dilation=1
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Path 3: 1x1 → 3x3 conv with dilation=2
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Path 4: 1x1 → 3x3 conv with dilation=3
        self.path4 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)
        return torch.cat([path1_out, path2_out, path3_out, path4_out], dim=1)

class Mix2InceptionBlockC(nn.Module):
    """
    Simplified inception block with two parallel paths.
    """
    def __init__(self, in_channels, filters=64):
        super(Mix2InceptionBlockC, self).__init__()
        
        # Path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Path 2: 3x3 convolution
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        return torch.cat([path1_out, path2_out], dim=1)

class MixModel2(nn.Module):
    """
    MixModel2 combining Space-to-Depth convolutions, Inception blocks with dilated convolutions,
    and Squeeze-and-Excitation attention for 32x32 images.
    
    This model features:
    1. Initial downsampling with Space-to-Depth
    2. Multi-scale feature extraction with Inception blocks
    3. Skip connections in feature stages for better gradient flow
    4. SE attention for channel-wise feature refinement
    5. Progressive downsampling with Space-to-Depth operations
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MixModel2, self).__init__()
        
        # Initial Space-to-Depth convolution (32x32x3 → 16x16x12)
        self.spd_initial = SPDConv(in_channels=3, out_channels=12, scale=2, kernel_size=1, padding=0)
        
        # Stage A: First inception stage at 16x16 resolution
        self.inception_a1 = Mix2InceptionBlockA(in_channels=12, filters=16)  # 16x16x12 → 16x16x64
        self.se_a1 = Mix2SEBlock(channels=64)  # SE attention on inception output
        
        self.inception_a2 = Mix2InceptionBlockA(in_channels=64, filters=16)  # 16x16x64 → 16x16x64
        self.se_a2 = Mix2SEBlock(channels=64)  # SE attention on inception output
        
        # Downsampling to Stage B (16x16x64 → 8x8x256)
        self.spd_b = SPDConv(in_channels=64, out_channels=256, scale=2, kernel_size=1, padding=0)
        
        # Stage B: Second inception stage at 8x8 resolution
        self.inception_b1 = Mix2InceptionBlockA(in_channels=256, filters=32)  # 8x8x256 → 8x8x128
        self.se_b1 = Mix2SEBlock(channels=128)  # SE attention on inception output
        
        self.inception_b2 = Mix2InceptionBlockA(in_channels=128, filters=32)  # 8x8x128 → 8x8x128
        self.se_b2 = Mix2SEBlock(channels=128)  # SE attention on inception output
        
        # Downsampling to Stage C (8x8x128 → 4x4x512)
        self.spd_c = SPDConv(in_channels=128, out_channels=512, scale=2, kernel_size=1, padding=0)
        
        # Stage C: Final inception stage at 4x4 resolution
        self.inception_c = Mix2InceptionBlockC(in_channels=512, filters=64)  # 4x4x512 → 4x4x128
        self.se_c = Mix2SEBlock(channels=128)  # SE attention on inception output
        
        # Global Average Pooling and Classification
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 4x4x128 → 1x1x128
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_classes)  # 128 → num_classes
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial Space-to-Depth
        x = self.spd_initial(x)  # 3x32x32 → 12x16x16
        
        # Stage A (16x16 resolution)
        x = self.inception_a1(x)  # 12x16x16 → 64x16x16
        x = self.se_a1(x)
        
        # Skip connection for Stage A's second block
        x_a2_input = x
        x = self.inception_a2(x)  # 64x16x16 → 64x16x16
        x = x + x_a2_input  # Skip connection
        x = self.se_a2(x)
        
        # Downsample to Stage B
        x = self.spd_b(x)  # 64x16x16 → 256x8x8
        
        # Stage B (8x8 resolution)
        x = self.inception_b1(x)  # 256x8x8 → 128x8x8
        x = self.se_b1(x)
        
        # Skip connection for Stage B's second block
        x_b2_input = x
        x = self.inception_b2(x)  # 128x8x8 → 128x8x8
        x = x + x_b2_input  # Skip connection
        x = self.se_b2(x)
        
        # Downsample to Stage C
        x = self.spd_c(x)  # 128x8x8 → 512x4x4
        
        # Stage C (4x4 resolution)
        x = self.inception_c(x)  # 512x4x4 → 128x4x4
        x = self.se_c(x)
        
        # Global Average Pooling and Classification
        x = self.gap(x)  # 128x4x4 → 128x1x1
        x = torch.flatten(x, 1)  # 128x1x1 → 128
        x = self.dropout(x)
        x = self.fc(x)  # 128 → num_classes
        
        return x

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Mixmodel3

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.692592Z","iopub.execute_input":"2025-04-19T04:08:43.692776Z","iopub.status.idle":"2025-04-19T04:08:43.712852Z","shell.execute_reply.started":"2025-04-19T04:08:43.692760Z","shell.execute_reply":"2025-04-19T04:08:43.712281Z"}}
class Mix3SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """
    def __init__(self, channels, reduction=16):
        super(Mix3SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 8)  # Ensure minimum size
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Mix3SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important spatial locations.
    """
    def __init__(self, kernel_size=7):
        super(Mix3SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        # Calculate spatial attention map
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(attention)
        attention = torch.sigmoid(attention)
        
        # Apply attention
        return x * attention

class Mix3DilatedMultiScaleBlock(nn.Module):
    """
    Multi-scale feature extraction block using dilated convolutions without spatial reduction.
    Includes residual connection and attention mechanisms.
    """
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4], use_attention=True):
        super(Mix3DilatedMultiScaleBlock, self).__init__()
        self.use_attention = use_attention
        
        # Calculate channels per path to maintain reasonable parameter count
        self.path_channels = out_channels // 4  # 4 paths
        
        # Path 1: 1x1 convolution only (point-wise)
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, self.path_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.path_channels),
            nn.ReLU(inplace=False)
        )
        
        # Path 2-4: Different dilated convolutions
        self.path_modules = nn.ModuleList()
        for dilation in dilations:
            self.path_modules.append(nn.Sequential(
                nn.Conv2d(in_channels, self.path_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.path_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    self.path_channels, 
                    self.path_channels, 
                    kernel_size=3, 
                    padding=dilation, 
                    dilation=dilation,
                    bias=False
                ),
                nn.BatchNorm2d(self.path_channels),
                nn.ReLU(inplace=False)
            ))
        
        # Projection to combine all paths back to out_channels
        self.project = nn.Sequential(
            nn.Conv2d(self.path_channels * (1 + len(dilations)), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Residual connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Attention mechanisms
        if use_attention:
            self.channel_attn = Mix3SEBlock(out_channels)
            self.spatial_attn = Mix3SpatialAttention()
            
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        residual = self.skip(x)
        
        # Multi-scale feature extraction
        path1_out = self.path1(x)
        other_paths = [module(x) for module in self.path_modules]
        
        # Concatenate all paths
        concat_features = torch.cat([path1_out] + other_paths, dim=1)
        
        # Project back to output channels
        out = self.project(concat_features)
        
        # Add residual connection
        out = out + residual
        out = self.relu(out)
        
        # Apply attention if enabled
        if self.use_attention:
            out = self.channel_attn(out)
            out = self.spatial_attn(out)
            
        return out

class MixModel3(nn.Module):
    """
    MixModel3: A CNN that maintains 32x32 spatial resolution throughout the network.
    
    Key features:
    1. No spatial downsampling - preserves all spatial information
    2. Uses dilated convolutions to increase receptive field without resolution loss
    3. Combines channel and spatial attention mechanisms
    4. Efficient parameter usage with multi-scale feature extraction
    5. Only reduces spatial dimensions at the final classification stage
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MixModel3, self).__init__()
        
        # Stage 1: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        
        # Stage 2: Multi-scale feature extraction blocks (maintains 32x32 resolution)
        self.stage2 = nn.Sequential(
            Mix3DilatedMultiScaleBlock(64, 96, dilations=[1, 2, 4]),
            Mix3DilatedMultiScaleBlock(96, 128, dilations=[1, 3, 5])
        )
        
        # Stage 3: Deeper feature extraction with increased dilations (maintains 32x32 resolution)
        self.stage3 = nn.Sequential(
            Mix3DilatedMultiScaleBlock(128, 192, dilations=[1, 3, 6]),
            Mix3DilatedMultiScaleBlock(192, 256, dilations=[1, 4, 8])
        )
        
        # Stage 4: Final feature refinement (maintains 32x32 resolution)
        self.stage4 = Mix3DilatedMultiScaleBlock(256, 384, dilations=[1, 2, 5, 9], use_attention=True)
        
        # Stage 5: Efficient feature aggregation
        # Multi-scale spatial pooling to capture different levels of detail
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 32x32 → 1x1
        self.mid_pool = nn.AdaptiveAvgPool2d((2, 2))     # 32x32 → 2x2
        self.local_pool = nn.AdaptiveAvgPool2d((4, 4))   # 32x32 → 4x4
        
        # Flatten and combine pooled features
        global_features = 384  # From global_pool
        mid_features = 384 * 4  # From mid_pool (2x2)
        local_features = 384 * 16  # From local_pool (4x4)
        total_features = global_features + mid_features + local_features
        
        # Feature fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate/2)
        )
        
        # Classification layer
        self.classifier = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: 32x32x3
        
        # Stage 1: Initial feature extraction
        x = self.stem(x)  # 32x32x64
        
        # Stage 2: Multi-scale feature extraction
        x = self.stage2(x)  # 32x32x128
        
        # Stage 3: Deeper feature extraction
        x = self.stage3(x)  # 32x32x256
        
        # Stage 4: Final feature refinement
        x = self.stage4(x)  # 32x32x384
        
        # Stage 5: Multi-scale feature aggregation
        global_features = self.global_pool(x)  # 1x1x384
        mid_features = self.mid_pool(x)        # 2x2x384
        local_features = self.local_pool(x)    # 4x4x384
        
        # Flatten and concatenate
        global_features = torch.flatten(global_features, 1)  # 384
        mid_features = torch.flatten(mid_features, 1)        # 1536 (384*2*2)
        local_features = torch.flatten(local_features, 1)    # 6144 (384*4*4)
        
        concat_features = torch.cat([global_features, mid_features, local_features], dim=1)
        
        # Feature fusion
        fused = self.fusion(concat_features)
        
        # Classification
        out = self.classifier(fused)
        
        return out

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Mixmodel4

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.713492Z","iopub.execute_input":"2025-04-19T04:08:43.713693Z","iopub.status.idle":"2025-04-19T04:08:43.734788Z","shell.execute_reply.started":"2025-04-19T04:08:43.713676Z","shell.execute_reply":"2025-04-19T04:08:43.734211Z"}}
class Mix4MultiBranchModule(nn.Module):
    """
    Multi-branch module with three paths for learning different spatial features:
    - Branch 1: Local features (1x1 conv)
    - Branch 2: Medium-scale features (3x3, 5x5 dilated=1, 7x7 dilated=2 convs)
    - Branch 3: Global features (5x5, 7x7 dilated=1, 9x9 dilated=2 convs)
    
    Branches process features in parallel to efficiently capture different receptive fields.
    
    Resolution flow:
    - Input: HxW with C channels
    - Branch 1 (Local features): HxW preserved (1x1 kernels don't affect spatial dimensions)
    - Branch 2 (Medium features): HxW preserved for all convs with proper padding
      * 3x3 conv (pad=1): Receptive field = 3x3
      * 5x5 conv (pad=2): Receptive field = 5x5
      * 7x7 conv with dilation=2 (pad=6): Effective receptive field = 13x13
    - Branch 3 (Global features): HxW preserved for all convs with proper padding
      * 5x5 conv (pad=2): Receptive field = 5x5
      * 7x7 conv (pad=3): Receptive field = 7x7
      * 9x9 conv with dilation=2 (pad=8): Effective receptive field = 17x17
    - Output: HxW with out_channels (spatial dimensions maintained)
    """
    def __init__(self, in_channels, out_channels, use_residual=True):
        super(Mix4MultiBranchModule, self).__init__()
        
        # Calculate channels per branch to maintain reasonable parameter count
        self.branch1_channels = out_channels // 4
        self.branch2_channels = out_channels // 4
        self.branch3_channels = out_channels // 2
        
        # Adjust if the division isn't exact
        total_channels = self.branch1_channels + self.branch2_channels + self.branch3_channels
        if total_channels != out_channels:
            self.branch3_channels += (out_channels - total_channels)
        
        # Branch 1: Local features with 1x1 conv (smallest receptive field)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, self.branch1_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.branch1_channels),
            nn.ReLU(inplace=False)
        )
        
        # Branch 2: Medium-scale features with specific convolutions
        branch2_channels_per_conv = self.branch2_channels // 3
        remainder = self.branch2_channels - (branch2_channels_per_conv * 3)
        
        self.branch2_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch2_channels_per_conv, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(branch2_channels_per_conv),
            nn.ReLU(inplace=False)
        )
        
        self.branch2_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch2_channels_per_conv, kernel_size=5, 
                     stride=1, padding=2, dilation=1, bias=False),
            nn.BatchNorm2d(branch2_channels_per_conv),
            nn.ReLU(inplace=False)
        )
        
        self.branch2_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, branch2_channels_per_conv + remainder, kernel_size=7, 
                     stride=1, padding=6, dilation=2, bias=False),
            nn.BatchNorm2d(branch2_channels_per_conv + remainder),
            nn.ReLU(inplace=False)
        )
        
        # Branch 3: Global features with larger kernels and dilations
        branch3_channels_per_conv = self.branch3_channels // 3
        remainder = self.branch3_channels - (branch3_channels_per_conv * 3)
        
        self.branch3_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch3_channels_per_conv, kernel_size=5, 
                     stride=1, padding=2, dilation=1, bias=False),
            nn.BatchNorm2d(branch3_channels_per_conv),
            nn.ReLU(inplace=False)
        )
        
        self.branch3_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, branch3_channels_per_conv, kernel_size=7, 
                     stride=1, padding=3, dilation=1, bias=False),
            nn.BatchNorm2d(branch3_channels_per_conv),
            nn.ReLU(inplace=False)
        )
        
        self.branch3_9x9 = nn.Sequential(
            nn.Conv2d(in_channels, branch3_channels_per_conv + remainder, kernel_size=9, 
                     stride=1, padding=8, dilation=2, bias=False),
            nn.BatchNorm2d(branch3_channels_per_conv + remainder),
            nn.ReLU(inplace=False)
        )
        
        # Residual connection setup
        self.use_residual = use_residual
        self.residual = nn.Identity()
        if use_residual and in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        # Input resolution: HxW with in_channels
        residual = x if not self.use_residual else self.residual(x)
        
        # Branch 1 - Local features with 1x1 conv (preserves spatial resolution)
        # Resolution: HxW → HxW with branch1_channels
        b1 = self.branch1(x)
        
        # Branch 2 - Medium features with specified convs (all preserve spatial resolution)
        # 3x3 conv - Resolution: HxW → HxW with branch2_channels/3
        b2_3x3 = self.branch2_3x3(x)
        
        # 5x5 conv - Resolution: HxW → HxW with branch2_channels/3 
        b2_5x5 = self.branch2_5x5(x)
        
        # 7x7 dilated conv - Resolution: HxW → HxW with branch2_channels/3
        # Effective receptive field: 13x13 due to dilation=2
        b2_7x7 = self.branch2_7x7(x)
        
        # Branch 3 - Global features with larger kernels (all preserve spatial resolution)
        # 5x5 conv - Resolution: HxW → HxW with branch3_channels/3
        b3_5x5 = self.branch3_5x5(x)
        
        # 7x7 conv - Resolution: HxW → HxW with branch3_channels/3
        b3_7x7 = self.branch3_7x7(x)
        
        # 9x9 dilated conv - Resolution: HxW → HxW with branch3_channels/3
        # Effective receptive field: 17x17 due to dilation=2
        b3_9x9 = self.branch3_9x9(x)
        
        # Concatenate branch outputs along channel dimension
        # Resolution: HxW, Channels = sum of all branch channels (equals out_channels)
        out = torch.cat([
            b1, 
            b2_3x3, b2_5x5, b2_7x7,
            b3_5x5, b3_7x7, b3_9x9
        ], dim=1)
        
        # Add residual connection if enabled (preserves spatial resolution)
        if self.use_residual:
            out = out + residual
            out = self.relu(out)
        
        # Output resolution: HxW with out_channels (spatial dimensions maintained)
        return out

class Mix4SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """
    def __init__(self, channels, reduction=16):
        super(Mix4SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 8)  # Ensure minimum size
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MixModel4(nn.Module):
    """
    MixModel4: A CNN with multi-branch modules capturing different spatial scales.
    
    Key features:
    1. Parallel branches for different receptive fields:
       - Branch 1: Local features with 1x1 conv
       - Branch 2: Medium features with parallel 3x3, 5x5, 7x7 convolutions
       - Branch 3: Global features with parallel 5x5, 7x7, 9x9 convolutions
    2. Minimal downsampling using SPDConv when needed
    3. Residual connections throughout the network
    4. Multi-scale feature aggregation for final classification
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MixModel4, self).__init__()
        
        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        
        # Stage 1: Multi-branch modules with SE attention (32x32 spatial size)
        self.stage1 = nn.Sequential(
            Mix4MultiBranchModule(64, 96, use_residual=True),
            Mix4SEBlock(96),
            Mix4MultiBranchModule(96, 128, use_residual=True),
            Mix4SEBlock(128)
        )
        
        # Optional downsampling with SPDConv (32x32 → 16x16)
        self.downsample1 = SPDConv(128, 192, scale=2)
        
        # Stage 2: Multi-branch modules with SE attention (16x16 spatial size)
        self.stage2 = nn.Sequential(
            Mix4MultiBranchModule(192, 256, use_residual=True),
            Mix4SEBlock(256),
            Mix4MultiBranchModule(256, 384, use_residual=True),
            Mix4SEBlock(384)
        )
        
        # Multi-scale pooling for feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global features
        self.mid_pool = nn.AdaptiveAvgPool2d((2, 2))     # Medium-scale features
        self.local_pool = nn.AdaptiveAvgPool2d((4, 4))   # Local features
        
        # Calculate feature dimensions
        global_features = 384           # 1x1x384
        mid_features = 384 * 2 * 2      # 2x2x384
        local_features = 384 * 4 * 4    # 4x4x384
        total_features = global_features + mid_features + local_features
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate/2)
        )
        
        # Classification layer
        self.classifier = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: 3x32x32
        
        # Initial feature extraction
        x = self.stem(x)  # 3x32x32 → 64x32x32 (maintains spatial resolution)
        
        # Stage 1: Multi-branch modules at 32x32 resolution
        x = self.stage1(x)  # 64x32x32 → 128x32x32 (maintains spatial resolution)
        
        # Downsampling (only once in the network)
        x = self.downsample1(x)  # 128x32x32 → 192x16x16 (spatial reduction by factor of 2)
        
        # Stage 2: Multi-branch modules at 16x16 resolution
        x = self.stage2(x)  # 192x16x16 → 384x16x16 (maintains 16x16 resolution)
        
        # Multi-scale feature aggregation
        global_features = self.global_pool(x)  # 384x16x16 → 384x1x1 (global pooling)
        mid_features = self.mid_pool(x)        # 384x16x16 → 384x2x2 (medium pooling)
        local_features = self.local_pool(x)    # 384x16x16 → 384x4x4 (local pooling)
        
        # Flatten and concatenate features
        global_features = torch.flatten(global_features, 1)  # 384x1x1 → 384 (flattened)
        mid_features = torch.flatten(mid_features, 1)        # 384x2x2 → 1536 (flattened)
        local_features = torch.flatten(local_features, 1)    # 384x4x4 → 6144 (flattened)
        concat_features = torch.cat([global_features, mid_features, local_features], dim=1)
        # Concatenated features: 384 + 1536 + 6144 = 8064 dimensions
        
        # Feature fusion
        fused = self.fusion(concat_features)  # 8064 → 256 (with intermediate 512)
        
        # Classification
        out = self.classifier(fused)  # 256 → num_classes

        return out

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## PixT

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.735420Z","iopub.execute_input":"2025-04-19T04:08:43.735623Z","iopub.status.idle":"2025-04-19T04:08:43.769180Z","shell.execute_reply.started":"2025-04-19T04:08:43.735606Z","shell.execute_reply":"2025-04-19T04:08:43.768604Z"}}
class TransformerConfig:
    """
    Configuration class for PixelTransformer hyperparameters.
    
    This allows easy customization of transformer architecture without 
    changing the model implementation.
    """
    def __init__(self, 
                 img_size=32,
                 d_model=128, 
                 nhead=None,  # Will be auto-calculated if None
                 num_layers=6,
                 dim_feedforward=None,  # Will be auto-calculated if None
                 dropout=0.1,
                 activation="gelu",
                 # Add new memory efficiency parameters
                 use_gradient_checkpointing=False,
                 sequence_reduction_factor=1,
                 share_layer_params=False,
                 use_sequence_downsampling=False):
        self.img_size = img_size
        self.d_model = d_model
        # Auto-calculate number of heads if not specified
        self.nhead = nhead if nhead is not None else max(4, d_model // 32)
        self.num_layers = num_layers
        # Auto-calculate feedforward dimension if not specified (typical transformer uses 4x d_model)
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else d_model * 4
        self.dropout = dropout
        self.activation = activation
        
        # Memory optimization parameters
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.sequence_reduction_factor = sequence_reduction_factor
        self.share_layer_params = share_layer_params
        self.use_sequence_downsampling = use_sequence_downsampling
    
    @classmethod
    def tiny(cls, img_size=32):
        """Small and efficient transformer configuration."""
        return cls(img_size=img_size, d_model=96, num_layers=4, dropout=0.1)
    
    @classmethod
    def small(cls, img_size=32):
        """Balanced transformer configuration for small images."""
        return cls(img_size=img_size, d_model=192, num_layers=6, dropout=0.1)
    
    @classmethod
    def base(cls, img_size=32):
        """Larger transformer configuration with more capacity."""
        return cls(img_size=img_size, d_model=256, num_layers=8, dropout=0.1)
    
    @classmethod
    def memory_efficient(cls, img_size=32):
        """Memory-efficient configuration with optimizations for reduced VRAM usage."""
        return cls(
            img_size=img_size, 
            d_model=192,              # Still decent model capacity
            num_layers=6,             # Keep layer depth for capacity
            dropout=0.1,
            use_gradient_checkpointing=True,  # Enable checkpointing
            share_layer_params=True,          # Share parameters between layers
            use_sequence_downsampling=True    # Use sequence reduction
        )

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer sequence.
    Works with flattened pixels from images, treating each pixel as a token position.
    """
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create standard 1D positional encoding matrix
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with multi-head self-attention and feed-forward networks.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection and layer norm
        # x shape: [batch_size, seq_len, d_model]
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)
        
        return x

# Add a sequence reduction module
class SequencePooling(nn.Module):
    """Reduces sequence length by pooling nearby tokens."""
    def __init__(self, d_model, reduction_factor=2, mode='mean'):
        super(SequencePooling, self).__init__()
        self.reduction_factor = reduction_factor
        self.mode = mode
        if mode == 'learned':
            self.projection = nn.Linear(d_model * reduction_factor, d_model)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # Check if the sequence length is divisible by reduction factor
        if seq_len % self.reduction_factor != 0:
            # Add padding tokens if needed
            pad_len = self.reduction_factor - (seq_len % self.reduction_factor)
            padding = torch.zeros(batch_size, pad_len, d_model, device=x.device)
            x = torch.cat([x, padding], dim=1)
            seq_len = x.shape[1]
        
        # Reshape to group tokens
        x = x.reshape(batch_size, seq_len // self.reduction_factor, self.reduction_factor, d_model)
        
        # Pool tokens
        if self.mode == 'mean':
            # Mean pooling (most efficient)
            x = torch.mean(x, dim=2)  # [batch_size, seq_len//reduction_factor, d_model]
        elif self.mode == 'max':
            # Max pooling
            x = torch.max(x, dim=2)[0]  # [batch_size, seq_len//reduction_factor, d_model]
        elif self.mode == 'learned':
            # Learned pooling
            x = x.reshape(batch_size, seq_len // self.reduction_factor, self.reduction_factor * d_model)
            x = self.projection(x)  # [batch_size, seq_len//reduction_factor, d_model]
        
        return x

class PixelTransformer(nn.Module):
    """
    Pixel Transformer (PixT) that treats each pixel as a token for 32x32 images.
    Memory-optimized version with gradient checkpointing and other efficiency options.
    
    Unlike Vision Transformer (ViT) which divides images into patches,
    PixT works directly with individual pixels as tokens, making it suitable
    for already-small images like 32x32.
    
    Flowchart:
    Input (32x32x3)
    → Pixel Embedding (1024x128)
    → Positional Encoding
    → Transformer Blocks
    → Classification Token
    → MLP Head
    → Output (num_classes)
    """
    def __init__(self, num_classes=10, d_model=128, nhead=8, num_layers=6, 
                dim_feedforward=512, dropout=0.1, img_size=32,
                use_gradient_checkpointing=False,
                sequence_reduction_factor=1,
                share_layer_params=False,
                use_sequence_downsampling=False):
        super(PixelTransformer, self).__init__()
        
        self.img_size = img_size
        self.num_pixels = img_size * img_size
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.share_layer_params = share_layer_params
        
        # Pixel embedding - projects each RGB pixel to d_model dimensions
        self.pixel_embedding = nn.Linear(3, d_model)
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Sequence reduction if enabled
        self.sequence_reduction = None
        if use_sequence_downsampling and sequence_reduction_factor > 1:
            # Apply sequence reduction after adding CLS token and positional encoding
            self.sequence_reduction = SequencePooling(
                d_model, 
                reduction_factor=sequence_reduction_factor,
                mode='mean'
            )
            # Calculate effective sequence length after downsampling
            effective_seq_len = (self.num_pixels // sequence_reduction_factor) + 1  # +1 for CLS token
        else:
            effective_seq_len = self.num_pixels + 1  # Original length +1 for CLS token
            
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model, 
            max_len=effective_seq_len, 
            dropout=dropout
        )
        
        # Create transformer blocks with parameter sharing if enabled
        if share_layer_params:
            # Create a single transformer block to be reused
            shared_block = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            # Create a list that refers to the same block multiple times
            self.transformer_blocks = nn.ModuleList([shared_block] * num_layers)
        else:
            # Create separate transformer blocks as usual
            self.transformer_blocks = nn.ModuleList([
                TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        # Initialize classification token
        torch.nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Input: [batch_size, channels, height, width]
        batch_size = x.size(0)
        
        # 1. Reshape to [batch_size, height*width, channels]
        x = x.permute(0, 2, 3, 1)  # [batch_size, height, width, channels]
        x = x.reshape(batch_size, self.num_pixels, 3)  # [batch_size, height*width, channels]
        
        # 2. Project RGB values to embedding dimension
        x = self.pixel_embedding(x)  # [batch_size, height*width, d_model]
        
        # 3. Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1+num_pixels, d_model]
        
        # 4. Add positional encoding
        x = self.positional_encoding(x)  # [batch_size, 1+num_pixels, d_model]
        
        # 5. Apply sequence reduction if enabled
        if self.sequence_reduction is not None:
            x = self.sequence_reduction(x)
        
        # 6. Pass through transformer blocks with or without gradient checkpointing
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        # 7. Extract classification token
        x = x[:, 0]  # [batch_size, d_model]
        
        # 8. Classification
        x = self.norm(x)
        x = self.classifier(x)
        
        return x

# Memory-efficient PixelTransformer variant
class MemoryEfficientPixT(nn.Module):
    """
    Memory-efficient variant of PixelTransformer with early spatial downsampling.
    This model drastically reduces memory usage by downsampling the spatial dimensions
    before processing with the transformer, reducing sequence length from 1024 to 256.
    """
    def __init__(self, num_classes=10, d_model=192, nhead=6, num_layers=6, 
                 dim_feedforward=768, dropout=0.1, img_size=32):
        super(MemoryEfficientPixT, self).__init__()
        
        self.img_size = img_size
        self.d_model = d_model
        
        # Initial convolutional layers to reduce spatial dimensions (32x32 -> 16x16)
        self.spatial_reduction = nn.Sequential(
            nn.Conv2d(3, d_model//2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model//2),
            nn.GELU(),
            nn.Conv2d(d_model//2, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        
        # Now we have 16x16 feature map with d_model channels
        # Total sequence length: 16x16 = 256 tokens
        self.reduced_pixels = (img_size // 2) ** 2
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding for reduced sequence length
        self.positional_encoding = PositionalEncoding(d_model, max_len=self.reduced_pixels+1, dropout=dropout)
        
        # Transformer encoder blocks with gradient checkpointing
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize convolution layers
        for m in self.spatial_reduction.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Initialize transformer and linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        # Initialize classification token
        torch.nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Input: [batch_size, channels, height, width]
        batch_size = x.size(0)
        
        # 1. Apply spatial reduction with convolutions
        x = self.spatial_reduction(x)  # [batch_size, d_model, height/2, width/2]
        
        # 2. Reshape to sequence format
        x = x.permute(0, 2, 3, 1)  # [batch_size, height/2, width/2, d_model]
        x = x.reshape(batch_size, self.reduced_pixels, self.d_model)  # [batch_size, reduced_pixels, d_model]
        
        # 3. Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1+reduced_pixels, d_model]
        
        # 4. Add positional encoding
        x = self.positional_encoding(x)
        
        # 5. Apply transformer blocks with gradient checkpointing
        for block in self.transformer_blocks:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        # 6. Extract classification token
        x = x[:, 0]  # [batch_size, d_model]
        
        # 7. Apply classification head
        x = self.norm(x)
        x = self.classifier(x)
        
        return x

def create_pixt_model(num_classes=10, img_size=32, d_model=128, dropout_rate=0.1, config=None):
    """
    Helper function to create a PixT model with specific parameters.
    
    Args:
        num_classes: Number of output classes
        img_size: Size of input images (assumes square images)
        d_model: Size of embedding dimension
        dropout_rate: Dropout rate for regularization
        config: Optional TransformerConfig instance for full customization
        
    Returns:
        PixelTransformer model
    """
    # Use provided config or create one from parameters
    if config is None:
        config = TransformerConfig(
            img_size=img_size,
            d_model=d_model,
            dropout=dropout_rate
        )
    
    # Use the more memory-efficient model if requested
    if hasattr(config, 'use_sequence_downsampling') and config.use_sequence_downsampling and img_size >= 32:
        print(f"Creating Memory-Efficient PixelTransformer for {config.img_size}x{config.img_size} images")
        print(f"  Using spatial downsampling to reduce sequence length")
        print(f"  Model: d_model={config.d_model}, heads={config.nhead}, layers={config.num_layers}")
        
        model = MemoryEfficientPixT(
            num_classes=num_classes,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            img_size=config.img_size
        )
    else:
        print(f"Creating PixelTransformer for {config.img_size}x{config.img_size} images with {config.d_model}-dim embeddings")
        print(f"  Layers: {config.num_layers}, Heads: {config.nhead}, FF dim: {config.dim_feedforward}")
        
        # Add memory optimization details if enabled
        if hasattr(config, 'use_gradient_checkpointing') and config.use_gradient_checkpointing:
            print("  Using gradient checkpointing to reduce memory usage")
        if hasattr(config, 'sequence_reduction_factor') and config.sequence_reduction_factor > 1:
            print(f"  Using sequence reduction with factor {config.sequence_reduction_factor}")
        if hasattr(config, 'share_layer_params') and config.share_layer_params:
            print("  Using parameter sharing between transformer layers")
            
        model = PixelTransformer(
            num_classes=num_classes,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            img_size=config.img_size,
            use_gradient_checkpointing=getattr(config, 'use_gradient_checkpointing', False),
            sequence_reduction_factor=getattr(config, 'sequence_reduction_factor', 1),
            share_layer_params=getattr(config, 'share_layer_params', False),
            use_sequence_downsampling=getattr(config, 'use_sequence_downsampling', False)
        )
    
    return model

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## VT

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.769762Z","iopub.execute_input":"2025-04-19T04:08:43.769951Z","iopub.status.idle":"2025-04-19T04:08:43.799158Z","shell.execute_reply.started":"2025-04-19T04:08:43.769934Z","shell.execute_reply":"2025-04-19T04:08:43.798568Z"}}
class EnhancedResidualBlock(nn.Module):
    """
    Enhanced residual block with more complex feature extraction but
    without spatial downsampling, optimized for small images.
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        
        # Second conv with dilation for larger receptive field
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        
        # Third conv for deeper feature extraction
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Skip connection if dimensions change
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu_out = nn.ReLU(inplace=False)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu_out(out)
        
        return out

class SPDInspiredBlock(nn.Module):
    """
    Block inspired by SPDResNet but without downsampling spatial dimensions.
    Uses grouped convolutions and branch design for complex feature extraction.
    """
    def __init__(self, in_channels, out_channels, groups=4):
        super(SPDInspiredBlock, self).__init__()
        mid_channels = out_channels // 2
        
        # Branch 1: Basic pathway
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, 
                     groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )
        
        # Branch 2: Dilated pathway for larger receptive field
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                     padding=2, dilation=2, groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )
        
        # Combine branches
        self.combine = nn.Sequential(
            nn.Conv2d(mid_channels*2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        
        # Concatenate branches along channel dimension
        combined = torch.cat([branch1, branch2], dim=1)
        out = self.combine(combined)
        
        out += identity
        out = self.relu(out)
        
        return out

class EnhancedSPDBackend(nn.Module):
    """
    Enhanced backend inspired by SPDResNet but preserving spatial dimensions
    for small 32x32 images. Combines multiple types of residual blocks.
    """
    def __init__(self, in_channels=3, out_channels=192):
        super(EnhancedSPDBackend, self).__init__()
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        
        # Layer 1: Standard enhanced residual blocks
        self.layer1 = nn.Sequential(
            EnhancedResidualBlock(64, 96),
            EnhancedResidualBlock(96, 96, dilation=2)
        )
        
        # Layer 2: SPD-inspired blocks
        self.layer2 = nn.Sequential(
            SPDInspiredBlock(96, 128),
            SPDInspiredBlock(128, 160)
        )
        
        # Layer 3: Final feature refinement
        self.layer3 = EnhancedResidualBlock(160, out_channels)
        
        # Final feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        
        # Attention mechanism (SE-like) for channel calibration
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.refinement(x)
        
        # Apply channel attention
        attention = self.channel_attention(x)
        x = x * attention
        
        return x

class FilterTokenizer(nn.Module):
    """
    Filter-based tokenizer that converts feature maps to a smaller set of tokens.
    Uses learned attention weights to aggregate spatial information.
    """
    def __init__(self, C, L):
        super(FilterTokenizer, self).__init__()
        """
        Args:
            C (int): Number of input channels in the feature map.
            L (int): Number of visual tokens to produce (token length).
        """
        self.C = C  # Channel dimension
        self.L = L  # Token length
        
        # Attention projection matrix - computes importance of each spatial position for each token
        self.attention = nn.Conv2d(C, L, kernel_size=1)
        
        # Value projection to compute token features from input features
        self.value_proj = nn.Conv2d(C, C, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map of shape [batch_size, C, H, W]
            
        Returns:
            torch.Tensor: Tokens of shape [batch_size, L, C]
        """
        batch_size, C, H, W = x.shape
        
        # Project input features to values
        values = self.value_proj(x)  # [batch_size, C, H, W]
        
        # Compute attention weights
        attn_weights = self.attention(x)  # [batch_size, L, H, W]
        
        # Reshape and apply softmax across spatial dimensions
        attn_weights = attn_weights.view(batch_size, self.L, -1)  # [batch_size, L, H*W]
        attn_weights = F.softmax(attn_weights, dim=2)  # Softmax across spatial dim
        
        # Reshape values for matrix multiplication
        values = values.view(batch_size, C, -1)  # [batch_size, C, H*W]
        values = values.permute(0, 2, 1)  # [batch_size, H*W, C]
        
        # Compute token features using attention
        tokens = torch.bmm(attn_weights, values)  # [batch_size, L, C]
        
        return tokens

class VisualTransformer(nn.Module):
    """
    Visual Transformer (VT) model optimized for small 32x32 images.
    
    Features:
    1. Enhanced SPD-inspired backend without downsampling to preserve spatial information
    2. Filter-based tokenizer to reduce sequence length to manageable size
    3. Transformer encoder with self-attention for global feature extraction
    4. Uses classification token for final prediction
    """
    def __init__(self, num_classes=10, backend_channels=192, token_length=48, 
                 d_model=192, nhead=8, num_layers=6, dim_feedforward=768, 
                 dropout=0.1, img_size=32, use_gradient_checkpointing=False):
        super(VisualTransformer, self).__init__()
        
        self.img_size = img_size
        self.d_model = d_model
        self.token_length = token_length
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Enhanced SPD-inspired backend without downsampling
        self.backbone = EnhancedSPDBackend(in_channels=3, out_channels=backend_channels)
        
        # Project backend channels to transformer dimension if different
        self.channel_proj = None
        if backend_channels != d_model:
            self.channel_proj = nn.Linear(backend_channels, d_model)
        
        # Filter tokenizer to reduce sequence length
        self.tokenizer = FilterTokenizer(C=backend_channels, L=token_length)
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=token_length + 1,  # +1 for CLS token
            dropout=dropout
        )
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize transformer layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize classification token
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Input: [batch_size, channels, height, width]
        batch_size = x.size(0)
        
        # 1. Process through ResNet backend
        features = self.backbone(x)  # [batch_size, backend_channels, height, width]
        
        # 2. Convert features to tokens with filter tokenizer
        tokens = self.tokenizer(features)  # [batch_size, token_length, backend_channels]
        
        # 3. Project token dimension if needed
        if self.channel_proj is not None:
            tokens = self.channel_proj(tokens)  # [batch_size, token_length, d_model]
        
        # 4. Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_model]
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # [batch_size, 1+token_length, d_model]
        
        # 5. Add positional encoding
        tokens = self.positional_encoding(tokens)  # [batch_size, 1+token_length, d_model]
        
        # 6. Process through transformer blocks
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens)
            else:
                tokens = block(tokens)
        
        # 7. Extract classification token
        cls_token = tokens[:, 0]  # [batch_size, d_model]
        
        # 8. Apply classification head
        cls_token = self.norm(cls_token)
        output = self.classifier(cls_token)
        
        return output

def create_vt_model(num_classes=10, img_size=32, config=None):
    """
    Create a Visual Transformer model with the given configuration.
    
    Args:
        num_classes: Number of output classes
        img_size: Size of input images (assumes square images)
        config: Optional TransformerConfig instance for customization
        
    Returns:
        VisualTransformer model
    """
    # Use provided config or create a default one
    if config is None:
        config = TransformerConfig(
            img_size=img_size,
            d_model=192,
            nhead=8,
            num_layers=6,
            dropout=0.1,
            use_gradient_checkpointing=False
        )
    
    # Determine token length based on image size
    # For 32x32 images, use 48 tokens
    # For larger images, scale accordingly
    token_length = 48
    if img_size > 32:
        # Scale tokens proportionally, but cap at reasonable values
        token_length = min(int(48 * (img_size / 32)), 256)
    
    # Use higher backend channels with the enhanced backbone
    backend_channels = config.d_model
    backend_channels = max(128, backend_channels)  # Ensure at least 128 channels
    
    print(f"Creating Enhanced Visual Transformer for {img_size}x{img_size} images")
    print(f"  Backend: SPD-inspired architecture with {backend_channels} channels")
    print(f"  Tokenizer: Converting to {token_length} tokens")
    print(f"  Transformer: d_model={config.d_model}, heads={config.nhead}, layers={config.num_layers}")
    
    if hasattr(config, 'use_gradient_checkpointing') and config.use_gradient_checkpointing:
        print("  Using gradient checkpointing to reduce memory usage")
    
    model = VisualTransformer(
        num_classes=num_classes,
        backend_channels=backend_channels,
        token_length=token_length,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        img_size=img_size,
        use_gradient_checkpointing=getattr(config, 'use_gradient_checkpointing', False)
    )
    
    return model

# Example factory functions similar to PixT for easy model creation
def create_vt_tiny(num_classes=10, img_size=32):
    """Create a tiny Visual Transformer model."""
    config = TransformerConfig(
        img_size=img_size,
        d_model=96,
        nhead=4,
        num_layers=4,
        dropout=0.1
    )
    return create_vt_model(num_classes, img_size, config)

def create_vt_small(num_classes=10, img_size=32):
    """Create a small Visual Transformer model."""
    config = TransformerConfig(
        img_size=img_size,
        d_model=192,
        nhead=6,
        num_layers=6,
        dropout=0.1
    )
    return create_vt_model(num_classes, img_size, config)

def create_vt_base(num_classes=10, img_size=32):
    """Create a base Visual Transformer model."""
    config = TransformerConfig(
        img_size=img_size,
        d_model=256,
        nhead=8,
        num_layers=8,
        dropout=0.1
    )
    return create_vt_model(num_classes, img_size, config)

def create_vt_memory_efficient(num_classes=10, img_size=32):
    """Create a memory-efficient Visual Transformer model."""
    config = TransformerConfig(
        img_size=img_size,
        d_model=192,
        nhead=6,
        num_layers=6,
        dropout=0.1,
        use_gradient_checkpointing=True
    )
    return create_vt_model(num_classes, img_size, config)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## PatchPixT

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.799773Z","iopub.execute_input":"2025-04-19T04:08:43.799961Z","iopub.status.idle":"2025-04-19T04:08:43.818069Z","shell.execute_reply.started":"2025-04-19T04:08:43.799945Z","shell.execute_reply":"2025-04-19T04:08:43.817487Z"}}
class PatchPixTConfig(TransformerConfig):
    """
    Configuration class for PatchPixT hyperparameters.
    Extends TransformerConfig with patch-specific settings.
    """
    def __init__(self, 
                 img_size=32,
                 patch_size=4,  # Default patch size
                 d_model=128, 
                 nhead=None,
                 num_layers=6,
                 dim_feedforward=None,
                 dropout=0.1,
                 activation="gelu",
                 use_gradient_checkpointing=False,
                 sequence_reduction_factor=1,
                 share_layer_params=False,
                 use_sequence_downsampling=False):
        super().__init__(
            img_size=img_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            use_gradient_checkpointing=use_gradient_checkpointing,
            sequence_reduction_factor=sequence_reduction_factor,
            share_layer_params=share_layer_params,
            use_sequence_downsampling=use_sequence_downsampling
        )
        self.patch_size = patch_size
    
    @classmethod
    def tiny_2x2(cls, img_size=32):
        """Small transformer with 2x2 patches."""
        return cls(img_size=img_size, d_model=96, num_layers=4, dropout=0.1, patch_size=2)
    
    @classmethod
    def small_4x4(cls, img_size=32):
        """Balanced transformer with 4x4 patches."""
        return cls(img_size=img_size, d_model=192, num_layers=6, dropout=0.1, patch_size=4)
    
    @classmethod
    def base_8x8(cls, img_size=32):
        """Larger transformer with 8x8 patches."""
        return cls(img_size=img_size, d_model=256, num_layers=8, dropout=0.1, patch_size=8)

class PatchEmbedding(nn.Module):
    """
    Patch embedding layer that converts image patches to embeddings.
    
    For example, with a patch size of 4, a 32x32 image would be divided into 
    64 patches of size 4x4, each embedded into a vector of dimension d_model.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, d_model=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Make sure img_size is divisible by patch_size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        
        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional layer for patch embedding
        # This projects each patch to d_model dimensions
        self.proj = nn.Conv2d(
            in_channels, 
            d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        # Apply patch projection
        x = self.proj(x)  # [batch_size, d_model, height/patch_size, width/patch_size]
        
        # Reshape to sequence format
        batch_size, d_model, h, w = x.shape
        x = x.flatten(2)  # [batch_size, d_model, height*width/patch_size²]
        x = x.transpose(1, 2)  # [batch_size, height*width/patch_size², d_model]
        
        return x

class PatchPixT(nn.Module):
    """
    Patch Pixel Transformer (PatchPixT) that treats patches of pixels as tokens.
    
    This model divides an image into patches (e.g., 2x2, 4x4, or 8x8) and processes
    each patch as a token, reducing sequence length compared to PixT while
    maintaining spatial information within patches.
    
    Flowchart:
    Input (32x32x3)
    → Patch Embedding (e.g., 64 patches for 4x4 patch size)
    → Positional Encoding
    → Transformer Blocks
    → Classification Token
    → MLP Head
    → Output (num_classes)
    """
    def __init__(self, num_classes=10, img_size=32, patch_size=4, d_model=128, 
                 nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1,
                 use_gradient_checkpointing=False,
                 sequence_reduction_factor=1,
                 share_layer_params=False,
                 use_sequence_downsampling=False):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.share_layer_params = share_layer_params
        
        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            d_model=d_model
        )
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Sequence reduction if enabled
        self.sequence_reduction = None
        if use_sequence_downsampling and sequence_reduction_factor > 1:
            # Apply sequence reduction after adding CLS token and positional encoding
            self.sequence_reduction = SequencePooling(
                d_model, 
                reduction_factor=sequence_reduction_factor,
                mode='mean'
            )
            # Calculate effective sequence length after downsampling
            effective_seq_len = (self.num_patches // sequence_reduction_factor) + 1  # +1 for CLS token
        else:
            effective_seq_len = self.num_patches + 1  # Original length +1 for CLS token
            
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model, 
            max_len=effective_seq_len, 
            dropout=dropout
        )
        
        # Create transformer blocks with parameter sharing if enabled
        if share_layer_params:
            # Create a single transformer block to be reused
            shared_block = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            # Create a list that refers to the same block multiple times
            self.transformer_blocks = nn.ModuleList([shared_block] * num_layers)
        else:
            # Create separate transformer blocks as usual
            self.transformer_blocks = nn.ModuleList([
                TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize classification token
        torch.nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Input: [batch_size, channels, height, width]
        batch_size = x.size(0)
        
        # 1. Apply patch embedding
        x = self.patch_embedding(x)  # [batch_size, num_patches, d_model]
        
        # 2. Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1+num_patches, d_model]
        
        # 3. Add positional encoding
        x = self.positional_encoding(x)  # [batch_size, 1+num_patches, d_model]
        
        # 4. Apply sequence reduction if enabled
        if self.sequence_reduction is not None:
            x = self.sequence_reduction(x)
        
        # 5. Pass through transformer blocks with or without gradient checkpointing
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        # 6. Extract classification token
        x = x[:, 0]  # [batch_size, d_model]
        
        # 7. Classification
        x = self.norm(x)
        x = self.classifier(x)
        
        return x

def create_patchpixt_model(
    num_classes=10, 
    img_size=32, 
    patch_size=4, 
    d_model=128, 
    dropout_rate=0.1, 
    config=None
):
    """
    Helper function to create a PatchPixT model with specific parameters.
    
    Args:
        num_classes: Number of output classes
        img_size: Size of input images (assumes square images)
        patch_size: Size of patches (2, 4, or 8 typically)
        d_model: Size of embedding dimension
        dropout_rate: Dropout rate for regularization
        config: Optional PatchPixTConfig instance for full customization
        
    Returns:
        PatchPixT model
    """
    # Use provided config or create one from parameters
    if config is None:
        config = PatchPixTConfig(
            img_size=img_size,
            patch_size=patch_size,
            d_model=d_model,
            dropout=dropout_rate
        )
    
    print(f"Creating PatchPixT with {config.patch_size}x{config.patch_size} patches for {config.img_size}x{config.img_size} images")
    print(f"  Resulting in {(config.img_size // config.patch_size) ** 2} tokens (plus 1 cls token)")
    print(f"  Model: d_model={config.d_model}, heads={config.nhead}, layers={config.num_layers}")
    
    # Add memory optimization details if enabled
    if config.use_gradient_checkpointing:
        print("  Using gradient checkpointing to reduce memory usage")
    if config.sequence_reduction_factor > 1:
        print(f"  Using sequence reduction with factor {config.sequence_reduction_factor}")
    if config.share_layer_params:
        print("  Using parameter sharing between transformer layers")
        
    model = PatchPixT(
        num_classes=num_classes,
        img_size=config.img_size,
        patch_size=config.patch_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        sequence_reduction_factor=config.sequence_reduction_factor,
        share_layer_params=config.share_layer_params,
        use_sequence_downsampling=config.use_sequence_downsampling
    )
    
    return model

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ##  MultiScalePatchPixT

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.818674Z","iopub.execute_input":"2025-04-19T04:08:43.818863Z","iopub.status.idle":"2025-04-19T04:08:43.842150Z","shell.execute_reply.started":"2025-04-19T04:08:43.818846Z","shell.execute_reply":"2025-04-19T04:08:43.841571Z"}}
class MultiScalePatchPixTConfig(PatchPixTConfig):
    """
    Configuration class for MultiScalePatchPixT hyperparameters.
    Extends PatchPixTConfig to handle multiple patch sizes.
    """
    def __init__(self, 
                 img_size=32,
                 patch_sizes=[2, 4, 8],  # Default patch sizes
                 d_model=128, 
                 nhead=None,
                 num_layers=6,
                 dim_feedforward=None,
                 dropout=0.1,
                 activation="gelu",
                 fusion_type="concat",  # How to fuse features: "concat", "sum", "weighted_sum"
                 use_gradient_checkpointing=False,
                 sequence_reduction_factor=1,
                 share_layer_params=False,
                 share_patch_params=True,  # Whether to share parameters across patch sizes
                 use_sequence_downsampling=False):
        super().__init__(
            img_size=img_size,
            patch_size=min(patch_sizes),  # Just a placeholder
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            use_gradient_checkpointing=use_gradient_checkpointing,
            sequence_reduction_factor=sequence_reduction_factor,
            share_layer_params=share_layer_params,
            use_sequence_downsampling=use_sequence_downsampling
        )
        self.patch_sizes = patch_sizes
        self.fusion_type = fusion_type
        self.share_patch_params = share_patch_params
    
    @classmethod
    def small(cls, img_size=32):
        """Small multi-scale transformer with default patch sizes."""
        return cls(img_size=img_size, d_model=128, num_layers=4, dropout=0.1)
    
    @classmethod
    def base(cls, img_size=32):
        """Larger multi-scale transformer with default patch sizes."""
        return cls(img_size=img_size, d_model=192, num_layers=6, dropout=0.1)

class PatchProcessor(nn.Module):
    """
    Processes image patches of a specific size through transformer layers.
    """
    def __init__(self, img_size, patch_size, d_model, nhead, num_layers, 
                 dim_feedforward, dropout=0.1, use_gradient_checkpointing=False,
                 sequence_reduction_factor=1, share_layer_params=False,
                 use_sequence_downsampling=False):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.share_layer_params = share_layer_params
        
        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            d_model=d_model
        )
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Sequence reduction if enabled
        self.sequence_reduction = None
        if use_sequence_downsampling and sequence_reduction_factor > 1:
            self.sequence_reduction = SequencePooling(
                d_model, 
                reduction_factor=sequence_reduction_factor,
                mode='mean'
            )
            # Calculate effective sequence length after downsampling
            effective_seq_len = (self.num_patches // sequence_reduction_factor) + 1  # +1 for CLS token
        else:
            effective_seq_len = self.num_patches + 1  # Original length +1 for CLS token
            
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model, 
            max_len=effective_seq_len, 
            dropout=dropout
        )
        
        # Create transformer blocks with parameter sharing if enabled
        if share_layer_params:
            # Create a single transformer block to be reused
            shared_block = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            # Create a list that refers to the same block multiple times
            self.transformer_blocks = nn.ModuleList([shared_block] * num_layers)
        else:
            # Create separate transformer blocks as usual
            self.transformer_blocks = nn.ModuleList([
                TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Input: [batch_size, channels, height, width]
        batch_size = x.size(0)
        
        # 1. Apply patch embedding
        x = self.patch_embedding(x)  # [batch_size, num_patches, d_model]
        
        # 2. Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1+num_patches, d_model]
        
        # 3. Add positional encoding
        x = self.positional_encoding(x)  # [batch_size, 1+num_patches, d_model]
        
        # 4. Apply sequence reduction if enabled
        if self.sequence_reduction is not None:
            x = self.sequence_reduction(x)
        
        # 5. Pass through transformer blocks with or without gradient checkpointing
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        # 6. Extract classification token and apply norm
        cls_token = x[:, 0]  # [batch_size, d_model]
        cls_token = self.norm(cls_token)
        
        # Also return the full sequence for optional use
        return cls_token, x

class MultiScalePatchPixT(nn.Module):
    """
    Multi-Scale Patch Pixel Transformer that processes an image using 
    multiple patch sizes simultaneously (e.g., 2x2, 4x4, and 8x8).
    
    This model captures features at different scales and combines them
    for final classification, allowing it to learn both fine-grained details
    and global structures.
    """
    def __init__(self, num_classes=10, img_size=32, patch_sizes=[2, 4, 8], 
                 d_model=128, nhead=8, num_layers=6, dim_feedforward=512, 
                 dropout=0.1, fusion_type="concat", share_patch_params=True,
                 use_gradient_checkpointing=False, sequence_reduction_factor=1,
                 share_layer_params=False, use_sequence_downsampling=False):
        super().__init__()
        
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.d_model = d_model
        self.fusion_type = fusion_type
        
        # Create a patch processor for each patch size
        if share_patch_params:
            # Create a single patch processor and share it across all patch sizes
            self.patch_processors = nn.ModuleList([
                PatchProcessor(
                    img_size=img_size,
                    patch_size=patch_size,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    sequence_reduction_factor=sequence_reduction_factor,
                    share_layer_params=share_layer_params,
                    use_sequence_downsampling=use_sequence_downsampling
                )
                for patch_size in patch_sizes
            ])
        else:
            # Each patch size gets its own processor with independent parameters
            self.patch_processors = nn.ModuleList([
                PatchProcessor(
                    img_size=img_size,
                    patch_size=patch_size,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    sequence_reduction_factor=sequence_reduction_factor,
                    share_layer_params=share_layer_params,
                    use_sequence_downsampling=use_sequence_downsampling
                )
                for patch_size in patch_sizes
            ])
        
        # Feature fusion layer depends on fusion type
        if fusion_type == "concat":
            # Concatenate features from all patch sizes
            fusion_dim = d_model * len(patch_sizes)
            self.fusion_layer = nn.Identity()
        elif fusion_type == "sum":
            # Simple sum of features
            fusion_dim = d_model
            self.fusion_layer = lambda x: torch.sum(torch.stack(x), dim=0)
        elif fusion_type == "weighted_sum":
            # Weighted sum with learnable weights
            fusion_dim = d_model
            self.fusion_weights = nn.Parameter(torch.ones(len(patch_sizes)))
            self.fusion_layer = lambda x: torch.sum(torch.stack([x[i] * self.fusion_weights[i] for i in range(len(x))]), dim=0)
        elif fusion_type == "attention":
            # Cross-attention fusion
            fusion_dim = d_model
            self.fusion_layer = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Process the input through each patch processor
        cls_tokens = []
        full_sequences = []
        
        for processor in self.patch_processors:
            cls_token, full_seq = processor(x)
            cls_tokens.append(cls_token)
            full_sequences.append(full_seq)
        
        # Fuse features based on fusion type
        if self.fusion_type == "concat":
            # Concatenate all CLS tokens
            fused_features = torch.cat(cls_tokens, dim=1)
        elif self.fusion_type == "sum" or self.fusion_type == "weighted_sum":
            # Sum features (possibly weighted)
            fused_features = self.fusion_layer(cls_tokens)
        elif self.fusion_type == "attention":
            # Use the first sequence as query, and all others as keys and values
            # This is a form of cross-attention between different patch scales
            query = cls_tokens[0].unsqueeze(1)  # [batch_size, 1, d_model]
            keys = torch.cat([token.unsqueeze(1) for token in cls_tokens[1:]], dim=1)  # [batch_size, n-1, d_model]
            values = keys
            
            # Apply attention fusion
            fused_features, _ = self.fusion_layer(query, keys, values)
            fused_features = fused_features.squeeze(1)  # [batch_size, d_model]
        
        # Classification
        output = self.classifier(fused_features)
        
        return output

def create_multiscale_patchpixt_model(
    num_classes=10, 
    img_size=32, 
    patch_sizes=[2, 4, 8], 
    d_model=128, 
    dropout_rate=0.1, 
    config=None
):
    """
    Helper function to create a MultiScalePatchPixT model with specific parameters.
    
    Args:
        num_classes: Number of output classes
        img_size: Size of input images (assumes square images)
        patch_sizes: List of patch sizes to use (e.g., [2, 4, 8])
        d_model: Size of embedding dimension
        dropout_rate: Dropout rate for regularization
        config: Optional MultiScalePatchPixTConfig instance for full customization
        
    Returns:
        MultiScalePatchPixT model
    """
    # Use provided config or create one from parameters
    if config is None:
        config = MultiScalePatchPixTConfig(
            img_size=img_size,
            patch_sizes=patch_sizes,
            d_model=d_model,
            dropout=dropout_rate
        )
    
    print(f"Creating MultiScalePatchPixT with patch sizes {config.patch_sizes} for {config.img_size}x{config.img_size} images")
    print(f"  Using fusion type: {config.fusion_type}")
    token_counts = [(config.img_size // size) ** 2 for size in config.patch_sizes]
    print(f"  Token counts per branch: {token_counts} (plus 1 cls token each)")
    print(f"  Model: d_model={config.d_model}, heads={config.nhead}, layers={config.num_layers}")
    
    # Add memory optimization details if enabled
    if config.use_gradient_checkpointing:
        print("  Using gradient checkpointing to reduce memory usage")
    if config.sequence_reduction_factor > 1:
        print(f"  Using sequence reduction with factor {config.sequence_reduction_factor}")
    if config.share_layer_params:
        print("  Using parameter sharing between transformer layers")
    if config.share_patch_params:
        print("  Using parameter sharing between patch sizes")
        
    model = MultiScalePatchPixT(
        num_classes=num_classes,
        img_size=config.img_size,
        patch_sizes=config.patch_sizes,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        fusion_type=config.fusion_type,
        share_patch_params=config.share_patch_params,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        sequence_reduction_factor=config.sequence_reduction_factor,
        share_layer_params=config.share_layer_params,
        use_sequence_downsampling=config.use_sequence_downsampling
    )
    
    return model

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## CNNMultiPatchPixT

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.842775Z","iopub.execute_input":"2025-04-19T04:08:43.842964Z","iopub.status.idle":"2025-04-19T04:08:43.872758Z","shell.execute_reply.started":"2025-04-19T04:08:43.842947Z","shell.execute_reply":"2025-04-19T04:08:43.872137Z"}}
class CNNMultiPatchPixTConfig:
    """
    Configuration class for CNNMultiPatchPixT hyperparameters.
    
    This model combines a CNN backbone with multiple patch-size transformers.
    """
    def __init__(self, 
                 img_size=32,
                 patch_sizes=[1, 2, 4],  # Default patch sizes
                 d_model=128, 
                 nhead=None,  # Will be auto-calculated if None
                 num_layers=6,
                 dim_feedforward=None,  # Will be auto-calculated if None
                 dropout=0.1,
                 activation="gelu",
                 fusion_type="concat",  # How to fuse features: "concat", "weighted_sum", "attention"
                 growth_rate=12,  # Growth rate for DenseNet layers
                 use_gradient_checkpointing=False,
                 share_layer_params=False,
                 cnn_dropout=0.1):  # Dropout for CNN layers
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.d_model = d_model
        # Auto-calculate number of heads if not specified
        self.nhead = nhead if nhead is not None else max(4, d_model // 32)
        self.num_layers = num_layers
        # Auto-calculate feedforward dimension if not specified
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else d_model * 4
        self.dropout = dropout
        self.activation = activation
        self.fusion_type = fusion_type
        self.growth_rate = growth_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.share_layer_params = share_layer_params
        self.cnn_dropout = cnn_dropout
    
    @classmethod
    def small(cls, img_size=32):
        """Small model configuration."""
        return cls(img_size=img_size, d_model=128, num_layers=4, growth_rate=12)
    
    @classmethod
    def base(cls, img_size=32):
        """Standard model configuration."""
        return cls(img_size=img_size, d_model=192, num_layers=6, growth_rate=24)
    
    @classmethod
    def large(cls, img_size=32):
        """Larger model configuration."""
        return cls(img_size=img_size, d_model=256, num_layers=8, growth_rate=32)

class DenseLayer(nn.Module):
    """
    Basic building block for DenseNet.
    """
    def __init__(self, in_channels, growth_rate, dropout=0.1):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    """
    DenseNet block consisting of multiple DenseLayers.
    """
    def __init__(self, in_channels, growth_rate, num_layers, dropout=0.1):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, dropout))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class TransitionBlock(nn.Module):
    """
    Transition layer between DenseNet blocks.
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x

class CNNBackbone(nn.Module):
    """
    DenseNet-like CNN backbone that provides features at multiple scales.
    """
    def __init__(self, growth_rate=12, dropout=0.1):
        super(CNNBackbone, self).__init__()
        
        # Initial convolution to create feature maps
        self.initial_conv = nn.Conv2d(3, 2 * growth_rate, kernel_size=3, padding=1, bias=False)
        
        # Block 1: 32x32 -> 32x32 (maintain spatial dimensions)
        self.block1 = DenseBlock(2 * growth_rate, growth_rate, num_layers=4, dropout=dropout)
        in_channels1 = 2 * growth_rate + 4 * growth_rate
        self.trans1 = TransitionBlock(in_channels1, in_channels1 // 2, dropout=dropout)
        
        # Block 2: 16x16 -> 16x16
        self.block2 = DenseBlock(in_channels1 // 2, growth_rate, num_layers=6, dropout=dropout)
        in_channels2 = in_channels1 // 2 + 6 * growth_rate
        self.trans2 = TransitionBlock(in_channels2, in_channels2 // 2, dropout=dropout)
        
        # Block 3: 8x8 -> 8x8
        self.block3 = DenseBlock(in_channels2 // 2, growth_rate, num_layers=8, dropout=dropout)
        
        # Store output channels for reference
        self.out_channels1 = in_channels1
        self.out_channels2 = in_channels2
        self.out_channels3 = in_channels2 // 2 + 8 * growth_rate
    
    def forward(self, x):
        # Initial convolution: (B, 3, 32, 32) -> (B, 2*growth_rate, 32, 32)
        x = self.initial_conv(x)
        
        # Block 1
        x1 = self.block1(x)  # (B, out_channels1, 32, 32)
        x = self.trans1(x1)  # (B, out_channels1//2, 16, 16)
        
        # Block 2
        x2 = self.block2(x)  # (B, out_channels2, 16, 16)
        x = self.trans2(x2)  # (B, out_channels2//2, 8, 8)
        
        # Block 3
        x3 = self.block3(x)  # (B, out_channels3, 8, 8)
        
        return x1, x2, x3

class CNNMultiPatchPixT(nn.Module):
    """
    CNN-backed Multi-Scale Patch Transformer with simplified architecture.
    
    This model uses a DenseNet-like CNN backbone to extract features at multiple 
    scales, applies uniform non-overlapping patching to each feature map, and
    directly fuses these features without branch-specific transformer layers.
    """
    def __init__(self, num_classes=10, img_size=32, patch_sizes=[1, 2, 4], 
                 d_model=128, nhead=8, num_layers=6, dim_feedforward=512, 
                 dropout=0.1, fusion_type="concat", growth_rate=12, 
                 use_gradient_checkpointing=False, share_layer_params=False,
                 cnn_dropout=0.1):
        super(CNNMultiPatchPixT, self).__init__()
        
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.d_model = d_model
        self.fusion_type = fusion_type
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # 1. CNN Backbone for extracting multi-scale features
        self.backbone = CNNBackbone(growth_rate=growth_rate, dropout=cnn_dropout)
        
        # 2. Patch embeddings for each branch - will extract non-overlapping patches
        # and project them to d_model dimension
        self.patch_embeddings = nn.ModuleList([
            # Branch 1: 4×4 patches from 32×32 feature map -> 64 tokens
            nn.Conv2d(self.backbone.out_channels1, d_model, kernel_size=4, stride=4, padding=0),
            
            # Branch 2: 2×2 patches from 16×16 feature map -> 64 tokens
            nn.Conv2d(self.backbone.out_channels2, d_model, kernel_size=2, stride=2, padding=0),
            
            # Branch 3: 1×1 patches from 8×8 feature map -> 64 tokens
            nn.Conv2d(self.backbone.out_channels3, d_model, kernel_size=1, stride=1, padding=0)
        ])
        
        # 3. CLS tokens for each branch
        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, d_model))
            for _ in range(3)  # 3 branches
        ])
        
        # 4. All branches will have 64 tokens + 1 CLS token = 65 tokens
        self.seq_length = 65
        
        # 5. Positional encodings for all branches (all have same sequence length)
        self.positional_encoding = PositionalEncoding(d_model, max_len=self.seq_length, dropout=dropout)
        
        # 6. Feature fusion mechanism
        if fusion_type == "concat":
            # Concatenate features from all branches
            fusion_dim = d_model * 3  # 3 branches
            self.fusion_layer = nn.Linear(fusion_dim, d_model)
        elif fusion_type == "attention":
            # Use attention to weight different branches
            self.fusion_layer = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
        elif fusion_type == "weighted_sum":
            # Weighted sum with learnable weights
            self.fusion_weights = nn.Parameter(torch.ones(3))  # 3 branches
            fusion_dim = d_model
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # 7. Shared transformer layers for fused features
        if share_layer_params:
            shared_block = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            self.transformer_blocks = nn.ModuleList([shared_block] * num_layers)
        else:
            self.transformer_blocks = nn.ModuleList([
                TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])
        
        # 8. Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # 9. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize linear layers and convolutions
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Extract multi-scale features from CNN backbone
        x1, x2, x3 = self.backbone(x)
        
        # List of features with corresponding patch embeddings
        features = [x1, x2, x3]
        
        branch_outputs = []
        
        # 2. Process each branch without transformer layers
        for i, (feature, patch_embedding, cls_token) in enumerate(
            zip(features, self.patch_embeddings, self.cls_tokens)
        ):
            # Apply patch embedding - this creates 64 tokens for each branch
            # through non-overlapping patches (4×4, 2×2, and 1×1)
            embedded = patch_embedding(feature)  # [batch_size, d_model, 8, 8]
            
            # Reshape to sequence format
            embedded = embedded.flatten(2).transpose(1, 2)  # [batch_size, 64, d_model]
            
            # Add classification token
            cls_tokens = cls_token.expand(batch_size, -1, -1)
            embedded = torch.cat((cls_tokens, embedded), dim=1)  # [batch_size, 65, d_model]
            
            # Add positional encoding
            embedded = self.positional_encoding(embedded)
            
            # Extract CLS token for this branch
            branch_cls = embedded[:, 0]  # [batch_size, d_model]
            branch_outputs.append(branch_cls)
        
        # 3. Fuse features from all branches
        if self.fusion_type == "concat":
            # Concatenate branch outputs
            fused = torch.cat(branch_outputs, dim=1)  # [batch_size, 3*d_model]
            fused = self.fusion_layer(fused)  # [batch_size, d_model]
        elif self.fusion_type == "attention":
            # Use first branch output as query
            query = branch_outputs[0].unsqueeze(1)  # [batch_size, 1, d_model]
            # Concatenate other branch outputs as keys and values
            keys = torch.stack(branch_outputs[1:], dim=1)  # [batch_size, 2, d_model]
            fused, _ = self.fusion_layer(query, keys, keys)
            fused = fused.squeeze(1)  # [batch_size, d_model]
        elif self.fusion_type == "weighted_sum":
            # Weighted sum with learnable weights
            weighted_outputs = [output * weight for output, weight in zip(branch_outputs, self.fusion_weights)]
            fused = sum(weighted_outputs)  # [batch_size, d_model]
        
        # 4. Process the fused representation
        fused = fused.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 5. Process through transformer blocks
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                fused = torch.utils.checkpoint.checkpoint(block, fused)
            else:
                fused = block(fused)
        
        # 6. Extract final representation
        fused = fused.squeeze(1)  # [batch_size, d_model]
        fused = self.norm(fused)
        
        # 7. Classification
        output = self.classifier(fused)
        
        return output

def create_cnn_multipatch_pixt_model(
    num_classes=10,
    img_size=32,
    patch_sizes=[1, 2, 4],
    d_model=128,
    dropout_rate=0.1,
    config=None
):
    """
    Helper function to create a CNNMultiPatchPixT model with specific parameters.
    
    Args:
        num_classes: Number of output classes
        img_size: Size of input images (assumes square images)
        patch_sizes: List of patch sizes for different branches
        d_model: Size of embedding dimension
        dropout_rate: Dropout rate for regularization
        config: Optional CNNMultiPatchPixTConfig instance for full customization
        
    Returns:
        CNNMultiPatchPixT model
    """
    # Use provided config or create one from parameters
    if config is None:
        config = CNNMultiPatchPixTConfig(
            img_size=img_size,
            patch_sizes=patch_sizes,
            d_model=d_model,
            dropout=dropout_rate
        )
    
    print(f"Creating CNNMultiPatchPixT model with CNN backbone and patch sizes {config.patch_sizes}")
    print(f"  CNN backbone with growth rate: {config.growth_rate}")
    print(f"  Transformer: d_model={config.d_model}, heads={config.nhead}, layers={config.num_layers}")
    print(f"  Using fusion type: {config.fusion_type}")
    
    # Print memory optimization details if enabled
    if config.use_gradient_checkpointing:
        print("  Using gradient checkpointing to reduce memory usage")
    if config.share_layer_params:
        print("  Using parameter sharing between transformer layers")
    
    model = CNNMultiPatchPixT(
        num_classes=num_classes,
        img_size=config.img_size,
        patch_sizes=config.patch_sizes,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        fusion_type=config.fusion_type,
        growth_rate=config.growth_rate,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        share_layer_params=config.share_layer_params,
        cnn_dropout=config.cnn_dropout
    )
    
    return model

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## TarloyIR

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.873352Z","iopub.execute_input":"2025-04-19T04:08:43.873555Z","iopub.status.idle":"2025-04-19T04:08:43.892982Z","shell.execute_reply.started":"2025-04-19T04:08:43.873524Z","shell.execute_reply":"2025-04-19T04:08:43.892410Z"}}
class TaylorWindowAttention(nn.Module):
    """
    Implements Direct-TaylorShift attention which uses a Taylor series approximation
    instead of the standard softmax for more efficient attention calculation.
    """
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(head_dim)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, num_heads, N, head_dim]
        
        # Apply scaling to query
        q = q * self.scale
        
        # Compute attention scores: A = QK^T / sqrt(d)
        A = q @ k.transpose(-2, -1)  # [B, num_heads, N, N]
        
        # Taylor-Softmax: (1 + A + 0.5 * A^2) / sum(1 + A + 0.5 * A^2)
        # More efficient than standard softmax for relatively small attention scores
        A2 = A ** 2
        numerator = 1 + A + 0.5 * A2
        denominator = numerator.sum(dim=-1, keepdim=True)  # Sum over last dimension
        attn = numerator / denominator  # [B, num_heads, N, N]
        attn = self.attn_drop(attn)
        
        # Apply attention weights to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TaylorTransformerBlock(nn.Module):
    """
    Transformer block using TaylorWindowAttention with pre-norm architecture.
    """
    def __init__(self, dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Taylor window attention
        self.attn = TaylorWindowAttention(dim, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TaylorIRClassifier(nn.Module):
    """
    Classification model using TaylorWindowAttention for 32x32x3 images.
    This model uses a convolutional feature extractor followed by Taylor-approximated
    transformer blocks for efficient global reasoning.
    """
    def __init__(self, num_classes=4, img_size=32, embed_dim=192, num_heads=6, 
                 num_layers=6, dim_feedforward=768, dropout=0.1, 
                 use_gradient_checkpointing=False):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Feature extraction with a simple stem (preserves spatial dimensions)
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # Image to sequence conversion (flatten spatial dimensions)
        self.num_patches = img_size * img_size
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            embed_dim, 
            max_len=self.num_patches + 1,  # +1 for CLS token
            dropout=dropout
        )
        
        # Transformer blocks with Taylor window attention
        self.transformer_blocks = nn.ModuleList([
            TaylorTransformerBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize linear layers and convolutions
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize classification token
        torch.nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Input: [batch_size, 3, H, W]
        batch_size = x.shape[0]
        
        # Feature extraction
        x = self.stem(x)  # [batch_size, embed_dim, H, W]
        
        # Reshape to sequence format
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [batch_size, H*W, embed_dim]
        
        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1+H*W, embed_dim]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        # Extract classification token
        x = x[:, 0]  # [batch_size, embed_dim]
        
        # Layer normalization and classification
        x = self.norm(x)
        x = self.classifier(x)
        
        return x

class TaylorConfig:
    """
    Configuration class for TaylorIR hyperparameters.
    """
    def __init__(self, 
                 img_size=32,
                 embed_dim=192, 
                 num_heads=6,
                 num_layers=6,
                 dim_feedforward=768,
                 dropout=0.1,
                 use_gradient_checkpointing=False):
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.use_gradient_checkpointing = use_gradient_checkpointing
    
    @classmethod
    def small(cls, img_size=32):
        """Small model configuration."""
        return cls(img_size=img_size, embed_dim=128, num_layers=4, num_heads=4)
    
    @classmethod
    def base(cls, img_size=32):
        """Standard model configuration."""
        return cls(img_size=img_size, embed_dim=192, num_layers=6, num_heads=6)
    
    @classmethod
    def large(cls, img_size=32):
        """Larger model configuration."""
        return cls(img_size=img_size, embed_dim=256, num_layers=8, num_heads=8)

def create_taylorir_model(num_classes=4, img_size=32, config=None):
    """
    Helper function to create a TaylorIR model with specific parameters.
    
    Args:
        num_classes: Number of output classes
        img_size: Size of input images (assumes square images)
        config: Optional TaylorConfig instance for full customization
        
    Returns:
        TaylorIRClassifier model
    """
    # Use provided config or create one from parameters
    if config is None:
        config = TaylorConfig(img_size=img_size)
    
    print(f"Creating TaylorIR classifier for {config.img_size}x{config.img_size} images")
    print(f"  Model: embed_dim={config.embed_dim}, heads={config.num_heads}, layers={config.num_layers}")
    
    # Add memory optimization details if enabled
    if config.use_gradient_checkpointing:
        print("  Using gradient checkpointing to reduce memory usage")
    
    model = TaylorIRClassifier(
        num_classes=num_classes,
        img_size=config.img_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        use_gradient_checkpointing=config.use_gradient_checkpointing
    )
    
    return model

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## AstroT

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.893511Z","iopub.execute_input":"2025-04-19T04:08:43.893710Z","iopub.status.idle":"2025-04-19T04:08:43.911070Z","shell.execute_reply.started":"2025-04-19T04:08:43.893693Z","shell.execute_reply":"2025-04-19T04:08:43.910495Z"}}
class RelativeAttention(nn.Module):
    """
    Attention mechanism with relative position bias using depthwise convolutions.
    This follows the implementation in Sample.py.
    """
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.proj_qkv = nn.Linear(in_channels, 3 * in_channels)
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                       padding=kernel_size//2, groups=in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C] where N = H * W
        qkv = self.proj_qkv(x_flat).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # [3, B, N, C]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, N, C]
        
        # Standard attention
        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, N, N]
        
        # Relative Position Bias
        relative_bias = self.depthwise_conv(x).flatten(2).transpose(1, 2)  # [B, N, C]
        relative_bias_attn = torch.matmul(q, relative_bias.transpose(-2, -1))  # [B, N, N]
        attn = attn + relative_bias_attn  # Combine standard and relative attention
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out_flat = torch.matmul(attn, v)  # [B, N, C]
        out = out_flat.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        return out

class StageBlock(nn.Module):
    """
    Block used within stages, consisting of depthwise and expansion convolutions.
    This follows the implementation in Sample.py but maintains spatial dimensions.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Depthwise Convolution (D-Conv)
        self.d_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, 
            padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Expansion Convolution (E-Conv)
        self.e_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip Connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.d_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.e_conv(x)
        x = self.bn2(x)
        x = x + identity
        x = self.relu(x)
        return x

class AstroTransformer(nn.Module):
    """
    Adapted from Sample.py's architecture for 32x32 images and 4-class classification.
    
    This version maintains the full 32x32 spatial resolution throughout the network,
    which is more appropriate for small images where spatial information is critical.
    
    Each stage processes features at different channel depths but maintains spatial dimensions,
    with skip connections between stages.
    """
    def __init__(self, num_classes=4, expansion=2, layers=[2, 2, 2]):
        super().__init__()
        
        # Stem Stage (S0) - maintain spatial size for small input
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.expansion = expansion
        self.layers = layers  # Number of blocks per stage: [L1, L2, L3]

        # Stage S1 - in:32 -> out:64 (for expansion=2), maintain 32x32 size
        self.s1 = self._make_stage(32, layers[0], stride=1)  # Use stride 1 to maintain spatial size
        self.adjust_s0_to_s1 = nn.Conv2d(32, expansion * 32, kernel_size=1, stride=1, bias=False)
        
        # Stage S2 - in:64 -> out:128 (for expansion=2), maintain 32x32 size
        self.s2 = self._make_stage(expansion * 32, layers[1], stride=1)
        self.adjust_s1_to_s2 = nn.Conv2d(expansion * 32, expansion**2 * 32, kernel_size=1, stride=1, bias=False)
        
        # Stage S3 - in:128 -> out:256 (for expansion=2), maintain 32x32 size
        self.s3 = self._make_stage(expansion**2 * 32, layers[2], stride=1)
        # No additional adjustment needed
        
        # Relative Attention after S3 (operating on 32x32 feature maps)
        self.relative_attention = RelativeAttention(expansion**3 * 32)

        # Global Average Pooling + Classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(expansion**3 * 32, num_classes)
    
    def _make_stage(self, in_channels, num_blocks, stride=1):
        """Create a stage with num_blocks blocks, all using stride 1 to maintain spatial dimensions."""
        out_channels = self.expansion * in_channels
        
        # All blocks use stride 1 to maintain spatial dimensions
        blocks = [StageBlock(in_channels, out_channels, stride=stride)]
        
        # Additional blocks
        for _ in range(1, num_blocks):
            blocks.append(StageBlock(out_channels, out_channels, stride=1))
            
        return nn.Sequential(*blocks)

    def forward(self, x):
        # Input: 32x32x3
        x = self.stem(x)  # Output: 32x32x32
        s0_out = x
        
        # Stage S1
        x = self.s1(x)  # Output: 32x32x64
        adjusted_s0 = self.adjust_s0_to_s1(s0_out)  # Adjust S0: 32x32x32 -> 32x32x64
        x = x + adjusted_s0
        s1_out = x
        
        # Stage S2
        x = self.s2(x)  # Output: 32x32x128
        adjusted_s1 = self.adjust_s1_to_s2(s1_out)  # Adjust S1: 32x32x64 -> 32x32x128
        x = x + adjusted_s1
        s2_out = x
        
        # Stage S3
        x = self.s3(x)  # Output: 32x32x256
        
        # Relative Attention
        x = self.relative_attention(x)  # Output: 32x32x256
        
        # Global Average Pooling
        x = self.global_avg_pool(x)  # Output: 1x1x256
        
        # Flatten and FC Layer
        x = torch.flatten(x, 1)  # Output: 256
        x = self.fc(x)  # Output: num_classes
        
        return x

class AstroConfig:
    """
    Configuration class for AstroTransformer hyperparameters.
    """
    def __init__(self, 
                 expansion=2,
                 layers=[2, 2, 2],
                 use_gradient_checkpointing=False):
        self.expansion = expansion
        self.layers = layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
    
    @classmethod
    def small(cls):
        """Small model configuration."""
        return cls(expansion=2, layers=[1, 1, 1])
    
    @classmethod
    def base(cls):
        """Standard model configuration."""
        return cls(expansion=2, layers=[2, 2, 2])
    
    @classmethod
    def large(cls):
        """Larger model configuration."""
        return cls(expansion=3, layers=[2, 3, 3])

def create_astro_model(num_classes=4, config=None):
    """
    Helper function to create an AstroTransformer model with specific parameters.
    
    Args:
        num_classes: Number of output classes
        config: Optional AstroConfig instance for full customization
        
    Returns:
        AstroTransformer model
    """
    # Use provided config or create one from parameters
    if config is None:
        config = AstroConfig()
    
    print(f"Creating AstroTransformer for 32x32 images")
    print(f"  Architecture: {len(config.layers)} stages with {config.layers} blocks")
    print(f"  Expansion factor: {config.expansion}")
    
    # Add memory optimization details if enabled
    if config.use_gradient_checkpointing:
        print("  Using gradient checkpointing to reduce memory usage")
    
    model = AstroTransformer(
        num_classes=num_classes,
        expansion=config.expansion,
        layers=config.layers
    )
    
    return model

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Model utils

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.911670Z","iopub.execute_input":"2025-04-19T04:08:43.911858Z","iopub.status.idle":"2025-04-19T04:08:43.935944Z","shell.execute_reply.started":"2025-04-19T04:08:43.911841Z","shell.execute_reply":"2025-04-19T04:08:43.935401Z"}}
def get_model(num_classes, config, device):
    """
    Use a model suited for the image size with architecture specified in config.
    Wrap with DataParallel if multiple GPUs are to be used.
    """
    model_type = getattr(config, 'model_type', 'densenet')
    
    # Create the base model based on model_type
    if model_type == 'lownet':
        print("Creating LowNet model for low-resolution image feature extraction...")
        model = AdaptedLowNet(num_classes=num_classes, 
                             dropout_rate=config.dropout_rate)
    elif model_type == 'spdresnet':
        print("Creating SPDResNet model with Space-to-Depth downsampling...")
        print("Using SPDResNet-18 architecture for small images")
        model = spdresnet18(num_classes=num_classes, dropout_rate=config.dropout_rate)
    elif model_type == 'inceptionfsd':
        print("Creating InceptionFSD model with multi-scale feature extraction...")
        model = InceptionFSD(num_classes=num_classes, 
                            dropout_rate=config.dropout_rate)
    elif model_type == 'spddualbranch':
        print("Creating SPDDualBranch model with Space-to-Depth downsampling in both branches...")
        model = SPDDualBranchNetwork(num_classes=num_classes,
                                   dropout_rate=config.dropout_rate)
    elif model_type == 'dilatedgroupconv':
        print("Creating DilatedGroupConvNet with dilated convolutions...")
        model = DilatedGroupConvNet(num_classes=num_classes, 
                                dropout_rate=config.dropout_rate)
    elif model_type == 'dual_branch':
        print("Creating Dual-Branch Network with common feature subspace...")
        model = DualBranchNetwork(num_classes=num_classes, 
                                 dropout_rate=config.dropout_rate)
    elif model_type == 'rlnet':
        print("Creating RL-Net with Multi-Kernel blocks and multi-scale feature extraction...")
        model = create_rlnet(num_classes=num_classes, 
                           input_channels=3,
                           dropout_rate=config.dropout_rate)
    elif model_type == 'rlspdnet':
        print("Creating RLSPDNet with Multi-Kernel blocks and Space-to-Depth downsampling...")
        model = create_rlspdnet(num_classes=num_classes, 
                             input_channels=3,
                             dropout_rate=config.dropout_rate)
    elif model_type == 'densenet':
        # Use DenseNet explicitly when requested
        print("Creating DenseNet7x7 model with 7x7 kernels...")
        model = DenseNet7x7(growth_rate=16, 
                            block_config=(3, 6, 12, 8),
                            num_classes=num_classes, 
                            dropout_rate=config.dropout_rate)
    elif model_type == 'smallresnet':
        print("Creating SmallResNet model...")
        model = SmallResNet(num_classes=num_classes, 
                           dropout_rate=config.dropout_rate)
    elif model_type == 'minixception':
        print("Creating MiniXception model with SPD downsampling and separable convolutions...")
        model = MiniXception(num_classes=num_classes, 
                           dropout_rate=config.dropout_rate)
    elif model_type == 'mixmodel1':
        print("Creating MixModel1 with SPDConv, Residual Inception blocks, and SE attention...")
        model = MixModel1(num_classes=num_classes, 
                         dropout_rate=config.dropout_rate)
    elif model_type == 'mixmodel2':
        print("Creating MixModel2 with multi-pathways and skip connections...")
        model = MixModel2(num_classes=num_classes, 
                         dropout_rate=config.dropout_rate)
    elif model_type == 'mixmodel3':
        print("Creating MixModel3 that maintains 32x32 resolution throughout the network...")
        model = MixModel3(num_classes=num_classes, 
                         dropout_rate=config.dropout_rate)
    elif model_type == 'mixmodel4':
        print("Creating MixModel4 with multi-branch spatial feature extraction at different scales...")
        model = MixModel4(num_classes=num_classes, 
                         dropout_rate=config.dropout_rate)
    elif model_type == 'trans':
        print("Creating Transformer model for image classification...")
        transformer_type = getattr(config, 'transformer_type', 'pixt')
        
        # Create transformer configuration using direct parameters
        transformer_config = TransformerConfig(
            img_size=config.image_size,
            d_model=config.transformer_d_model,
            nhead=config.transformer_nhead,
            num_layers=config.transformer_num_layers,
            dim_feedforward=getattr(config, 'transformer_dim_feedforward', config.transformer_d_model * 4),
            dropout=config.transformer_dropout_rate,
            use_gradient_checkpointing=config.transformer_use_gradient_checkpointing,
            sequence_reduction_factor=config.transformer_sequence_reduction_factor,
            share_layer_params=config.transformer_share_layer_params,
            use_sequence_downsampling=getattr(config, 'transformer_use_sequence_downsampling', False)
        )
        
        # Create the appropriate transformer model based on type
        if transformer_type == 'vt':
            model = create_vt_model(
                num_classes=num_classes,
                img_size=config.image_size,
                config=transformer_config
            )
        elif transformer_type == "patchpixt":
            # Create PatchPixT with directly specified parameters
            patch_size = getattr(config, 'transformer_patch_size', 4)
            
            # Create PatchPixT config with manual parameters
            patch_config = PatchPixTConfig(
                img_size=config.image_size,
                patch_size=patch_size,
                d_model=config.transformer_d_model,
                nhead=config.transformer_nhead,
                num_layers=config.transformer_num_layers,
                dim_feedforward=getattr(config, 'transformer_dim_feedforward', config.transformer_d_model * 4),
                dropout=config.transformer_dropout_rate,
                use_gradient_checkpointing=config.transformer_use_gradient_checkpointing,
                sequence_reduction_factor=config.transformer_sequence_reduction_factor,
                share_layer_params=config.transformer_share_layer_params,
                use_sequence_downsampling=getattr(config, 'transformer_use_sequence_downsampling', False)
            )
            
            model = create_patchpixt_model(
                num_classes=num_classes,
                img_size=config.image_size,
                patch_size=patch_size,
                config=patch_config
            )
        elif transformer_type == "cnn_multipatch_pixt":
            # Create CNNMultiPatchPixT model with CNN backbone and multiple patch sizes
            patch_sizes = getattr(config, 'transformer_patch_sizes', [1, 2, 4])
            fusion_type = getattr(config, 'transformer_fusion_type', 'concat')
            growth_rate = getattr(config, 'transformer_growth_rate', 12)
            
            # Create CNNMultiPatchPixT config
            cnn_multipatch_config = CNNMultiPatchPixTConfig(
                img_size=config.image_size,
                patch_sizes=patch_sizes,
                d_model=config.transformer_d_model,
                nhead=config.transformer_nhead,
                num_layers=config.transformer_num_layers,
                dim_feedforward=getattr(config, 'transformer_dim_feedforward', config.transformer_d_model * 4),
                dropout=config.transformer_dropout_rate,
                fusion_type=fusion_type,
                growth_rate=growth_rate,
                use_gradient_checkpointing=config.transformer_use_gradient_checkpointing,
                share_layer_params=config.transformer_share_layer_params,
                cnn_dropout=config.dropout_rate
            )
            
            model = create_cnn_multipatch_pixt_model(
                num_classes=num_classes,
                img_size=config.image_size,
                patch_sizes=patch_sizes,
                config=cnn_multipatch_config
            )
        elif transformer_type == "taylorir":
            # Create TaylorIR model with specified parameters
            taylor_config = TaylorConfig(
                img_size=config.image_size,
                embed_dim=config.transformer_d_model,
                num_heads=config.transformer_nhead,
                num_layers=config.transformer_num_layers,
                dim_feedforward=getattr(config, 'transformer_dim_feedforward', config.transformer_d_model * 4),
                dropout=config.transformer_dropout_rate,
                use_gradient_checkpointing=config.transformer_use_gradient_checkpointing
            )
            
            model = create_taylorir_model(
                num_classes=num_classes,
                img_size=config.image_size,
                config=taylor_config
            )
        elif transformer_type == "astro":
            # Create AstroTransformer model with Sample.py-based implementation
            astro_config = AstroConfig(
                expansion=getattr(config, 'astro_expansion', 2),
                layers=getattr(config, 'astro_layers', [2, 2, 2]),
                use_gradient_checkpointing=config.transformer_use_gradient_checkpointing
            )
            
            model = create_astro_model(
                num_classes=num_classes,
                config=astro_config
            )
        elif transformer_type == "pixt":  # Default to 'pixt'
            model = create_pixt_model(
                num_classes=num_classes,
                config=transformer_config
            )
        else: 
            print("Unknown transformer type, stop to save your time")
            raise ValueError(f"Unknown transformer type: {transformer_type}")
    else:
        print("Unknown model type, stop to save your time")
        raise ValueError(f"Unknown model type: {model_type}")

    # Move model to device first
    model = model.to(device)

    # Wrap with DataParallel if using multiple GPUs
    if hasattr(config, 'use_multi_gpu') and config.use_multi_gpu:
        # Get available GPU devices
        if hasattr(config, 'gpu_ids') and config.gpu_ids:
            gpu_ids = config.gpu_ids
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        
        if len(gpu_ids) > 1:
            print(f"Using DataParallel with GPUs: {gpu_ids}")
            model = nn.DataParallel(model, device_ids=gpu_ids)
    
    return model


def get_layer_wise_lr_optimizer(model, config):
    """Creates an Adam optimizer with layer-wise learning rates or a regular optimizer for custom models."""
    # Print model structure for debugging
    print("Model Structure:")
    
    # Check if model is a Sequential model (our custom CNN)
    if isinstance(model, nn.Sequential):
        print("  Using standard optimizer for Sequential model")
        # For Sequential models, use standard optimizer without layer-wise LR
        return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
    
    # For models with named modules like MobileNetV3
    # Group parameters by module type for better organization
    feature_params = []
    classifier_params = []
    other_params = []  # Add a catch-all for other parameters
    
    # Keep track of parameters we've added to groups
    param_set = set()
    
    # Organize parameters by layer type
    for name, module in model.named_children():
        print(f"Layer group: {name}")
        
        # Skip modules with no parameters
        if sum(1 for _ in module.parameters()) == 0:
            print(f"  {name}: Skipping - no trainable parameters")
            continue
        
        if name == 'features':
            # Apply gradually decreasing LR to feature layers
            for i, layer in enumerate(module):
                # Skip layers with no parameters
                if sum(1 for p in layer.parameters() if p.requires_grad) == 0:
                    continue
                    
                layer_lr = config.learning_rate * (config.layer_decay_rate ** i)
                print(f"  Feature block {i}: lr = {layer_lr:.6f}")
                
                # Only include parameters that require gradients
                params = [p for p in layer.parameters() if p.requires_grad]
                if params:  # Only add if there are actually parameters
                    feature_params.append({'params': params, 'lr': layer_lr})
                    param_set.update(params)  # Add to our tracking set
                    
        elif name == 'classifier':
            # Use base LR for classifier (final layers)
            # Only include parameters that require gradients
            params = [p for p in module.parameters() if p.requires_grad]
            if params:  # Check if there are any trainable parameters
                print(f"  Classifier: lr = {config.learning_rate}")
                classifier_params.append({'params': params, 'lr': config.learning_rate})
                param_set.update(params)  # Add to our tracking set
        else:
            # Other layers - collect their parameters too
            params = [p for p in module.parameters() if p.requires_grad]
            if params:  # Check if there are any trainable parameters
                print(f"  {name}: lr = {config.learning_rate}")
                other_params.append({'params': params, 'lr': config.learning_rate})
                param_set.update(params)  # Add to our tracking set
    
    # Combine all parameter groups
    param_groups = feature_params + classifier_params + other_params
    
    # Check for parameters that weren't included in any group
    all_params = set(p for p in model.parameters() if p.requires_grad)
    missed_params = all_params - param_set
    if missed_params:
        print(f"  Found {len(missed_params)} parameters not assigned to any group, adding with base LR")
        other_params.append({'params': list(missed_params), 'lr': config.learning_rate})
    
    # If param_groups is still empty, fall back to using all model parameters
    if len(param_groups) == 0:
        print("  No parameters found in named children, using all model parameters")
        return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
    
    # Create optimizer with parameter groups
    return optim.Adam(param_groups, lr=config.learning_rate, weight_decay=config.l2_reg)


def load_model_from_checkpoint(checkpoint_path, num_classes, config, device):
    """Load a model from a checkpoint file with support for DataParallel."""
    # Create base model on specified device
    model = get_model(num_classes, config, device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats and DataParallel prefixes
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle case where model was saved with DataParallel but is now loaded without it
    if not isinstance(model, nn.DataParallel) and all(k.startswith('module.') for k in state_dict.keys()):
        # Remove 'module.' prefix from state_dict keys
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix (7 characters)
        state_dict = new_state_dict
    
    # Handle case where model was saved without DataParallel but is now loaded with it
    elif isinstance(model, nn.DataParallel) and not all(k.startswith('module.') for k in state_dict.keys()):
        # Add 'module.' prefix to state_dict keys
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict['module.' + k] = v
        state_dict = new_state_dict
    
    # Load the state dict
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        # Try to load with strict=False which will ignore missing keys
        print("Attempting to load with strict=False")
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    return model

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Training

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.936588Z","iopub.execute_input":"2025-04-19T04:08:43.936780Z","iopub.status.idle":"2025-04-19T04:08:43.987568Z","shell.execute_reply.started":"2025-04-19T04:08:43.936763Z","shell.execute_reply":"2025-04-19T04:08:43.986938Z"}}
def set_seed(seed=SEED):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, output_dir):
    """Train with mixup, early stopping, scheduler, and optimizations."""
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # Add early stopping warmup default if not in config
    early_stopping_warmup = getattr(config, 'early_stopping_warmup', 5)
    
    # Check saving behavior configuration
    save_last_model = getattr(config, 'save_last_model', False)
    save_only_at_end = getattr(config, 'save_only_at_end', False)
    
    if save_only_at_end:
        print("Note: Configured to save model only at the end of training to reduce I/O operations")
    elif save_last_model:
        print("Note: Configured to save the last model instead of the best validation model")
    
    # History tracking for plotting
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize gradient scaler for AMP
    scaler = GradScaler() if config.use_amp else None
    
    # Check if using a transformer model with AMP
    is_transformer = isinstance(model, (PixelTransformer, MemoryEfficientPixT, VisualTransformer)) or 'PixT' in str(type(model)) or 'VT' in str(type(model))
    use_transformer_amp = is_transformer and hasattr(config, 'transformer_use_amp') and config.transformer_use_amp
    if use_transformer_amp and not config.use_amp:
        print("Enabling AMP specifically for transformer model")
        scaler = GradScaler()
    
    # Training loop
    for epoch in range(config.num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Mixup augmentation
            use_mixup = np.random.rand() < config.mixup_prob
            if use_mixup:
                lam = np.random.beta(config.mixup_alpha, config.mixup_alpha)
                index = torch.randperm(inputs.size(0)).to(device)
                inputs_mix = lam * inputs + (1 - lam) * inputs[index]
                labels_a, labels_b = labels, labels[index]
                inputs = inputs_mix
            
            # Forward pass with optional AMP
            if config.use_amp or use_transformer_amp:
                with autocast():
                    outputs = model(inputs)
                    if use_mixup:
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    else:
                        loss = criterion(outputs, labels)
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if config.grad_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_val)
                    
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training flow without AMP
                outputs = model(inputs)
                if use_mixup:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)
                
                loss.backward()
                
                # Gradient clipping
                if config.grad_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_val)
                    
                optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate training accuracy for non-mixup batches
            if not use_mixup:
                _, predicted_train = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train if total_train > 0 else 0
        
        # Validation phase
        model.eval()  # Explicitly set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted_val = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()
        
        # Calculate validation metrics
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_val / total_val if total_val > 0 else 0
        
        # Update scheduler
        if config.scheduler_type == "plateau":
            scheduler.step(val_loss)
        elif config.scheduler_type == "cosine":
            scheduler.step()
        
        # Get current learning rate (just once)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        history['lr'].append(current_lr)
        
        # Print epoch results
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.num_epochs} [{elapsed_time:.1f}s]")
        print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"  Valid Loss: {val_loss:.4f}, Valid Acc: {val_accuracy:.4f}")
        print(f"  Learning Rate: {current_lr:.8f}")
        
        # Track the best model for final return value
        if val_accuracy > best_val_acc:
            print(f"  Validation accuracy improved from {best_val_acc:.4f} to {val_accuracy:.4f}")
            best_val_acc = val_accuracy
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save best model checkpoint only if we're not saving only at the end
            if not save_only_at_end and not save_last_model:
                checkpoint_path = os.path.join(output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_accuracy': best_val_acc,
                    'history': history,
                }, checkpoint_path)
                print(f"  Saved best model checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # If save_last_model is True, save the current model at each epoch (unless save_only_at_end is True)
        if save_last_model and not save_only_at_end:
            checkpoint_path = os.path.join(output_dir, 'model_weights.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'history': history,
            }, checkpoint_path)
            print(f"  Saved last epoch model to {checkpoint_path}")
        
        # Early stopping check with warm-up period
        if epoch >= early_stopping_warmup:
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs (after {early_stopping_warmup} warm-up epochs)")
                
                # Save model when early stopping is triggered, if save_only_at_end is True
                if save_only_at_end:
                    if save_last_model:
                        # Save the current model if save_last_model is True
                        checkpoint_path = os.path.join(output_dir, 'model_weights.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_accuracy': val_accuracy,
                            'history': history,
                        }, checkpoint_path)
                        print(f"Early stopping - saved last model to {checkpoint_path}")
                    else:
                        # Save the best model if save_last_model is False
                        checkpoint_path = os.path.join(output_dir, 'best_model.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': best_model_state,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': best_val_loss,
                            'val_accuracy': best_val_acc,
                            'history': history,
                        }, checkpoint_path)
                        print(f"Early stopping - saved best model to {checkpoint_path}")
                
                break
        else:
            print(f"  Warm-up phase: {epoch+1}/{early_stopping_warmup} epochs")
    
    # Save final model state if we reached the end of training
    # and haven't saved it yet due to save_only_at_end=True
    if save_only_at_end and epoch == config.num_epochs - 1:
        if save_last_model:
            # Save the last model if save_last_model is True
            checkpoint_path = os.path.join(output_dir, 'model_weights.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'history': history,
            }, checkpoint_path)
            print(f"End of training - saved last model to {checkpoint_path}")
        else:
            # Save the best model if save_last_model is False
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': best_val_acc,
                'history': history,
            }, checkpoint_path)
            print(f"End of training - saved best model to {checkpoint_path}")
    
    # Always load the best model for validation and analysis, even if we saved the last model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation accuracy: {best_val_acc:.4f}")
    
    # Validate the best model to get predictions for analysis        
    model.eval()
    all_true_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_true_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Get dataset class names for analysis
    if hasattr(val_loader.dataset, 'dataset'):  # For Subset
        classes = val_loader.dataset.dataset.classes
    else:
        classes = val_loader.dataset.classes
    
    # Analyze false predictions
    analysis = analyze_false_predictions(all_true_labels, all_predictions, classes)
    
    # Print report
    print_false_prediction_report(analysis)
    
    # Save analysis to directory passed as parameter 
    analysis_path = os.path.join(output_dir, 'false_prediction_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(convert_to_json_serializable(analysis), f, indent=2)
    
    return model, best_val_acc, history, analysis


def cross_validate(config, device):
    """Perform cross-validation with enhanced features."""
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Check for multi-GPU setup
    if hasattr(config, 'use_multi_gpu') and config.use_multi_gpu:
        num_gpus = torch.cuda.device_count()
        if (num_gpus > 1):
            print(f"Multi-GPU training enabled with {num_gpus} GPUs")
            # Adjust batch size if using multiple GPUs to utilize them effectively
            effective_batch_size = config.batch_size * num_gpus
            print(f"Effective batch size: {effective_batch_size} (per GPU: {config.batch_size})")
        else:
            print("Multi-GPU training requested but only one GPU found.")
    
    # Create version directory
    version_dir = os.path.join(config.output_dir, config.version)
    os.makedirs(version_dir, exist_ok=True)
    
    # Create mushroom transform parameters dictionary from config
    mushroom_params = {
        'radial_strength': getattr(config, 'radial_distortion_strength', 0.15),
        'radial_p': getattr(config, 'radial_distortion_p', 0.3),
        'elastic_alpha': getattr(config, 'elastic_deform_alpha', 2.0),
        'elastic_sigma': getattr(config, 'elastic_deform_sigma', 1.5),
        'elastic_p': getattr(config, 'elastic_deform_p', 0.15),
        'focus_zoom_strength': getattr(config, 'focus_zoom_strength', 0.2),
        'focus_zoom_p': getattr(config, 'focus_zoom_p', 0.3),
        'aspect_ratio_p': getattr(config, 'aspect_ratio_p', 0.3),
        'grid_shuffle_p': getattr(config, 'grid_shuffle_p', 0.2),
        'polar_p': getattr(config, 'polar_transform_p', 0.2),
        'tps_strength': getattr(config, 'tps_strength', 0.05),
        'tps_p': getattr(config, 'tps_p', 0.1)
    }
    
    # Get appropriate transforms based on configuration with consistent parameters
    if getattr(config, 'use_albumentations', False) and ALBUMENTATIONS_AVAILABLE:
        print(f"Using Albumentations transforms with strength: {config.aug_strength}")
        train_transform, val_transform = get_albumentation_transforms(
            aug_strength=getattr(config, 'aug_strength', 'high'),
            image_size=config.image_size,
            multi_scale=getattr(config, 'use_multi_scale', False),
            pixel_percent=getattr(config, 'pixel_percent', 0.05),
            crop_scale=getattr(config, 'crop_scale', 0.9)
        )
    elif getattr(config, 'use_multi_scale', False):
        print("Using enhanced multi-scale transforms")
        train_transform, val_transform = get_enhanced_transforms(
            multi_scale=True,
            image_size=config.image_size,
            pixel_percent=getattr(config, 'pixel_percent', 0.05),
            crop_scale=getattr(config, 'crop_scale', 0.9),
            advanced_spatial_transforms=getattr(config, 'use_advanced_spatial_transforms', True),
            mushroom_transform_params=mushroom_params
        )
    else:
        print("Using standard transforms")
        train_transform, val_transform = get_transforms(
            image_size=config.image_size,
            aug_strength=getattr(config, 'aug_strength', 'standard')
        )
        
    # Initialize dataset once for fold information
    try:
        dataset = MushroomDataset(config.csv_path, transform=None)
        if 'fold' not in dataset.data.columns:
            raise ValueError("CSV file must contain a 'fold' column for cross-validation")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise
    
    val_metrics = []
    fold_results = {}
    all_histories = {}
    all_analyses = {}
    
    # Create class weights if enabled
    class_weights = None
    if config.use_class_weights:
        # Create a tensor of ones for all classes
        class_weights = torch.ones(len(dataset.classes))
        
        # If the prioritized class exists in our classes, give it more weight
        if config.prioritized_class in dataset.classes:
            priority_idx = dataset.classes.index(config.prioritized_class)
            class_weights[priority_idx] = config.weight_multiplier
            print(f"Applying weight {config.weight_multiplier} to class '{config.prioritized_class}' (index {priority_idx})")
        else:
            print(f"Warning: Prioritized class '{config.prioritized_class}' not found in dataset classes: {dataset.classes}")
        
        # Move class weights to the correct device
        class_weights = class_weights.to(device)
    
    # Create a base dataset that will be reused across folds to save memory and loading time
    base_train_dataset = MushroomDataset(config.csv_path, transform=None)
    base_val_dataset = MushroomDataset(config.csv_path, transform=None)
    
    for fold_idx, fold in enumerate(config.train_folds):
        print(f"\n===== Fold {fold+1}/{config.num_folds} ({fold_idx+1}/{len(config.train_folds)}) =====")
        
        try:
            # Create train/val datasets for this fold with appropriate transforms
            train_indices = dataset.data[dataset.data['fold'] != fold].index.tolist()
            val_indices = dataset.data[dataset.data['fold'] == fold].index.tolist()
            if len(train_indices) == 0 or len(val_indices) == 0:
                print(f"Warning: Empty train or validation set for fold {fold}. Skipping...")
                continue
            
            # Create fold directory
            fold_dir = os.path.join(version_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # Create subsets for this fold using CustomSubset for more efficient transform application
            train_subset = CustomSubset(Subset(base_train_dataset, train_indices), train_transform)
            val_dataset = CustomSubset(Subset(base_val_dataset, val_indices), val_transform)
            
            # Apply MixupDataset wrapper to training data if enabled
            if getattr(config, 'use_mixup_class', False):
                original_num_classes = len(dataset.classes)
                print(f"Adding mixup class to training data (original classes: {original_num_classes})")
                train_dataset = MixupDataset(
                    train_subset,
                    mixup_ratio=config.mixup_class_ratio,
                    mixup_class_name=config.mixup_class_name,
                    strategy=config.mixup_strategy
                )
                print(f"Training dataset now has {len(train_dataset.classes)} classes: {train_dataset.classes}")
                print(f"Added {train_dataset.num_mixup} mixup samples to training data")
                # Store original num_classes for model creation
                original_num_classes = train_dataset.get_original_num_classes()
            else:
                train_dataset = train_subset
                original_num_classes = len(dataset.classes)
            
            # Create data loaders with optimized settings and error handling
            train_loader = DataLoader(
                train_dataset,  
                batch_size=config.batch_size, 
                shuffle=True, 
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                drop_last=False  # Don't drop last batch even if smaller than batch_size
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config.batch_size, 
                shuffle=False, 
                num_workers=config.num_workers,
                pin_memory=config.pin_memory
            )
            
            print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
            
            # Initialize model with proper error handling
            try:
                if getattr(config, 'use_mixup_class', False):
                    # For training, create a model with extra class for mixup
                    model = get_model(len(train_dataset.classes), config, device)
                else:
                    # Normal case - create model with the original number of classes
                    model = get_model(original_num_classes, config, device)
            except Exception as e:
                print(f"Error initializing model: {str(e)}")
                print("Skipping fold due to model initialization failure")
                continue
            
            # Get label smoothing value from config
            label_smoothing = getattr(config, 'label_smoothing', 0.0)
            
            # Determine which loss function to use
            loss_type = getattr(config, 'loss_type', 'ce')
            
            if loss_type == "poly":
                print(f"Using PolyLoss with epsilon={config.poly_loss_epsilon}")
                
                # Set up PolyLoss with class weights if needed
                if class_weights is not None:
                    if getattr(config, 'use_mixup_class', False):
                        # Add a weight of 1.0 for the mixup class
                        extended_weights = torch.cat([class_weights, torch.tensor([1.0], device=device)])
                        criterion = create_poly_loss(ce_weight=extended_weights, epsilon=config.poly_loss_epsilon)
                        print(f"Using weighted PolyLoss with extended weights: {extended_weights.cpu().numpy()}")
                    else:
                        criterion = create_poly_loss(ce_weight=class_weights, epsilon=config.poly_loss_epsilon)
                        print(f"Using weighted PolyLoss with weights: {class_weights.cpu().numpy()}")
                else:
                    criterion = create_poly_loss(epsilon=config.poly_loss_epsilon)
            else:
                # Default to CrossEntropyLoss with label smoothing
                if class_weights is not None:
                    if getattr(config, 'use_mixup_class', False):
                        # Add a weight of 1.0 for the mixup class
                        extended_weights = torch.cat([class_weights, torch.tensor([1.0], device=device)])
                        criterion = nn.CrossEntropyLoss(weight=extended_weights, label_smoothing=label_smoothing)
                        print(f"Using weighted CrossEntropyLoss with extended weights: {extended_weights.cpu().numpy()} and label smoothing: {label_smoothing}")
                    else:
                        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
                        print(f"Using weighted CrossEntropyLoss with weights: {class_weights.cpu().numpy()} and label smoothing: {label_smoothing}")
                else:
                    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                    if label_smoothing > 0:
                        print(f"Using CrossEntropyLoss with label smoothing: {label_smoothing}")
                    else:
                        print("Using standard CrossEntropyLoss without label smoothing")
            
            optimizer = get_layer_wise_lr_optimizer(model, config)
            
            # Create scheduler with proper error handling
            try:
                if config.scheduler_type == "plateau":
                    scheduler = ReduceLROnPlateau(
                        optimizer, 
                        mode='min', 
                        factor=config.scheduler_factor, 
                        patience=config.scheduler_patience, 
                        verbose=True
                    )
                elif config.scheduler_type == "cosine":
                    scheduler = CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=10,
                        T_mult=2,
                        eta_min=1e-6
                    )
                else:
                    raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
            except Exception as e:
                print(f"Error creating scheduler: {str(e)}")
                print("Using default scheduler (ReduceLROnPlateau)")
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            
            # Train model - updated to also return analysis, passing fold_dir as output directory
            model, val_accuracy, history, analysis = train_model(
                model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                scheduler, 
                config, 
                device,
                fold_dir  # Pass fold_dir as the output directory
            )
            
            # Store results
            val_metrics.append(val_accuracy)
            fold_results[fold] = val_accuracy
            all_histories[fold] = history
            all_analyses[fold] = analysis
            
            # Save fold-specific model with error handling
            try:
                torch.save(model.state_dict(), os.path.join(fold_dir, 'model_weights.pth'))
            except Exception as e:
                print(f"Error saving model: {str(e)}")
            
            # Save analysis to fold directory with error handling
            try:
                analysis_path = os.path.join(fold_dir, 'false_prediction_analysis.json')
                with open(analysis_path, 'w') as f:
                    json.dump(convert_to_json_serializable(analysis), f, indent=2)
            except Exception as e:
                print(f"Error saving analysis: {str(e)}")
            
            # Clear some memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"Error processing fold {fold}: {str(e)}")
            print("Continuing to next fold...")
            continue
    
    # Only compute summary statistics if we have results
    if val_metrics:
        # Report cross-validation results
        avg_val_accuracy = np.mean(val_metrics)
        std_val_accuracy = np.std(val_metrics)
        print("\n===== Cross-Validation Summary =====")
        for fold, acc in fold_results.items():
            print(f"Fold {fold+1}: {acc:.4f}")
        print(f"Average: {avg_val_accuracy:.4f} ± {std_val_accuracy:.4f}")
        
        # Save overall CV results
        cv_results = {
            'fold_accuracies': fold_results,
            'mean_accuracy': float(avg_val_accuracy),
            'std_accuracy': float(std_val_accuracy),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v 
                      for k, v in config.__dict__.items()},
        }
        with open(os.path.join(version_dir, 'cv_results.json'), 'w') as f:
            json.dump(cv_results, f, indent=4)
        return avg_val_accuracy, fold_results, all_histories, all_analyses
    else:
        print("No valid results from any fold. Cross-validation failed.")
        return 0.0, {}, {}, {}


def evaluate_model(model, data_loader, criterion, device, num_classes=None):
    """Evaluate model performance on a dataset with comprehensive metrics."""
    # Set model to evaluation model
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Prepare storage for detailed metrics
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    # Track progress for large datasets
    total_batches = len(data_loader)
    print_interval = max(1, total_batches // 10)
    
    try:
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(data_loader):
                # Progress indication
                if batch_idx % print_interval == 0:
                    print(f"Evaluating batch {batch_idx}/{total_batches}")
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Compute probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and targets for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
        
        # Combine all probabilities
        if all_probabilities:
            all_probabilities = np.vstack(all_probabilities)
        
        # Calculate basic metrics
        avg_loss = running_loss / total if total > 0 else float('inf')
        accuracy = correct / total if total > 0 else 0.0
        
        # Prepare return value as a comprehensive results dictionary
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total,
        }
        
        # If class names are available, add detailed metrics
        if num_classes is not None:
            # Create confusion matrix and get per-class metrics
            analysis = analyze_false_predictions(all_targets, all_predictions, 
                                              list(range(num_classes)))
            results['analysis'] = analysis
            results['per_class_accuracy'] = {
                i: stats['accuracy'] for i, stats in analysis['per_class'].items()
            }
        
        # Add predictions and probabilities for external analysis
        results['predictions'] = np.array(all_predictions)
        results['targets'] = np.array(all_targets)
        results['probabilities'] = all_probabilities
        
        return results
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'error': str(e)
        }

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Inference

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:43.990833Z","iopub.execute_input":"2025-04-19T04:08:43.991064Z","iopub.status.idle":"2025-04-19T04:08:44.028720Z","shell.execute_reply.started":"2025-04-19T04:08:43.991045Z","shell.execute_reply":"2025-04-19T04:08:44.028140Z"}}
def preprocess_image(image_path, transform=None):
    """Preprocess an image for model inference."""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Match the model's expected input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    return img_tensor


def predict_image(model, image_path, class_names, device, transform=None):
    """Run inference on a single image and return predictions."""
    try:
        img_tensor = preprocess_image(image_path, transform)
        img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted_idx.item()]
        confidence = confidence.item()
        
        # Get top-k predictions
        k = min(3, len(class_names))
        topk_values, topk_indices = torch.topk(probabilities, k)
        topk_predictions = [
            (class_names[idx.item()], prob.item()) 
            for idx, prob in zip(topk_indices[0], topk_values[0])
        ]
        
        # Add full probability distribution
        class_probabilities = {class_name: probabilities[0, idx].item() 
                              for idx, class_name in enumerate(class_names)}
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'top_predictions': topk_predictions,
            'class_probabilities': class_probabilities,
            'success': True
        }
    except Exception as e:
        print(f"Error predicting image {image_path}: {str(e)}")
        return {
            'class': None,
            'confidence': 0.0,
            'top_predictions': [],
            'error': str(e),
            'success': False
        }


def batch_inference(model, image_dir, class_names, device, transform=None, batch_size=16):
    """Run inference on multiple images in a directory."""
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_paths.extend(list(Path(image_dir).glob(f'*.{ext}')))
    
    # Sort paths for consistent ordering
    image_paths = sorted(image_paths)
    print(f"Running inference on {len(image_paths)} images in batches of {batch_size}")
    if not image_paths:
        print(f"No images found in {image_dir}")
        return [], torch.tensor([]), class_names  # Return consistent tuple structure
    
    results = []
    all_probabilities = []
    processed_count = 0
    failed_count = 0
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        valid_tensors = []
        valid_paths = []
        
        # Process each image with error handling
        for img_path in batch_paths:
            try:
                tensor = preprocess_image(str(img_path), transform)
                valid_tensors.append(tensor)
                valid_paths.append(img_path)
            except Exception as e:
                failed_count += 1
                print(f"Error processing image {img_path}: {str(e)}")
                # Continue to next image instead of failing entire batch
        
        if not valid_tensors:
            print(f"No valid images in current batch, skipping")
            continue
        
        # Stack successful tensors and run inference
        try:
            batch_tensor = torch.stack(valid_tensors).to(device)
            with torch.no_grad():
                outputs = model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_indices = torch.max(probabilities, 1)
            
            # Store the full probabilities tensor for later processing
            all_probabilities.append(probabilities.cpu())
            
            # Process each result
            for j, (img_path, pred_idx, conf) in enumerate(zip(valid_paths, predicted_indices, confidence)):
                predicted_class = class_names[pred_idx.item()]
                
                # Get top-k predictions for this image
                k = min(3, len(class_names))
                topk_values, topk_indices = torch.topk(probabilities[j], k)
                topk_predictions = [
                    (class_names[idx.item()], prob.item()) 
                    for idx, prob in zip(topk_indices, topk_values)
                ]
                
                # Get image filename without extension for CSV
                filename = Path(img_path).name
                
                # Add full probability distribution to results
                class_probs = {class_name: probabilities[j, idx].item() 
                              for idx, class_name in enumerate(class_names)}
                
                results.append({
                    'image_path': str(img_path),
                    'filename': filename,
                    'class': predicted_class,
                    'class_id': pred_idx.item(),
                    'confidence': conf.item(),
                    'top_predictions': topk_predictions,
                    'class_probabilities': class_probs  # Store all class probabilities
                })
                processed_count += 1
        except Exception as e:
            print(f"Error during batch inference: {str(e)}")
            # Continue to next batch
    
    # Print summary
    print(f"Processed {processed_count} images successfully, {failed_count} images failed")
    
    # Combine all probability tensors
    if all_probabilities:
        all_probs = torch.cat(all_probabilities, dim=0)
    else:
        all_probs = torch.tensor([])
    
    return results, all_probs, class_names


def save_inference_results(results, output_file):
    """Save inference results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


def save_submission_csv(results, output_file, config=None):
    """Save prediction results to a submission CSV file in the format id,type."""
    if config and getattr(config, 'use_mixup_class', False):
        # For mixup class setup, process the class probabilities without considering mixup class
        # Create DataFrame with filename and class probabilities
        data = []
        for result in results:
            if 'class_probabilities' in result:
                # Extract data with all probabilities
                row = {'filename': result['filename']}
                row.update(result['class_probabilities'])
                data.append(row)
            else:
                # Fallback if no probabilities
                data.append({
                    'filename': result['filename'],
                    'class': result['class']
                })
        
        if data and 'class_probabilities' in results[0]:  # Only if we have probability data
            # Create DataFrame and process
            df = pd.DataFrame(data)
            
            # Get the mixup class name
            mixup_class_name = getattr(config, 'mixup_class_name', 'mixup')
            
            # Get all columns except filename and mixup class
            original_class_columns = [col for col in df.columns 
                                    if col != 'filename' and col != mixup_class_name]
            
            if original_class_columns:
                # Find original class with highest probability (ignore mixup class)
                print(f"Generating submission by ignoring '{mixup_class_name}' class")
                argmax_result = df[original_class_columns].idxmax(axis=1)
                
                # Create submission with max probability class
                submission_df = pd.DataFrame({
                    'id': df['filename'],
                    'type': argmax_result
                })
                
                # Map class names to CLASS_MAP indices
                submission_df['type'] = submission_df['type'].map(CLASS_MAP)
                submission_df.to_csv(output_file, index=False)
                print(f"Submission saved to {output_file} (ignoring mixup class)")
                return
    
    # Standard path for non-mixup models or fallback
    df = pd.DataFrame([{
        'id': result['filename'],
        'type': result['class'] 
    } for result in results])
    
    df['type'] = df['type'].map(CLASS_MAP)
    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")


def save_logits_csv(filenames, probabilities, class_names, output_file):
    """Save all class probabilities to a logits CSV file."""
    # Create a DataFrame with one row per image
    data = {'filename': filenames}
    # Add one column per class with the probability values
    for i, class_name in enumerate(class_names):
        data[class_name] = probabilities[:, i].numpy()
    
    # Create and save DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Logits saved to {output_file}")


def run_inference(config, model_path, input_path, output_dir, device):
    """Run the inference pipeline on a directory of images."""
    try:
        # Get the class names
        dataset = MushroomDataset(config.csv_path, transform=None)
        class_names = dataset.classes
        
        # If we're using mixup class, add it to the class names if it's not already there
        if getattr(config, 'use_mixup_class', False) and config.mixup_class_name not in class_names:
            class_names = class_names + [config.mixup_class_name]
            print(f"Added mixup class '{config.mixup_class_name}' to class names for inference")
        
        # Load the model with proper error handling
        try:
            model = load_model_from_checkpoint(model_path, len(class_names), config, device)
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            return [], None, None
        
        # Create validation transform with correct image size from config
        _, val_transform = get_transforms(image_size=config.image_size)
        
        # Ensure input_path is a Path object
        input_path = Path(input_path)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output paths:
        output_json_path = os.path.join(output_dir, "inference_results.json")
        output_submission_path = os.path.join(output_dir, "submission.csv")
        output_logits_path = os.path.join(output_dir, "logits.csv")
        
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory. Please provide a directory of images.")
            return [], None, None
        
        # Run batch inference
        print(f"Running batch inference on directory: {input_path}")
        results, all_probabilities, class_names = batch_inference(
            model, input_path, class_names, device, val_transform, 
            batch_size=config.inference_batch_size
        )
        
        if not results or len(all_probabilities) == 0:
            print("No valid results generated. Check if the input directory contains valid images.")
            return [], torch.tensor([]), class_names
        
        # Get filenames in the same order as probabilities
        filenames = [result['filename'] for result in results]
        
        # Save results in all formats
        try:
            save_inference_results(results, output_json_path)
            save_submission_csv(results, output_submission_path, config)  # Pass config parameter
            save_logits_csv(filenames, all_probabilities, class_names, output_logits_path)
            print(f"Results saved to {output_dir}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
        
        # Check if any results have ground truth labels
        has_true_labels = any('true_label' in result for result in results)
        
        if has_true_labels:
            try:
                # Extract true labels and predictions where available
                true_labels = []
                predicted_labels = []
                for result in results:
                    if 'true_label' in result:
                        true_labels.append(result['true_label'])
                        predicted_labels.append(result['class_id'])
                
                # Only perform analysis if we have ground truth labels
                if true_labels:
                    # Analyze false predictions
                    analysis = analyze_false_predictions(true_labels, predicted_labels, class_names)
                    
                    # Save analysis
                    analysis_path = os.path.join(output_dir, "false_prediction_analysis.json")
                    with open(analysis_path, 'w') as f:
                        json.dump(convert_to_json_serializable(analysis), f, indent=2)
                    
                    # Print report
                    print_false_prediction_report(analysis)
            except Exception as e:
                print(f"Error during analysis: {str(e)}")
        
        return results, all_probabilities, class_names
    except Exception as e:
        print(f"Inference pipeline failed: {str(e)}")
        traceback.print_exc()
        return [], None, None


def combine_fold_predictions(fold_predictions, class_names, ensemble_method="mean"):
    """Combine predictions from multiple folds using voting or averaging of full probability distributions."""
    if not fold_predictions:
        return []
    
    # Group predictions by filename
    combined_results = {}
    for fold_preds in fold_predictions:
        for pred in fold_preds:
            filename = pred['filename']
            
            if filename not in combined_results:
                combined_results[filename] = {
                    'filename': filename,
                    'image_path': pred['image_path'],
                    'fold_predictions': [],
                    'fold_class_probabilities': []  # Store full probability distributions
                }
            
            combined_results[filename]['fold_predictions'].append({
                'fold': pred.get('fold', -1),
                'class': pred['class'],
                'class_id': pred['class_id'],
                'confidence': pred['confidence']
            })
            # Store full class probability distributions if available
            if 'class_probabilities' in pred:
                combined_results[filename]['fold_class_probabilities'].append(
                    pred['class_probabilities']
                )
    
    # Process each image's multiple predictions
    final_results = []
    for filename, result in combined_results.items():
        # Initialize to avoid variable scope issues
        class_probabilities = None
        
        try:
            if ensemble_method == "vote":
                # Use majority voting
                votes = {}
                for pred in result['fold_predictions']:
                    class_name = pred['class']
                    votes[class_name] = votes.get(class_name, 0) + 1
                
                # Find class with most votes
                if votes:
                    final_class = max(votes.items(), key=lambda x: x[1])[0]
                    
                    # Find class_id and average confidence for this class
                    matching_preds = [p for p in result['fold_predictions'] if p['class'] == final_class]
                    if matching_preds:
                        final_class_id = matching_preds[0]['class_id']
                        confidences = [p['confidence'] for p in matching_preds]
                        final_confidence = sum(confidences) / len(confidences)
                    else:
                        # Shouldn't happen but handle it anyway
                        print(f"Warning: No matching predictions found for voted class '{final_class}'")
                        final_class_id = 0
                        final_confidence = 0.5
                else:
                    # No votes (should never happen)
                    print(f"Warning: No votes for {filename}, using first prediction")
                    final_class = result['fold_predictions'][0]['class']
                    final_class_id = result['fold_predictions'][0]['class_id']
                    final_confidence = result['fold_predictions'][0]['confidence']
            else:  # Default is "mean" - average probability distributions
                # Check if we have full probability distributions
                if result['fold_class_probabilities'] and len(result['fold_class_probabilities']) > 0:
                    # Initialize an averaged probability distribution
                    avg_probs = {class_name: 0.0 for class_name in class_names}
                    
                    # Sum probabilities for each class across all folds
                    for probs in result['fold_class_probabilities']:
                        for class_name, prob in probs.items():
                            if class_name in avg_probs:
                                avg_probs[class_name] += prob
                    
                    # Average the summed probabilities
                    num_folds = len(result['fold_class_probabilities'])
                    for class_name in avg_probs:
                        avg_probs[class_name] /= num_folds
                    
                    # Find the class with highest average probability
                    if avg_probs:
                        final_class = max(avg_probs.items(), key=lambda x: x[1])[0]
                        final_confidence = avg_probs[final_class]
                        
                        # Find the class_id for the final class
                        try:
                            final_class_id = class_names.index(final_class)
                        except ValueError:
                            print(f"Warning: Class '{final_class}' not found in class_names")
                            final_class_id = 0
                    else:
                        # Empty probabilities (should never happen)
                        print(f"Warning: Empty probability distribution for {filename}")
                        final_class = class_names[0]
                        final_class_id = 0
                        final_confidence = 0.0
                    
                    # Store the full averaged distribution
                    class_probabilities = avg_probs
                else:
                    # Fall back to confidence averaging if no distributions
                    print(f"Warning: No probability distributions for {filename}")
                    
                    # Create a mapping of class_id to class_name
                    id_to_name = {p['class_id']: p['class'] for p in result['fold_predictions']}
                    
                    # Get average confidence per class
                    class_scores = {}
                    for pred in result['fold_predictions']:
                        class_id = pred['class_id']
                        if class_id not in class_scores:
                            class_scores[class_id] = []
                        class_scores[class_id].append(pred['confidence'])
                    
                    # Average the scores
                    avg_scores = {cid: sum(scores)/len(scores) for cid, scores in class_scores.items()}
                    
                    # Find class with highest average score
                    if avg_scores:
                        final_class_id = max(avg_scores.items(), key=lambda x: x[1])[0]
                        final_confidence = avg_scores[final_class_id]
                        final_class = id_to_name.get(final_class_id, class_names[final_class_id] 
                                                   if 0 <= final_class_id < len(class_names) else "unknown")
                    else:
                        # No scores (should never happen)
                        print(f"Warning: No valid scores for {filename}")
                        final_class = class_names[0]
                        final_class_id = 0
                        final_confidence = 0.0
                    
                    # Create probability distribution as fallback
                    class_probabilities = {class_name: 0.0 for class_name in class_names}
                    for cid, score in avg_scores.items():
                        class_name = id_to_name.get(cid, "")
                        if class_name in class_probabilities:
                            class_probabilities[class_name] = score
            
            # Create the final result entry
            result_entry = {
                'filename': filename,
                'image_path': result['image_path'],
                'class': final_class,
                'class_id': final_class_id,
                'confidence': final_confidence,
                'fold_predictions': result['fold_predictions']
            }
            
            # Add full probability distribution if available
            if ensemble_method == "mean" and class_probabilities is not None:
                result_entry['class_probabilities'] = class_probabilities
            
            final_results.append(result_entry)
        except Exception as e:
            print(f"Error processing ensemble for {filename}: {str(e)}")
            # Add basic entry so we don't lose this image
            if result['fold_predictions']:
                first_pred = result['fold_predictions'][0]
                final_results.append({
                    'filename': filename,
                    'image_path': result['image_path'],
                    'class': first_pred['class'],
                    'class_id': first_pred['class_id'],
                    'confidence': first_pred['confidence'],
                    'fold_predictions': result['fold_predictions'],
                    'error': str(e)
                })
    
    return final_results

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Main

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:08:44.029641Z","iopub.execute_input":"2025-04-19T04:08:44.029855Z","iopub.status.idle":"2025-04-19T04:08:44.050918Z","shell.execute_reply.started":"2025-04-19T04:08:44.029837Z","shell.execute_reply":"2025-04-19T04:08:44.050345Z"}}
def main(config):
    """Run both training and inference in a single pipeline."""
    try:
        # Initialize enhanced configuration
        config = config
        initialize_config()  # Update config based on dataset paths
        
        if config.debug:
            print("WARNING: THIS IS DEBUG MODE")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"Using {config.num_workers} workers for data loading")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Running experiment version: {config.version}")
        
        # Create output directory
        version_dir = os.path.join(config.output_dir, config.version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save config to JSON for reproducibility
        try:
            with open(os.path.join(version_dir, 'config.json'), 'w') as f:
                # Handle all potentially non-serializable types
                config_dict = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None), list, dict)) else v 
                              for k, v in config.__dict__.items()}
                json.dump(config_dict, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save configuration to JSON: {str(e)}")
        
        # Save start time for benchmarking
        start_time = time.time()
        
        # Create mushroom transform parameters dictionary for consistent use
        mushroom_params = {
            'radial_strength': getattr(config, 'radial_distortion_strength', 0.15),
            'radial_p': getattr(config, 'radial_distortion_p', 0.3),
            'elastic_alpha': getattr(config, 'elastic_deform_alpha', 2.0),
            'elastic_sigma': getattr(config, 'elastic_deform_sigma', 1.5),
            'elastic_p': getattr(config, 'elastic_deform_p', 0.15),
            'focus_zoom_strength': getattr(config, 'focus_zoom_strength', 0.2),
            'focus_zoom_p': getattr(config, 'focus_zoom_p', 0.3),
            'aspect_ratio_p': getattr(config, 'aspect_ratio_p', 0.3),
            'grid_shuffle_p': getattr(config, 'grid_shuffle_p', 0.2),
            'polar_p': getattr(config, 'polar_transform_p', 0.2),
            'tps_strength': getattr(config, 'tps_strength', 0.05),
            'tps_p': getattr(config, 'tps_p', 0.1)
        }
        
        # Configure transforms at the global scope
        if getattr(config, 'use_albumentations', False):
            print(f"Using Albumentations augmentation with {config.aug_strength} strength")
            train_transform, val_transform = get_albumentation_transforms(
                aug_strength=getattr(config, 'aug_strength', 'high'), 
                image_size=config.image_size, 
                multi_scale=getattr(config, 'use_multi_scale', False),
                pixel_percent=getattr(config, 'pixel_percent', 0.05),
                crop_scale=getattr(config, 'crop_scale', 0.9)
            )
        elif getattr(config, 'use_multi_scale', False):
            print("Using multi-scale training transforms")
            if(getattr(config, 'use_advanced_spatial_transforms', True)):
                print("Using advanced spatial transform with these below config:")
                print(mushroom_params)
            train_transform, val_transform = get_enhanced_transforms(
                multi_scale=True,
                image_size=config.image_size,
                pixel_percent=getattr(config, 'pixel_percent', 0.05),
                crop_scale=getattr(config, 'crop_scale', 0.9),
                advanced_spatial_transforms=getattr(config, 'use_advanced_spatial_transforms', True),
                mushroom_transform_params=mushroom_params
            )
        else:
            print("Using standard transforms")
            train_transform, val_transform = get_transforms(
                image_size=config.image_size, 
                aug_strength="standard"
            )
        
        # === Training Phase ===
        print("\n=== Starting Training Phase (Cross-validation) with Enhanced Features ===")
        
        # Run cross-validation with the enhanced model architecture
        avg_val_accuracy, fold_results, cv_histories, analyses = cross_validate(config, device)

        # Report training time
        train_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(train_time)}")
        
        # === Inference Phase ===
        if config.run_inference_after_training:
            if not config.inference_input_path or not os.path.exists(config.inference_input_path):
                print(f"Warning: Inference path not found or not specified: {config.inference_input_path}")
            else:
                print("\n=== Starting Inference Phase ===")
                
                # Load class names from dataset
                try:
                    dataset = MushroomDataset(config.csv_path, transform=None)
                    class_names = dataset.classes
                except Exception as e:
                    print(f"Error loading dataset for class names: {str(e)}")
                    print("Using default class names for inference")
                    class_names = CLASS_NAMES
                
                # Run inference for each trained fold model
                all_fold_results = []
                
                for fold in config.train_folds:
                    print(f"\n--- Running inference with fold {fold+1} model ---")
                    fold_dir = os.path.join(version_dir, f'fold_{fold}')
                    model_path = os.path.join(fold_dir, 'model_weights.pth')
                    
                    # Check if model exists
                    if not os.path.exists(model_path):
                        print(f"Warning: Model not found at {model_path}, skipping.")
                        continue
                    
                    # Run inference with this fold's model
                    results, probs, _ = run_inference(
                        config, 
                        model_path, 
                        config.inference_input_path, 
                        fold_dir,
                        device
                    )
                    
                    # Add fold information to results if valid
                    if results:
                        for result in results:
                            result['fold'] = fold
                        all_fold_results.append(results)
                    else:
                        print(f"No valid results from fold {fold+1}, skipping in ensemble")
                
                # Combine predictions from all folds using multiple ensemble methods
                if all_fold_results and len(all_fold_results) > 1:
                    # Get ensemble methods (ensure it's a list)
                    ensemble_methods = config.ensemble_methods if isinstance(config.ensemble_methods, list) else [config.ensemble_methods]
                    
                    # Create results for each ensemble method
                    for method in ensemble_methods:
                        if not method:  # Skip empty/None methods
                            continue
                        
                        print(f"\n--- Creating ensemble using method: {method} ---")
                        combined_results = combine_fold_predictions(
                            all_fold_results, 
                            class_names, 
                            ensemble_method=method
                        )
                        
                        if combined_results:
                            # Create ensemble directory with method name
                            ensemble_dir = os.path.join(version_dir, f"ensemble_{method}")
                            os.makedirs(ensemble_dir, exist_ok=True)
                            
                            # Process and save ensemble results
                            try:                                
                                combined_json_path = os.path.join(ensemble_dir, "inference_results.json")
                                combined_submission_path = os.path.join(ensemble_dir, "submission.csv")
                                
                                save_inference_results(combined_results, combined_json_path)
                                save_submission_csv(combined_results, combined_submission_path, config)
                                
                                # If this is the primary method, also save at version level
                                if method == ensemble_methods[0]:
                                    version_submission_path = os.path.join(version_dir, "submission.csv")
                                    save_submission_csv(combined_results, version_submission_path, config)
                                    print(f"Primary ensemble also saved to {version_submission_path}")
                                
                                print(f"Ensemble '{method}' predictions saved to {ensemble_dir}")
                            except Exception as e:
                                print(f"Error saving ensemble results: {str(e)}")
                        else:
                            print(f"Error: Ensemble method '{method}' produced no valid results")
                    
                    # Generate comparison report if multiple methods used
                    if len(ensemble_methods) > 1:
                        try:
                            # Create comparison directory
                            comparison_dir = os.path.join(version_dir, "ensemble_comparison")
                            os.makedirs(comparison_dir, exist_ok=True)
                            
                            # Save comparison report
                            comparison_path = os.path.join(comparison_dir, "methods_comparison.json")
                            with open(comparison_path, 'w') as f:
                                json.dump({
                                    "methods": ensemble_methods,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "config": {k: str(v) if not isinstance(v, (int, float, bool, str, type(None), list, dict)) else v 
                                             for k, v in config.__dict__.items()},
                                }, f, indent=4)
                            print(f"\nEnsemble methods comparison saved to {comparison_path}")
                        except Exception as e:
                            print(f"Error generating ensemble comparison: {str(e)}")
                elif not config.ensemble_methods or (len(config.ensemble_methods) == 1 and not config.ensemble_methods[0]):
                    print("Ensemble is disabled. Each fold's predictions are saved separately.")
                else:
                    print("No valid inference results from any fold, skipping ensemble")
        
        # Report total execution time
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {format_time(total_time)}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 1
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0  # Return success code

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Results

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Running

# %% [code] {"execution":{"iopub.status.busy":"2025-04-19T04:08:44.118821Z","iopub.execute_input":"2025-04-19T04:08:44.119041Z","iopub.status.idle":"2025-04-19T04:08:44.142781Z","shell.execute_reply.started":"2025-04-19T04:08:44.119023Z","shell.execute_reply":"2025-04-19T04:08:44.142206Z"},"jupyter":{"source_hidden":true}}
# 0.0
# Set default class names
CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)', 'linh chi trắng']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

@dataclass
class Config:
    """Configuration for hyperparameters."""
    debug = False
    version: str = "exp0.0"  # Version for organizing outputs
    data_dir: str = 'split_cv/'
    csv_path: str = os.path.join(data_dir, 'train_cv.csv')
    output_dir: str = 'output'
    inference_input_path: str = TEST_DATA_DIR_PATH  # Directory or file for inference
    image_size: int = 32
    batch_size: int = 256  # Reduced from 512 for better generalization
    num_epochs: int = 500 if not debug else 2
    learning_rate: float = 0.01  # Reduced from 0.005
    dropout_rate: float = 0.0  # Increased from 0.2 for better regularization
    l2_reg: float = 0.00001  # Increased from 0.0001
    num_folds: int = 6  # Folds 0 to 5
    train_folds = None
    early_stopping_patience: int = 50  # Increased from 10
    early_stopping_warmup: int = 75
    mixup_alpha: float = 0.5
    mixup_prob: float = 0.0  # Increased from 0.2
    scheduler_factor: float = 0.1  # Increased from 0.1 for faster LR reduction
    scheduler_patience: int = 15  # Increased from 2
    layer_decay_rate: float = 0.95
    pretrained: bool = True  # Changed to True - using pretrained weights helps
    num_workers: int = 4 if torch.cuda.is_available() else 0  # Number of workers for data loading
    pin_memory: bool = True        # Pin memory for faster GPU transfer
    use_amp: bool = False           # Use automatic mixed precision
    grad_clip_val: float = 1.0     # Gradient clipping value
    scheduler_type: str = "plateau" # "plateau" or "cosine"
    seed: int = SEED                 # Random seed
    run_inference_after_training: bool = True  # Whether to run inference after training
    inference_batch_size: int = 256  # Batch size for inference
    submission_path: str = None  # Will be auto-set to output_dir/submission.csv if None
    logits_path: str = None  # Will be auto-set to output_dir/logits.csv if None
    save_per_fold_results: bool = True  # Save results for each fold separately
    use_class_weights: bool = False  # Whether to use class weights in loss function
    weight_multiplier: float = 2.0  # How much extra weight to give to the prioritized class
    prioritized_class: str = "Đùi gà Baby (cắt ngắn)"  # Class to prioritize
    ensemble_methods = ["mean", "vote"]  # List of methods to combine predictions
    save_last_model: bool = True  # If True, save the last model instead of the best model
    save_only_at_end: bool = True  # If True, save model only at the end of training to reduce I/O operations
    
@dataclass 
class TransConfig:
    # Transformer parameters
    transformer_size: str = None  # Set to None to use manual configuration instead of presets
    transformer_d_model: int = 128  # Embedding dimension
    transformer_nhead: int = 8  # Number of attention heads
    transformer_num_layers: int = 6  # Number of transformer layers
    transformer_dim_feedforward: int = 512  # Size of feedforward layer in transformer
    transformer_dropout_rate: float = 0.1
    transformer_type: str = "pixt"  # Options: "pixt", "vt", "patchpixt", etc.
    transformer_patch_size: int = 4  # Patch size for PatchPixT (2, 4, or 8)
    transformer_patch_sizes = None  # List of patch sizes for MultiPatchPixT models
    transformer_fusion_type: str = "concat"  # How to fuse features
    transformer_growth_rate: int = 12  # Growth rate for CNN in CNNMultiPatchPixT
    
    # Memory efficiency options for transformers
    transformer_use_gradient_checkpointing: bool = False
    transformer_sequence_reduction_factor: int = 1
    transformer_share_layer_params: bool = False
    transformer_use_amp: bool = False  # Use automatic mixed precision for transformers
    
    
    # AstroTransformer specific parameters
    astro_expansion: int = 2  # Expansion factor for AstroTransformer
    astro_layers: list = None  # Number of blocks per stage, defaults to [2, 2, 2]

@dataclass
class EnhancedConfig(Config, TransConfig):
    """Enhanced configuration with additional parameters for advanced techniques."""
    model_type: str = "spdresnet" # v1.22 change from rlnet to spdresnet
    use_multi_scale: bool = True  # Whether to use multi-scale training
    use_albumentations: bool = False  # Whether to use Albumentations augmentation library
    use_advanced_spatial_transforms: bool = True  # Whether to use advanced spatial transformations
    aug_strength: str = "low"  # Options: "low", "medium", "high"
    pixel_percent: float = 0.00 
    crop_scale: float = 0.9
    label_smoothing: float = 0.0  # Label smoothing factor (0.0 means no smoothing)
    
    # Transform-related parameters
    elastic_deform_p: float = 0.15  # Probability for elastic deformation
    elastic_deform_alpha: float = 2.0  # Alpha parameter for elastic deformation
    elastic_deform_sigma: float = 1.5  # Sigma parameter for elastic deformation
    focus_zoom_strength: float = 0.2  # Strength for central focus zoom
    focus_zoom_p: float = 0.3  # Probability for central focus zoom
    aspect_ratio_p: float = 0.3  # Probability for aspect ratio variation
    grid_shuffle_p: float = 0.2  # Probability for grid shuffle
    polar_transform_p: float = 0.2  # Probability for polar transform
    tps_strength: float = 0.05  # Strength for thin plate spline
    tps_p: float = 0.1  # Probability for thin plate spline
    radial_distortion_strength: float = 0.15  # Strength for radial distortion
    radial_distortion_p: float = 0.3  # Probability for radial distortion
    
    # Mixup class parameters
    use_mixup_class: bool = True  # Whether to add a mixup class to training
    mixup_class_ratio: float = 0.2  # Ratio of mixup samples to original samples
    mixup_class_name: str = "mixup"  # Name of the mixup class
    mixup_strategy: str = "average"  # How to combine images: "average", "overlay", "mosaic"
    
    
    # Multi-GPU support
    use_multi_gpu: bool = True  # Whether to use multiple GPUs if available
    gpu_ids = None  # Specific GPU IDs to use, None means use all available
def initialize_config():
    """Initialize config settings based on data paths and defaults."""
    if Config.train_folds is None:
        Config.train_folds = list(range(Config.num_folds))
        print(f"No train_folds specified, using all {Config.num_folds} folds")
        
# %% [code] {"execution":{"iopub.status.busy":"2025-04-19T04:08:44.143365Z","iopub.execute_input":"2025-04-19T04:08:44.143569Z","iopub.status.idle":"2025-04-19T04:12:48.954335Z","shell.execute_reply.started":"2025-04-19T04:08:44.143551Z","shell.execute_reply":"2025-04-19T04:12:48.953507Z"},"jupyter":{"outputs_hidden":true}}
config = EnhancedConfig()
main(config)

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-19T04:12:48.955168Z","iopub.execute_input":"2025-04-19T04:12:48.955400Z","iopub.status.idle":"2025-04-19T04:12:48.980808Z","shell.execute_reply.started":"2025-04-19T04:12:48.955378Z","shell.execute_reply":"2025-04-19T04:12:48.980193Z"}}
# 0.1
# Set default class names
CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)', 'linh chi trắng']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

@dataclass
class Config:
    """Configuration for hyperparameters."""
    debug = False
    version: str = "exp0.1"  # Version for organizing outputs
    data_dir: str = 'split_cv/'
    csv_path: str = os.path.join(data_dir, 'train_cv.csv')
    output_dir: str = 'output'
    inference_input_path: str = TEST_DATA_DIR_PATH
    image_size: int = 32
    batch_size: int = 256  # Reduced from 512 for better generalization
    num_epochs: int = 500 if not debug else 2
    learning_rate: float = 0.01  # Reduced from 0.005
    dropout_rate: float = 0.0  # Increased from 0.2 for better regularization
    l2_reg: float = 0.00001  # Increased from 0.0001
    num_folds: int = 6  # Folds 0 to 5
    train_folds = None
    early_stopping_patience: int = 50  # Increased from 10
    early_stopping_warmup: int = 75
    mixup_alpha: float = 0.5
    mixup_prob: float = 0.0  # Increased from 0.2
    scheduler_factor: float = 0.1  # Increased from 0.1 for faster LR reduction
    scheduler_patience: int = 15  # Increased from 2
    layer_decay_rate: float = 0.95
    pretrained: bool = True  # Changed to True - using pretrained weights helps
    num_workers: int = 4 if torch.cuda.is_available() else 0  # Number of workers for data loading
    pin_memory: bool = True        # Pin memory for faster GPU transfer
    use_amp: bool = False           # Use automatic mixed precision
    grad_clip_val: float = 1.0     # Gradient clipping value
    scheduler_type: str = "plateau" # "plateau" or "cosine"
    seed: int = SEED                 # Random seed
    run_inference_after_training: bool = True  # Whether to run inference after training
    inference_batch_size: int = 256  # Batch size for inference
    submission_path: str = None  # Will be auto-set to output_dir/submission.csv if None
    logits_path: str = None  # Will be auto-set to output_dir/logits.csv if None
    save_per_fold_results: bool = True  # Save results for each fold separately
    use_class_weights: bool = False  # Whether to use class weights in loss function
    weight_multiplier: float = 2.0  # How much extra weight to give to the prioritized class
    prioritized_class: str = "Đùi gà Baby (cắt ngắn)"  # Class to prioritize
    ensemble_methods = ["mean", "vote"]  # List of methods to combine predictions
    save_last_model: bool = True  # If True, save the last model instead of the best model
    save_only_at_end: bool = True  # If True, save model only at the end of training to reduce I/O operations
    
@dataclass 
class TransConfig:
    # Transformer parameters
    transformer_size: str = None  # Set to None to use manual configuration instead of presets
    transformer_d_model: int = 128  # Embedding dimension
    transformer_nhead: int = 8  # Number of attention heads
    transformer_num_layers: int = 6  # Number of transformer layers
    transformer_dim_feedforward: int = 512  # Size of feedforward layer in transformer
    transformer_dropout_rate: float = 0.1
    transformer_type: str = "pixt"  # Options: "pixt", "vt", "patchpixt", etc.
    transformer_patch_size: int = 4  # Patch size for PatchPixT (2, 4, or 8)
    transformer_patch_sizes = None  # List of patch sizes for MultiPatchPixT models
    transformer_fusion_type: str = "concat"  # How to fuse features
    transformer_growth_rate: int = 12  # Growth rate for CNN in CNNMultiPatchPixT
    
    # Memory efficiency options for transformers
    transformer_use_gradient_checkpointing: bool = False
    transformer_sequence_reduction_factor: int = 1
    transformer_share_layer_params: bool = False
    transformer_use_amp: bool = False  # Use automatic mixed precision for transformers
    
    
    # AstroTransformer specific parameters
    astro_expansion: int = 2  # Expansion factor for AstroTransformer
    astro_layers: list = None  # Number of blocks per stage, defaults to [2, 2, 2]

@dataclass
class EnhancedConfig(Config, TransConfig):
    """Enhanced configuration with additional parameters for advanced techniques."""
    model_type: str = "rlnet" # v1.22 change from rlnet to spdresnet
    use_multi_scale: bool = True  # Whether to use multi-scale training
    use_albumentations: bool = False  # Whether to use Albumentations augmentation library
    use_advanced_spatial_transforms: bool = True  # Whether to use advanced spatial transformations
    aug_strength: str = "low"  # Options: "low", "medium", "high"
    pixel_percent: float = 0.00 
    crop_scale: float = 0.9
    label_smoothing: float = 0.0  # Label smoothing factor (0.0 means no smoothing)
    
    # Transform-related parameters
    elastic_deform_p: float =  0.15 #0.15  # Probability for elastic deformation
    elastic_deform_alpha: float = 2.0  # Alpha parameter for elastic deformation
    elastic_deform_sigma: float = 1.5  # Sigma parameter for elastic deformation
    focus_zoom_strength: float = 0.2  # Strength for central focus zoom
    focus_zoom_p: float = 0.3 #0.3  # Probability for central focus zoom
    aspect_ratio_p: float = 0.3 #0.3  # Probability for aspect ratio variation
    grid_shuffle_p: float = 0.2 #0.2  # Probability for grid shuffle
    polar_transform_p: float = 0.2 # 0.2  # Probability for polar transform
    tps_strength: float = 0.05  # Strength for thin plate spline
    tps_p: float = 0.1 #0.1  # Probability for thin plate spline
    radial_distortion_strength: float = 0.15  # Strength for radial distortion
    radial_distortion_p: float = 0.3 # 0.3  # Probability for radial distortion
    
    # Mixup class parameters
    use_mixup_class: bool = True  # Whether to add a mixup class to training
    mixup_class_ratio: float = 0.2  # Ratio of mixup samples to original samples
    mixup_class_name: str = "mixup"  # Name of the mixup class
    mixup_strategy: str = "average"  # How to combine images: "average", "overlay", "mosaic"
    
    
    # Multi-GPU support
    use_multi_gpu: bool = True  # Whether to use multiple GPUs if available
    gpu_ids = None  # Specific GPU IDs to use, None means use all available

def initialize_config():
    """Initialize config settings based on data paths and defaults."""
    if Config.train_folds is None:
        Config.train_folds = list(range(Config.num_folds))
        print(f"No train_folds specified, using all {Config.num_folds} folds")

# %% [code] {"execution":{"iopub.status.busy":"2025-04-19T04:12:48.981432Z","iopub.execute_input":"2025-04-19T04:12:48.981651Z","iopub.status.idle":"2025-04-19T04:16:26.547046Z","shell.execute_reply.started":"2025-04-19T04:12:48.981632Z","shell.execute_reply":"2025-04-19T04:16:26.546317Z"},"jupyter":{"outputs_hidden":true}}
config = EnhancedConfig()
main(config)


# FINAL SUBMISSION
import pandas as pd
import numpy as np
exp_list = ["exp0.0","exp0.1"]

path_list = []

for exp in exp_list:
    for i in range(6):
        path_list.append(f"{OUTPUT_DIR_PATH}/{exp}/fold_{i}/logits.csv")
            

df_list = []
for path in path_list:
    df_list.append(pd.read_csv(path))

map_label = {
    "bào ngư xám + trắng": "1",
    "linh chi trắng": "3",
    "nấm mỡ": "0",
    "Đùi gà Baby (cắt ngắn)": "2"
}

sum_df = pd.DataFrame()
for df in df_list:
    numeric_df = df.drop(columns=['filename', 'mixup'])
    if sum_df.empty:
        sum_df = numeric_df
    else:
        sum_df = sum_df.add(numeric_df, fill_value=0)

class_columns = sum_df.columns
max_class_indices = sum_df.values.argmax(axis=1)
max_classes = [class_columns[idx] for idx in max_class_indices]

mapped_labels = [map_label.get(cls, 'unknown') for cls in max_classes]

result_df = pd.DataFrame({
    'image_name': df_list[0]['filename'],
    'label': mapped_labels
})

result_df.to_csv(f"{OUTPUT_DIR_PATH}/results.csv", index=False)
