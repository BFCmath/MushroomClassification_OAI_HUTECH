import os
import pandas as pd
import numpy as np
import random
from dataclasses import dataclass
from datetime import datetime
import time
from pathlib import Path
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from PIL import Image
import cv2
import torchvision.transforms.functional as TF
from PIL import ImageOps, ImageEnhance

# Import LowNet from the same directory
from scripts.main.LowNet import VariableReLU

# Add Albumentations for advanced augmentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("Albumentations not available. Using torchvision transforms only.")
    ALBUMENTATIONS_AVAILABLE = False


# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)', 'linh chi trắng'] 
CLASS_NAMES = ['nm', 'bn', 'dg', 'lc']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
# ### Config Class ###
@dataclass
class Config:
    """Configuration for hyperparameters."""
    debug = False
    version: str = "exp1.0"  # Version for organizing outputs
    data_dir: str = '/kaggle/input/oai-cv/'
    csv_path: str = os.path.join(data_dir, 'train_cv.csv')
    output_dir: str = '/kaggle/working/'
    inference_input_path: str = '/kaggle/input/aio-hutech/test'  # Directory or file for inference
    image_size: int = 32
    batch_size: int = 256  # Reduced from 512 for better generalization
    num_epochs: int = 500 if not debug else 2
    learning_rate: float = 0.001  # Reduced from 0.005
    dropout_rate: float = 0.2  # Increased from 0.2 for better regularization
    l2_reg: float = 0.00001  # Increased from 0.0001
    num_folds: int = 6  # Folds 0 to 5
    train_folds = None
    early_stopping_patience: int = 15  # Increased from 10
    early_stopping_warmup: int = 25
    mixup_alpha: float = 0.5
    mixup_prob: float = 0.0  # Increased from 0.2
    scheduler_factor: float = 0.5  # Increased from 0.1 for faster LR reduction
    scheduler_patience: int = 2  # Increased from 2
    layer_decay_rate: float = 0.95
    pretrained: bool = True  # Changed to True - using pretrained weights helps
    num_workers: int = 4
    pin_memory: bool = True        # Pin memory for faster GPU transfer
    use_amp: bool = False           # Use automatic mixed precision
    grad_clip_val: float = 10.0     # Gradient clipping value
    scheduler_type: str = "cosine" # "plateau" or "cosine"
    seed: int = 42                 # Random seed
    run_inference_after_training: bool = True  # Whether to run inference after training
    inference_batch_size: int = 256  # Batch size for inference
    submission_path: str = None  # Will be auto-set to output_dir/submission.csv if None
    logits_path: str = None  # Will be auto-set to output_dir/logits.csv if None
    ensemble_method: str = "mean"  # How to combine multiple model predictions: "mean" or "vote"
    save_per_fold_results: bool = True  # Save results for each fold separately
    use_class_weights: bool = False  # Whether to use class weights in loss function
    weight_multiplier: float = 2.0  # How much extra weight to give to the prioritized class
    prioritized_class: str = "Đùi gà Baby (cắt ngắn)"  # Class to prioritize

# Update Config class with new parameters
@dataclass
class EnhancedConfig(Config):
    """Enhanced configuration with additional parameters for advanced techniques."""
    model_type: str = "dual_branch"  # Options: dual_branch, densenet, smallresnet, mobilenet, inceptionfsd, dilatedgroupconv, dilatedgroupconv
    use_multi_scale: bool = False  # Whether to use multi-scale training
    use_albumentations: bool = True  # Whether to use Albumentations augmentation library
    aug_strength: str = "high"  # Options: "low", "medium", "high"
    
if Config.csv_path in ["/kaggle/input/oai-cv/train_cv.csv"]:
    CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)', 'linh chi trắng'] 
    CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    Config.num_folds = 6
if Config.csv_path in ['/kaggle/input/oai-cv/train_group_cv.csv']:
    CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)', 'linh chi trắng'] 
    CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    Config.num_folds = 5
if Config.train_folds is None:
    Config.train_folds = list(range(Config.num_folds))
    print(f"No train_folds specified, using all {Config.num_folds} folds")
    
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

# Add new custom color transforms
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
def get_enhanced_transforms(multi_scale=False, image_size=32, aug_strength="high"):
    """
    Creates enhanced transforms specifically designed for low-resolution images
    with options for multi-scale training and advanced color augmentation.
    """
    # Base transformations for low-resolution images
    base_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        
        # Basic color jitter - keep this as it works well with other transforms
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        
        # Advanced color transformations
        AdvancedColorTransforms.RandomGrayscale(p=0.1),
        AdvancedColorTransforms.RandomColorDrop(p=0.1),
        AdvancedColorTransforms.RandomChannelSwap(p=0.1),
        AdvancedColorTransforms.RandomGamma(gamma_range=(0.7, 1.3), p=0.2),
        AdvancedColorTransforms.SimulateLightingCondition(p=0.2),
        AdvancedColorTransforms.SimulateHSVNoise(p=0.2),
        
        # Other transformations
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    ]
    
    # Add random erasing to simulate occlusions
    post_tensor_transforms = [
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        # Add Gaussian noise
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
def get_albumentation_transforms(aug_strength="high", image_size=32, multi_scale=False):
    """
    Creates advanced image augmentations using Albumentations library.
    
    Args:
        aug_strength: Strength of augmentation ('low', 'medium', 'high')
        image_size: Target image size
        multi_scale: Whether to use multi-scale transforms when falling back
    """
    if not ALBUMENTATIONS_AVAILABLE:
        print("Warning: Albumentations not available. Falling back to enhanced transforms.")
        # Pass through the same parameters instead of hardcoding multi_scale=True
        return get_enhanced_transforms(multi_scale=multi_scale, image_size=image_size, aug_strength=aug_strength)
    
    # Configure strength parameters based on aug_strength
    if aug_strength == "low":
        p_optical = 0.3
        p_affine = 0.5
        distort_limit = 0.5
    elif aug_strength == "medium":
        p_optical = 0.5
        p_affine = 0.7
        distort_limit = 0.7
    else:  # high
        p_optical = 0.7
        p_affine = 0.85
        distort_limit = 1.0
        
    # Full training transforms with all augmentations
    train_transform = A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(limit=0.2, p=0.75),
        A.OneOf([
            A.MedianBlur(p=0.5),
            A.GaussianBlur(p=0.5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        A.OneOf([
            A.OpticalDistortion(distort_limit=distort_limit),
            A.GridDistortion(num_steps=5, distort_limit=distort_limit),
            A.ElasticTransform(alpha=3),
        ], p=p_optical),

        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=p_affine),
        A.Resize(image_size, image_size),
        # Add operations equivalent to post-tensor transforms
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),  # Similar to RandomErasing
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),  # Similar to Gaussian noise
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

# ### Space-to-Depth Convolution for Information Preservation ###
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

# ### Inception Module and InceptionFSD Model Implementation ###

class InceptionModule(nn.Module):
    """
    Inception module with parallel pathways for multi-scale feature extraction.
    Inspired by GoogleNet's Inception architecture.
    """
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
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
    
    def forward(self, x):
        # Process input through each branch
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)
        
        # Concatenate outputs along the channel dimension
        outputs = [branch1_output, branch2_output, branch3_output, branch4_output]
        return torch.cat(outputs, 1)


class InceptionFSD(nn.Module):
    """
    Inception-based Feature Scale Detector that combines ideas from GoogleNet (Inception) 
    and SSD (Single Shot Detector) to extract features at multiple scales.
    
    This model:
    1. Uses Inception modules for multi-scale feature extraction within each layer
    2. Extracts feature maps from different network depths to capture both local and global context
    3. Combines these multi-scale features for final classification
    """
    def __init__(self, num_classes=10, dropout_rate=0.4):
        super(InceptionFSD, self).__init__()
        
        # Initial feature extraction
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
            ch5x5red=4,
            ch5x5=8,
            pool_proj=8
        )  # Output: 64 channels (16+32+8+8)
        
        # Reduction block 1 - reduce spatial dimensions
        self.reduction1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 16x16
            nn.Conv2d(64, 80, kernel_size=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True)
        )
        
        # Second inception block
        self.inception2 = InceptionModule(
            in_channels=80,
            ch1x1=32,
            ch3x3red=32,
            ch3x3=48,
            ch5x5red=8,
            ch5x5=16,
            pool_proj=16
        )  # Output: 112 channels (32+48+16+16)
        
        # Reduction block 2 - reduce spatial dimensions
        self.reduction2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 8x8
            nn.Conv2d(112, 160, kernel_size=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True)
        )
        
        # Third inception block
        self.inception3 = InceptionModule(
            in_channels=160,
            ch1x1=64,
            ch3x3red=64,
            ch3x3=96,
            ch5x5red=16,
            ch5x5=32,
            pool_proj=32
        )  # Output: 224 channels (64+96+32+32)
        
        # Multi-scale feature aggregation - pooling layers for different scales
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global context
        self.mid_pool = nn.AdaptiveAvgPool2d((2, 2))     # Mid-level context
        self.local_pool = nn.AdaptiveAvgPool2d((4, 4))   # Local context
        
        # Calculate flattened feature dimensions
        global_features = 224  # From inception3
        mid_features = 224 * 4  # 2x2 spatial dimension
        local_features = 112 * 16  # 4x4 spatial dimension
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(global_features + mid_features + local_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
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
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)
        
        # Feature extraction at different scales
        inception1_out = self.inception1(x)
        
        x = self.reduction1(inception1_out)
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
                stride=stride, 
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
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        # Main branch
        out = self.input_proj(x)
        out = self.grouped_conv(out)
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
    3. Replaces pooling operations with strided convolutions
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
        
        # Transition 1: Strided convolution instead of pooling (32x32 → 16x16)
        self.transition1 = DilatedGroupConvBlock(64, 128, dilation=1, stride=2)
        
        # Stage 2: Medium-scale features (16x16)
        self.stage2 = nn.Sequential(
            DilatedGroupConvBlock(128, 128, dilation=1, stride=1),
            DilatedGroupConvBlock(128, 128, dilation=2, stride=1),
            DilatedGroupConvBlock(128, 128, dilation=4, stride=1)
        )
        
        # Transition 2: Strided convolution instead of pooling (16x16 → 8x8)
        self.transition2 = DilatedGroupConvBlock(128, 256, dilation=1, stride=2)
        
        # Stage 3: Deep features with increased dilation (8x8)
        self.stage3 = nn.Sequential(
            DilatedGroupConvBlock(256, 256, dilation=1, stride=1),
            DilatedGroupConvBlock(256, 256, dilation=2, stride=1)
        )
        
        # Global feature extraction without pooling
        # Use strided convolutions to reduce to 1x1
        self.global_features = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 8x8 → 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 4x4 → 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=0, bias=False),  # 2x2 → 1x1
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

# Fix the get_model function which was corrupted
def get_model(num_classes, config):
    """
    Use a model suited for the image size with architecture specified in config.
    """
    model_type = getattr(config, 'model_type', 'densenet')
    
    if model_type == 'lownet':
        print("Creating LowNet model for low-resolution image feature extraction...")
        model = AdaptedLowNet(num_classes=num_classes, 
                                    dropout_rate=config.dropout_rate)
        return model
    elif model_type == 'dilatedgroupconv':
        print("Creating InceptionFSD model with multi-scale feature extraction...")
        model = InceptionFSD(num_classes=num_classes, 
                            dropout_rate=config.dropout_rate)
        return model
    elif model_type == 'dual_branch':
        print("Creating Dual-Branch Network with common feature subspace...")
        model = DualBranchNetwork(num_classes=num_classes, 
                                 dropout_rate=config.dropout_rate)
        return model
    elif model_type == 'densenet':
        # Use DenseNet explicitly when requested
        print("Creating DenseNet7x7 model with 7x7 kernels...")
        model = DenseNet7x7(growth_rate=16, 
                            block_config=(3, 6, 12, 8),
                            num_classes=num_classes, 
                            dropout_rate=config.dropout_rate)
        return model
    elif model_type == 'smallresnet':
        print("Creating SmallResNet model...")
        model = SmallResNet(num_classes=num_classes, 
                           dropout_rate=config.dropout_rate)
        return model
    elif model_type == 'mobilenet':
        # Use MobileNetV3 for explicit mobilenet requests
        try:
            print("Creating MobileNetV3 model...")
            # Handle both PyTorch 1.x and 2.x style initialization
            try:
                # PyTorch 2.x style with weights parameter
                model = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if config.pretrained else None)
            except TypeError:
                # Fall back to PyTorch 1.x style with pretrained parameter
                model = models.mobilenet_v3_small(pretrained=config.pretrained)
            
            model.classifier[2] = nn.Dropout(p=config.dropout_rate)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            
            # Initialize the new layer properly
            nn.init.xavier_uniform_(model.classifier[3].weight)
            nn.init.zeros_(model.classifier[3].bias)
            
            return model
        except Exception as e:
            print(f"MobileNetV3 initialization failed: {str(e)}")
            # Fall back to DenseNet if MobileNet fails
            print("Falling back to DenseNet7x7 model...")
            return DenseNet7x7(growth_rate=16, block_config=(3, 6, 12, 8),
                              num_classes=num_classes, dropout_rate=config.dropout_rate)
    else:
        # Default to DenseNet7x7 for small images and unknown model types
        if getattr(config, 'image_size', 32) <= 32:
            print(f"Unknown model type '{model_type}' for small images, using DenseNet7x7...")
            return DenseNet7x7(growth_rate=16, block_config=(3, 6, 12, 8),
                              num_classes=num_classes, dropout_rate=config.dropout_rate)
        else:
            # For larger images, attempt to use MobileNetV3 (with fallback to DenseNet)
            print(f"Unknown model type '{model_type}' for larger images, trying MobileNetV3...")
            try:
                model = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if config.pretrained else None)
                model.classifier[2] = nn.Dropout(p=config.dropout_rate)
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
                return model
            except:
                # Ultimate fallback
                print("MobileNet failed, using DenseNet7x7 as final fallback")
                return DenseNet7x7(growth_rate=16, block_config=(3, 6, 12, 8),
                                  num_classes=num_classes, dropout_rate=config.dropout_rate)

# ### Optimized Layer-wise Learning Rate Optimizer ###
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
    
    # Check for parameters that weren't included in any group
    all_params = set(p for p in model.parameters() if p.requires_grad)
    missed_params = all_params - param_set
    
    if missed_params:
        print(f"  Found {len(missed_params)} parameters not assigned to any group, adding with base LR")
        other_params.append({'params': list(missed_params), 'lr': config.learning_rate})
    
    # Combine all parameter groups
    param_groups = feature_params + classifier_params + other_params
    
    # If param_groups is still empty, fall back to using all model parameters
    if len(param_groups) == 0:
        print("  No parameters found in named children, using all model parameters")
        return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
    
    # Create optimizer with parameter groups
    return optim.Adam(param_groups, lr=config.learning_rate, weight_decay=config.l2_reg)

# ### Training Function with AMP and Gradient Clipping ###
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device):
    """Train with mixup, early stopping, scheduler, and optimizations."""
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # Add early stopping warmup default if not in config
    early_stopping_warmup = getattr(config, 'early_stopping_warmup', 5)
    
    # History tracking for plotting
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    # Create output directory if it doesn't exist
    version_dir = os.path.join(config.output_dir, config.version)
    os.makedirs(version_dir, exist_ok=True)
  
    # Initialize gradient scaler for AMP
    scaler = GradScaler() if config.use_amp else None
    
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
            if config.use_amp:
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
        
        # Save best model (by validation accuracy)
        if val_accuracy > best_val_acc:
            print(f"  Validation accuracy improved from {best_val_acc:.4f} to {val_accuracy:.4f}")
            best_val_acc = val_accuracy
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint to version directory instead of output_dir
            checkpoint_path = os.path.join(version_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': best_val_acc,
                'history': history,
            }, checkpoint_path)
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # Early stopping check with warm-up period
        if epoch >= early_stopping_warmup:
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs (after {early_stopping_warmup} warm-up epochs)")
                break
        else:
            print(f"  Warm-up phase: {epoch+1}/{early_stopping_warmup} epochs")
    
    # Load best model before returning
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
    # This assumes val_loader.dataset is a Subset of MushroomDataset
    if hasattr(val_loader.dataset, 'dataset'):  # For Subset
        classes = val_loader.dataset.dataset.classes
    else:
        classes = val_loader.dataset.classes
    
    # Analyze false predictions
    analysis = analyze_false_predictions(all_true_labels, all_predictions, classes)
    
    # Print report
    print_false_prediction_report(analysis)
    
    # Save analysis to version directory instead of output_dir
    analysis_path = os.path.join(version_dir, 'false_prediction_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(convert_to_json_serializable(analysis), f, indent=2)
    
    # Plot confusion matrix to version directory
    cm_path = os.path.join(version_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_true_labels, all_predictions, classes, cm_path)
    
    return model, best_val_acc, history, analysis

# ### New function to plot and save training history ###
def plot_training_history(history, save_path):
    """Plot and save training metrics history."""
    plt.figure(figsize=(15, 10))
    
    # Plot training & validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    
    # Also save history as JSON for future reference
    json_path = os.path.splitext(save_path)[0] + '.json'
    with open(json_path, 'w') as f:
        json.dump(history, f)
    
    print(f"Training history saved to {save_path} and {json_path}")
    plt.close()

# ### Cross-Validation ###
def cross_validate(config, device):
    """Perform cross-validation with enhanced features."""
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Create version directory
    version_dir = os.path.join(config.output_dir, config.version)
    os.makedirs(version_dir, exist_ok=True)
    
    # Get appropriate transforms based on configuration with consistent parameters
    if getattr(config, 'use_albumentations', False) and ALBUMENTATIONS_AVAILABLE:
        print(f"Using Albumentations transforms with strength: {config.aug_strength}")
        train_transform, val_transform = get_albumentation_transforms(
            aug_strength=getattr(config, 'aug_strength', 'high'),
            image_size=config.image_size,
            multi_scale=getattr(config, 'use_multi_scale', False)
        )
    elif getattr(config, 'use_multi_scale', False):
        print("Using enhanced multi-scale transforms")
        train_transform, val_transform = get_enhanced_transforms(
            multi_scale=True,
            image_size=config.image_size,
            aug_strength=getattr(config, 'aug_strength', 'high')
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
    
    # Create a local copy of train_folds to avoid modifying the config object
    train_folds = config.train_folds

    
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
    
    for fold_idx, fold in enumerate(train_folds):
        print(f"\n===== Fold {fold+1}/{config.num_folds} ({fold_idx+1}/{len(train_folds)}) =====")
        
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
            train_dataset = CustomSubset(Subset(base_train_dataset, train_indices), train_transform)
            val_dataset = CustomSubset(Subset(base_val_dataset, val_indices), val_transform)
            
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
                model = get_model(len(dataset.class_to_idx), config).to(device)
            except Exception as e:
                print(f"Error initializing model: {str(e)}")
                print("Skipping fold due to model initialization failure")
                continue
            
            # Loss function with class weights if enabled
            if class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                print(f"Using weighted CrossEntropyLoss with weights: {class_weights.cpu().numpy()}")
            else:
                criterion = nn.CrossEntropyLoss()
                
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
            
            # Train model - updated to also return analysis
            model, val_accuracy, history, analysis = train_model(
                model,
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                scheduler, 
                config, 
                device
            )

            # Generate and save visualizations
            plot_training_history(history, os.path.join(fold_dir, 'training_history.png'))

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

# ### Inference Functions ###
def load_model_from_checkpoint(checkpoint_path, num_classes, config, device):
    """Load a model from a checkpoint file."""
    model = get_model(num_classes, config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    return model

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
            outputs = model(img_tensor)  # Fixed: was using 'inputs' variable that didn't exist
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
            'class_probabilities': class_probabilities,  # Add full probability distribution
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
        image_paths.extend(list(Path(image_dir).glob(f'*.{ext.upper()}')))
    
    # Sort paths for consistent ordering
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return [], torch.tensor([]), class_names  # Return consistent tuple structure
    
    results = []
    all_probabilities = []
    processed_count = 0
    failed_count = 0
    
    print(f"Running inference on {len(image_paths)} images in batches of {batch_size}")
    
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
                filename = Path(img_path).stem
                
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

def save_submission_csv(results, output_file):
    """Save prediction results to a submission CSV file in the format id,type."""
    df = pd.DataFrame([{
        'id': result['filename'],
        'type': result['class'] # still class, mapping to correct class_id later
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
        
        # Generate output paths
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
            save_submission_csv(results, output_submission_path)
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
                    
                    # Plot confusion matrix
                    cm_path = os.path.join(output_dir, "confusion_matrix.png")
                    plot_confusion_matrix(true_labels, predicted_labels, class_names, cm_path)
            except Exception as e:
                print(f"Error during analysis: {str(e)}")
        
        return results, all_probabilities, class_names
        
    except Exception as e:
        print(f"Inference pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], None, None

# Fix combine_fold_predictions to properly average full probability distributions
def combine_fold_predictions(fold_predictions, class_names, ensemble_method="mean"):
    """Combine predictions from multiple folds using voting or averaging of full probability distributions."""
    if not fold_predictions:
        return []
    
    # Group predictions by filename
    combined_results = {}
    
    # Process predictions from all folds
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
        # Initialize to None to avoid variable scope issues later
        class_probabilities = None
        
        try:
            if ensemble_method == "vote":
                # Use majority voting to select the final class
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
                        final_class_id = matching_preds[0]['class_id']  # Use first matching prediction's class_id
                        confidences = [p['confidence'] for p in matching_preds]
                        final_confidence = sum(confidences) / len(confidences)
                    else:
                        # This should not happen, but handle it just in case
                        print(f"Warning: No matching predictions found for voted class '{final_class}' in {filename}")
                        final_class_id = 0
                        final_confidence = 0.5
                else:
                    # No votes (should never happen)
                    print(f"Warning: No votes for {filename}, using first prediction")
                    final_class = result['fold_predictions'][0]['class']
                    final_class_id = result['fold_predictions'][0]['class_id']
                    final_confidence = result['fold_predictions'][0]['confidence']
                
            else:  # Default is "mean" - properly average full probability distributions
                # Check if we have full probability distributions
                if result['fold_class_probabilities'] and len(result['fold_class_probabilities']) > 0:
                    # Initialize an averaged probability distribution
                    avg_probs = {class_name: 0.0 for class_name in class_names}
                    
                    # Sum probabilities for each class across all folds
                    for probs in result['fold_class_probabilities']:
                        for class_name, prob in probs.items():
                            if class_name in avg_probs:  # Ensure class exists in dictionary
                                avg_probs[class_name] += prob
                    
                    # Average the summed probabilities
                    num_folds = len(result['fold_class_probabilities'])
                    for class_name in avg_probs:
                        avg_probs[class_name] /= num_folds
                    
                    # Find the class with highest average probability
                    if avg_probs:
                        final_class = max(avg_probs.items(), key=lambda x: x[1])[0]
                        final_confidence = avg_probs[final_class]
                        
                        # Find the class_id for the final class (safely)
                        try:
                            final_class_id = class_names.index(final_class)
                        except ValueError:
                            print(f"Warning: Class '{final_class}' not found in class_names. Using first class.")
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
                    # Fall back to the old method if no full distributions are available
                    print(f"Warning: No probability distributions for {filename}. Using confidence averaging.")
                    
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
                        final_class = id_to_name.get(final_class_id, class_names[final_class_id] if 0 <= final_class_id < len(class_names) else "unknown")
                    else:
                        # No scores (should never happen)
                        print(f"Warning: No valid scores for {filename}")
                        final_class = class_names[0]
                        final_class_id = 0
                        final_confidence = 0.0
                    
                    # Create an probability distribution as a fallback
                    class_probabilities = {class_name: 0.0 for class_name in class_names}
                    for cid, score in avg_scores.items():
                        class_name = id_to_name.get(cid, "")
                        if class_name in class_probabilities:
                            class_probabilities[class_name] = score
            
            # Create the final result
            result_entry = {
                'filename': filename,
                'image_path': result['image_path'],
                'class': final_class,
                'class_id': final_class_id,
                'confidence': final_confidence,
                'fold_predictions': result['fold_predictions']
            }
            
            # Add full probability distribution if available (mean method)
            if ensemble_method == "mean" and class_probabilities is not None:
                result_entry['class_probabilities'] = class_probabilities
            
            final_results.append(result_entry)
            
        except Exception as e:
            print(f"Error processing ensemble for {filename}: {str(e)}")
            # Add a basic entry so we don't lose this image in results
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
    # Convert numpy arrays to lists if needed
    if hasattr(true_labels, 'tolist'):
        true_labels = true_labels.tolist()
    if hasattr(predicted_labels, 'tolist'):
        predicted_labels.tolist()
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Calculate false predictions per class
    false_pred_per_class = {}
    
    # For each true class, count misclassifications
    for true_idx, class_name in enumerate(class_names):
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

def plot_confusion_matrix(true_labels, predicted_labels, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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

# ### Knowledge Distillation Loss ### (Commented out for now)
"""
class DistillationLoss(nn.Module):
    
    Implements knowledge distillation loss combining:
    - Standard cross-entropy loss with ground truth (hard targets)
    - KL divergence loss with teacher's soft predictions
    
    def __init__(self, temperature=3.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature  # Temperature for softening probability distributions
        self.alpha = alpha  # Weight for distillation loss vs standard CE loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_outputs, teacher_outputs, targets):
        # Standard cross-entropy with ground truth
        hard_loss = self.ce_loss(student_outputs, targets)
        
        # Knowledge distillation loss
        soft_student = nn.functional.log_softmax(student_outputs / self.temperature, dim=1)
        soft_teacher = nn.functional.softmax(teacher_outputs / self.temperature, dim=1)
        distill_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Combine losses
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * distill_loss
        
        return total_loss
"""

# ### Knowledge Distillation Training Function ### (Commented out for now)
"""
def train_with_distillation(student_model, teacher_model, train_loader, val_loader, 
                           optimizer, scheduler, config, device):
    # ...existing code...
"""

def evaluate_model(model, data_loader, criterion, device, num_classes=None):
    """Evaluate model performance on a dataset with comprehensive metrics."""
    # Set model to evaluation mode
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
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
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

# Update main function to use enhanced features but skip distillation
def format_time(seconds):
    """Format seconds into hours, minutes, seconds string."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

def main():
    """Run both training and inference in a single pipeline."""
    try:
        # Initialize enhanced configuration
        config = EnhancedConfig()
        if config.debug:
            print("WARNING: THIS IS DEBUG MODE")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Running experiment version: {config.version}")
        
        # Create output directory
        version_dir = os.path.join(config.output_dir, config.version)
        os.makedirs(version_dir, exist_ok=True)    
        
        # Save config to JSON for reproducibility with better error handling
        try:
            with open(os.path.join(version_dir, 'config.json'), 'w') as f:
                # Handle all potentially non-serializable types, not just Path
                config_dict = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None), list, dict)) else v 
                              for k, v in config.__dict__.items()}
                json.dump(config_dict, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save configuration to JSON: {str(e)}")
        
        # Save start time for benchmarking
        start_time = time.time()
        
        # Configure transforms at the global scope rather than locally
        if getattr(config, 'use_albumentations', False) and ALBUMENTATIONS_AVAILABLE:
            print(f"Using Albumentations augmentation with {config.aug_strength} strength")
            # Use global variables to properly override transformations
            train_transform, val_transform = get_albumentation_transforms(
                aug_strength=getattr(config, 'aug_strength', 'high'), 
                image_size=config.image_size,
                multi_scale=getattr(config, 'use_multi_scale', False)
            )
        elif getattr(config, 'use_multi_scale', False):
            print("Using multi-scale training transforms")
            train_transform, val_transform = get_enhanced_transforms(
                multi_scale=True,
                image_size=config.image_size,
                aug_strength=getattr(config, 'aug_strength', 'high')
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

        # Save combined analysis if we have results
        if analyses:
            try:
                combined_analysis_path = os.path.join(version_dir, 'combined_analysis.json')
                with open(combined_analysis_path, 'w') as f:
                    json.dump(convert_to_json_serializable(analyses), f, indent=2)
                    
                # Print summary of problematic classes across folds
                print("\n=== Summary of Problematic Classes Across Folds ===")
                class_problem_scores = {}
                
                for fold, analysis in analyses.items():
                    if not analysis or 'per_class' not in analysis:
                        continue
                        
                    for class_name, stats in analysis['per_class'].items():
                        if class_name not in class_problem_scores:
                            class_problem_scores[class_name] = {'total_false': 0, 'count': 0}
                        
                        class_problem_scores[class_name]['total_false'] += stats.get('false', 0)
                        class_problem_scores[class_name]['count'] += 1
                
                # Calculate average false predictions per class
                if class_problem_scores:
                    for class_name, stats in class_problem_scores.items():
                        stats['avg_false'] = stats['total_false'] / max(stats['count'], 1)  # Avoid division by zero
                    
                    # Sort by average false predictions (descending)
                    sorted_problems = sorted(class_problem_scores.items(), 
                                          key=lambda x: x[1]['avg_false'], reverse=True)
                    
                    print("\nClasses sorted by average false predictions:")
                    for class_name, stats in sorted_problems:
                        print(f"  {class_name}: {stats['avg_false']:.2f} avg false predictions")
            except Exception as e:
                print(f"Error generating analysis summary: {str(e)}")
        else:
            print("No analysis results available from cross-validation")
        
        # Report training time
        train_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(train_time)}")
        
        # === Inference Phase ===
        if config.run_inference_after_training:
            if not config.inference_input_path or not os.path.exists(config.inference_input_path):
                print(f"Warning: Inference path not found or not specified: {config.inference_input_path}")
            else:
                print("\n=== Starting Inference Phase ===")
                
                # Load class names from dataset once for efficiency
                try:
                    dataset = MushroomDataset(config.csv_path, transform=None)
                    class_names = dataset.classes
                except Exception as e:
                    print(f"Error loading dataset for class names: {str(e)}")
                    print("Using default class names for inference")
                    class_names = CLASS_NAMES  # Fall back to global constants
                
                # Run inference for each trained fold model
                all_fold_results = []
                
                for fold in config.train_folds:
                    print(f"\n--- Running inference with fold {fold+1} model ---")
                    fold_dir = os.path.join(version_dir, f'fold_{fold}')
                    model_path = os.path.join(fold_dir, 'model_weights.pth')
                    
                    # Check if model exists
                    if not os.path.exists(model_path):
                        print(f"Warning: Model for fold {fold+1} not found at {model_path}, skipping.")
                        continue
                    
                    # Run inference with this fold's model
                    results, probs, _ = run_inference(
                        config, 
                        model_path, 
                        config.inference_input_path, 
                        fold_dir,  # Save in fold directory
                        device
                    )
                    
                    # Add fold information to results if we got valid results
                    if results:
                        for result in results:
                            result['fold'] = fold
                        all_fold_results.append(results)
                    else:
                        print(f"No valid results from fold {fold+1}, skipping in ensemble")
                
                # Combine predictions from all folds (if enabled and we have multiple folds with results)
                if all_fold_results:  # Make sure we have at least some results
                    if config.ensemble_method and len(all_fold_results) > 1:
                        print(f"\n--- Creating ensemble using method: {config.ensemble_method} ---")
                        
                        combined_results = combine_fold_predictions(
                            all_fold_results, 
                            class_names,
                            ensemble_method=config.ensemble_method
                        )
                        
                        if combined_results:  # Make sure ensemble produced valid results
                            # Create ensemble directory
                            ensemble_dir = os.path.join(version_dir, "ensemble")
                            os.makedirs(ensemble_dir, exist_ok=True)
                            
                            # Save combined results
                            try:
                                combined_json_path = os.path.join(ensemble_dir, "inference_results.json")
                                combined_submission_path = os.path.join(ensemble_dir, "submission.csv")
                                combined_version_submission_path = os.path.join(version_dir, "submission.csv")

                                save_inference_results(combined_results, combined_json_path)
                                save_submission_csv(combined_results, combined_submission_path)
                                save_submission_csv(combined_results, combined_version_submission_path)  # Also at version level
                                
                                print(f"Ensemble predictions from {len(all_fold_results)} models saved to {ensemble_dir}")
                                print(f"Ensemble submission also saved to {combined_version_submission_path}")
                            except Exception as e:
                                print(f"Error saving ensemble results: {str(e)}")
                        else:
                            print("Error: Ensemble produced no valid results")
                    elif not config.ensemble_method and len(config.train_folds) > 1:
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

# ### LowNet Implementation with Adaptations ###
class AdaptedLowNet(nn.Module):
    """
    Adaptation of LowNet for implement_ideas.py framework.
    
    Modifications:
    - Supports RGB input (3 channels) instead of grayscale (1 channel)
    - Allows configurable number of output classes
    - Adds configurable dropout rate
    - Uses padding to maintain spatial dimensions compatibility
    """
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(AdaptedLowNet, self).__init__()
        
        # Low-Resolution Feature Extractor (3 Conv Layers)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)