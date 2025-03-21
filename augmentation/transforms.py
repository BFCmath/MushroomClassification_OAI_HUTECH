import numpy as np
import random
from PIL import Image
import cv2
from PIL import ImageOps, ImageEnhance
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn as nn
import torch
# Add Albumentations for advanced augmentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("Albumentations not available. Using torchvision transforms only.")
    ALBUMENTATIONS_AVAILABLE = False
 
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
def get_enhanced_transforms(multi_scale=False, image_size=32, pixel_percent = 0.05):
    """
    Creates enhanced transforms specifically designed for low-resolution images
    with options for multi-scale training and advanced color augmentation.
    """
    # Base transformations for low-resolution images
    base_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        
        # Add new spatial transformation for crop and zoom
        AdvancedSpatialTransforms.RandomCropAndZoom(crop_scale=0.9, p=0.3),
        
        # Basic color jitter - keep this as it works well with other transforms
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        
        # Advanced color transformations
        AdvancedColorTransforms.RandomGrayscale(p=0.1),
        AdvancedColorTransforms.RandomColorDrop(p=0.1),
        AdvancedColorTransforms.RandomChannelSwap(p=0.1),
        AdvancedColorTransforms.RandomGamma(gamma_range=(0.7, 1.3), p=0.2),
        AdvancedColorTransforms.SimulateLightingCondition(p=0.2),
        AdvancedColorTransforms.SimulateHSVNoise(p=0.2),
        AdvancedColorTransforms.RandomPixelNoise(p=0.2, percent_pixels=pixel_percent),
        
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
        return get_enhanced_transforms(multi_scale=multi_scale, image_size=image_size)
    
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
