import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Paste the CutBlur class here
class CutBlur:
    """
    Randomly cut a rectangular region, blur it and put it back into the original image.
    Good for simulating local focus issues in photographs.
    
    Args:
        blur_strength: Strength of the blur (sigma value for Gaussian blur)
        min_size: Minimum size of the cut region relative to image size (0.1 = 10%)
        max_size: Maximum size of the cut region relative to image size (0.5 = 50%)
        p: Probability of applying transform
    """
    def __init__(self, blur_strength=3.0, min_size=0.1, max_size=0.4, p=0.3):
        self.blur_strength = blur_strength
        self.min_size = min_size
        self.max_size = max_size
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            # Convert to numpy array for processing
            img_array = np.array(img)
            height, width = img_array.shape[:2]
            
            # Determine cut region size
            cut_ratio = random.uniform(self.min_size, self.max_size)
            cut_width = int(width * cut_ratio)
            cut_height = int(height * cut_ratio)
            
            # Determine random position for the cut
            x = random.randint(0, max(0, width - cut_width))
            y = random.randint(0, max(0, height - cut_height))
            
            # Extract the region to blur
            region = img_array[y:y+cut_height, x:x+cut_width]
            
            # Apply Gaussian blur to the region
            blurred_region = cv2.GaussianBlur(region, (0, 0), self.blur_strength)
            
            # Put blurred region back into the original image
            result = img_array.copy()
            result[y:y+cut_height, x:x+cut_width] = blurred_region
            
            return Image.fromarray(result)
        
        return img

# Function to apply CutBlur and display results
def apply_and_display_cutblur(image_path, configs):
    # Load the original image
    original_img = Image.open(image_path).convert('RGB')
    
    # Create a figure for displaying images
    num_configs = len(configs)
    fig, axes = plt.subplots(1, num_configs + 1, figsize=(5 * (num_configs + 1), 5))
    
    # Display original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Apply CutBlur with different configurations
    for i, config in enumerate(configs):
        cutblur = CutBlur(**config)
        # Apply transformation (run multiple times if p < 1 to ensure effect)
        transformed_img = original_img
        attempts = 0
        max_attempts = 10  # Prevent infinite loops
        while transformed_img == original_img and attempts < max_attempts:
            transformed_img = cutblur(original_img)
            attempts += 1
        
        axes[i + 1].imshow(transformed_img)
        title = f"Blur: {config['blur_strength']}, p: {config['p']}\nMin: {config['min_size']}, Max: {config['max_size']}"
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Define different configurations to test
configs = [
    {'blur_strength': 1.0, 'min_size': 0.4, 'max_size': 0.5, 'p': 1.0},  # Default with guaranteed application
    {'blur_strength': 1.2, 'min_size': 0.4, 'max_size': 0.5, 'p': 1.0},  # Stronger blur, larger region
    {'blur_strength': 2.0, 'min_size': 0.05, 'max_size': 0.2, 'p': 0.5}, # Subtle blur, smaller region, 50% chance
]

# Run the script
image_path = 'augmentation/example.jpg'
apply_and_display_cutblur(image_path, configs)