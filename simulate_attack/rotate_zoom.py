import os
import random
import numpy as np
import cv2

def rotate_and_zoom(image, angle_range=(-30, 30), scale_range=(0.8, 1.2), border_mode=cv2.BORDER_REPLICATE):
    """
    Apply random rotation and zooming to an image, simulating a camera view change without black borders.

    Parameters:
    - image: Input image as a numpy array (BGR format from OpenCV).
    - angle_range: Tuple of (min, max) rotation angles in degrees (default: -30 to 30).
    - scale_range: Tuple of (min, max) scaling factors (default: 0.8 to 1.2).
    - border_mode: OpenCV border mode to handle out-of-bound pixels (default: cv2.BORDER_REPLICATE).

    Returns:
    - Augmented image with rotation and scaling applied, fully filled with content.
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Randomly select angle and scale within the specified ranges
    angle = random.uniform(angle_range[0], angle_range[1])
    scale = random.uniform(scale_range[0], scale_range[1])
    
    # Define the center point for rotation and scaling
    center = (width / 2, height / 2)
    
    # Create the transformation matrix for rotation and scaling
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply the transformation, using border replication to avoid black areas
    augmented_image = cv2.warpAffine(image, M, (width, height), borderMode=border_mode)
    
    return augmented_image

# Set input and output directories
input_folder = "test"
output_folder = "attack_dataset/rotate_zoom_test"

# Create output folder if it doesn’t exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of .jpg files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

# Set random seed for reproducibility
random.seed(42)

# Process each image
for img_file in image_files:
    input_path = os.path.join(input_folder, img_file)
    output_path = os.path.join(output_folder, img_file)
    
    # Load the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Warning: Could not load image {input_path}. Skipping.")
        continue
    
    # Apply rotation and zoom augmentation with border replication
    augmented_image = rotate_and_zoom(image, angle_range=(-90, 90), scale_range=(0.8, 1.2), border_mode=cv2.BORDER_REPLICATE)
    
    # Save the augmented image
    cv2.imwrite(output_path, augmented_image)
    print(f"Processed and saved: {output_path}")