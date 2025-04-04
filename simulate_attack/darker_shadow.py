import os
import numpy as np
import cv2

def darker_shadow(image, threshold=100, darken_factor=0.5, min_intensity=0):
    """
    Simulate deeper, blacker shadows by darkening areas of the image that are already in shadow.

    Parameters:
    - image: Input image as a numpy array (BGR format from OpenCV).
    - threshold: Pixel intensity below which areas are considered shadows (0-255).
    - darken_factor: Factor to darken shadow areas (0 < factor < 1, e.g., 0.5 for half intensity).
    - min_intensity: Minimum pixel value to avoid losing all detail (e.g., 0).

    Returns:
    - Augmented image with deeper shadows as a numpy array.
    """
    # Convert to grayscale to find shadow areas
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a mask for shadow areas (where intensity < threshold)
    shadow_mask = gray_image < threshold
    
    # Copy the image to apply the augmentation
    augmented_image = image.copy()
    
    # Darken shadow areas in each color channel (B, G, R)
    for c in range(3):
        channel = augmented_image[:, :, c]
        channel[shadow_mask] = np.clip(channel[shadow_mask] * darken_factor, min_intensity, 255)
    
    return augmented_image

# Set input and output folders
input_folder = "test"
output_folder = "attack_dataset/darker_shadow_test"

# Create output folder if it doesn’t exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get all .jpg files from the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

# Process each image
for img_file in image_files:
    input_path = os.path.join(input_folder, img_file)
    output_path = os.path.join(output_folder, img_file)
    
    # Load the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Warning: Could not load image {input_path}. Skipping.")
        continue
    
    # Apply the darker shadow augmentation
    augmented_image = darker_shadow(image, threshold=100, darken_factor=0.7, min_intensity=0)
    
    # Save the augmented image
    cv2.imwrite(output_path, augmented_image)
    print(f"Processed and saved: {output_path}")