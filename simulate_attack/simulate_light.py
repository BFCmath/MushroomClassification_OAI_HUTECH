import os
import random
import numpy as np
import cv2

def simulate_light(image, max_add=100, sigma_factor=4, margin=0.1):
    """
    Simulate light from a bulb on the image by adding a Gaussian intensity mask
    with a randomly positioned light source.

    Parameters:
    - image: Input image as a numpy array (BGR format from OpenCV).
    - max_add: Maximum value to add to pixel intensities (controls light strength).
    - sigma_factor: Factor to control the spread of the light (higher means wider spread).
    - margin: Fraction of the image size to keep the light source away from the edges.

    Returns:
    - Augmented image as a numpy array.
    """
    height, width = image.shape[:2]
    
    # Calculate margins to keep the light source away from the edges
    min_x = int(width * margin)
    max_x = int(width * (1 - margin))
    min_y = int(height * margin)
    max_y = int(height * (1 - margin))
    
    # Randomly select the light source position
    cx = random.randint(min_x, max_x)  # x-coordinate
    cy = random.randint(min_y, max_y)  # y-coordinate
    
    # Define the Gaussian spread based on image size
    sigma = min(width, height) / sigma_factor
    
    # Create coordinate grids
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    
    # Generate the Gaussian mask centered at (cx, cy)
    M = max_add * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    
    # Apply the mask to the image and clip values to [0, 255]
    augmented_image = np.clip(image + M[:, :, None], 0, 255).astype(np.uint8)
    return augmented_image

# Define input and output folders
input_folder = "test"
output_folder = "attack_dataset/simulate_light_test"

# Create the output folder if it doesn’t exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of all .jpg files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

# Optional: Set a random seed for reproducibility
random.seed(42)  # Remove this line if you want different results each run

# Process each image
for img_file in image_files:
    input_path = os.path.join(input_folder, img_file)
    output_path = os.path.join(output_folder, img_file)
    
    # Load the original image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Warning: Could not load image {input_path}. Skipping.")
        continue
    
    # Apply the light simulation with a random light source position
    augmented_image = simulate_light(image, max_add=100, sigma_factor=4, margin=0.1)
    
    # Save the augmented image
    cv2.imwrite(output_path, augmented_image)
    print(f"Processed and saved: {output_path}")