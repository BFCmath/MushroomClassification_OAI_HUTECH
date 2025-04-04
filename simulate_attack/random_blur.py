import os
import random
import numpy as np
import cv2

# --- Function: random_blur ---
def random_blur(image, max_k=5):
    """
    Applies Gaussian blur to the image with a randomly selected odd kernel size.

    Parameters:
    - image: Input image as a numpy array (BGR format from OpenCV).
    - max_k: Maximum value for k, where kernel size is (2*k+1) x (2*k+1).
             e.g., max_k=5 means kernel sizes can be 3x3, 5x5, ..., 11x11.
             Must be >= 1.

    Returns:
    - Blurred image as a numpy array.
    """
    if max_k < 1:
        print("Warning: max_k must be >= 1 for random_blur. Using max_k=1.")
        max_k = 1
        
    # Select a random k value (determines kernel size)
    # k=0 means kernel size 1x1 (no blur), so we start k from 1.
    # A higher k leads to a larger kernel and more blur.
    k = random.randint(1, max_k) 
    
    # Calculate odd kernel size (e.g., 3, 5, 7...)
    kernel_size = 2 * k + 1
    
    # Apply Gaussian blur
    # sigmaX=0 means sigma is calculated automatically from kernel size by OpenCV
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0) 
    
    return blurred_image
# --- End of random_blur function ---

# --- Main Script Logic ---

# Define input and output folders
input_folder = "test"
output_folder = "random_blur_test" # Specific output folder for this augmentation

# Create the output folder if it doesn’t exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output folder: {output_folder}")
else:
    print(f"Output folder already exists: {output_folder}")

# Get list of all .jpg files in the input folder
try:
    # Use lower() for case-insensitive matching of .jpg extension
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"Warning: No .jpg files found in the input folder: {input_folder}")
except FileNotFoundError:
    print(f"Error: Input folder not found: {input_folder}")
    exit() # Stop the script if the input folder doesn't exist

# Optional: Set a random seed for reproducibility if desired
random.seed(42) # Uncomment this line if you want the same random blur each time

print(f"\nStarting random blur augmentation...")
# Process each image
processed_count = 0
skipped_count = 0
for img_file in image_files:
    input_path = os.path.join(input_folder, img_file)
    output_path = os.path.join(output_folder, img_file)
    
    # Load the original image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Warning: Could not load image {input_path}. Skipping.")
        skipped_count += 1
        continue
    
    # Apply the random blur augmentation
    # Adjust 'max_k' to control the maximum amount of blur (higher means more blur potential)
    augmented_image = random_blur(image, max_k=5) 
    
    # Save the augmented image
    success = cv2.imwrite(output_path, augmented_image)
    if success:
        print(f"Processed and saved: {output_path}")
        processed_count += 1
    else:
        print(f"Error: Failed to save image {output_path}")
        skipped_count += 1

print(f"\nProcessing complete.")
print(f"Successfully processed: {processed_count} images.")
print(f"Skipped/Failed: {skipped_count} images.")