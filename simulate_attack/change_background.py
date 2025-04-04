import os
import random
import numpy as np
import cv2
import colorsys 

# --- Helper Function: Create Background Mask (Heuristic) ---
# (Keep the create_background_mask_heuristic function as before)
def create_background_mask_heuristic(image, corner_fraction=0.1, color_threshold=40, blur_ksize=5, morph_ksize=5):
    h, w = image.shape[:2]
    blur_ksize = blur_ksize if blur_ksize % 2 != 0 else blur_ksize + 1
    morph_ksize = morph_ksize if morph_ksize % 2 != 0 else morph_ksize + 1
    cw = int(w * corner_fraction)
    ch = int(h * corner_fraction)
    corners = [
        image[0:ch, 0:cw], image[0:ch, w-cw:w],
        image[h-ch:h, 0:cw], image[h-ch:h, w-cw:w]
    ]
    corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners if c.size > 0])
    if corner_pixels.shape[0] == 0:
        fg_mask = np.full((h,w), 255, dtype=np.uint8)
        bg_mask = cv2.bitwise_not(fg_mask)
        return bg_mask, fg_mask
    avg_bg_color = np.mean(corner_pixels, axis=0)
    blurred_image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    diff = blurred_image.astype(np.float32) - avg_bg_color.astype(np.float32)
    dist_sq = np.sum(diff**2, axis=2)
    background_mask = np.where(dist_sq < color_threshold**2, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    foreground_mask = cv2.bitwise_not(background_mask)
    return background_mask, foreground_mask

# --- Function: Change Background Color with SMOOTH transition & MAX ALPHA ---
def change_background_color_smooth_limited(image, background_mask, extreme_color, 
                                           mix_factor=0.7, noise_level=20, 
                                           mask_blur_ksize=15, max_effect_alpha=1.0): # Added max_effect_alpha
    """
    Changes the background color using a blurred mask for smooth transitions,
    limiting the maximum effect by scaling the alpha.

    Parameters:
    - image: Original BGR image.
    - background_mask: Binary mask (255 for background pixels).
    - extreme_color: The BGR tuple for the new dominant background color.
    - mix_factor: How much of the extreme_color to mix into the *target* background (0.0 to 1.0).
    - noise_level: Amount of random noise (0-255) to add to the extreme color.
    - mask_blur_ksize: Kernel size for blurring the mask. Must be odd.
    - max_effect_alpha: Maximum influence of the modified background (0.0 to 1.0). 
                          1.0 = full effect in background areas, 0.2 = 20% effect max.

    Returns:
    - Augmented image.
    """
    h, w = image.shape[:2]
    
    # --- Input validation ---
    if mask_blur_ksize <= 0:
         mask_blur_ksize = 1
    elif mask_blur_ksize % 2 == 0:
        mask_blur_ksize += 1
    mix_factor = np.clip(mix_factor, 0.0, 1.0)
    max_effect_alpha = np.clip(max_effect_alpha, 0.0, 1.0) # Ensure valid range
    # --- End validation ---

    # 1. Create the target background color layer (extreme color + noise)
    new_bg_layer = np.full_like(image, extreme_color, dtype=np.uint8)
    if noise_level > 0:
        noise = np.random.randint(-noise_level, noise_level + 1, new_bg_layer.shape, dtype=np.int16)
        new_bg_layer = np.clip(new_bg_layer.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Create the "target background pixels" (what background *would* be if fully changed)
    target_background_pixels = cv2.addWeighted(
        image.astype(np.float32), 1.0 - mix_factor,
        new_bg_layer.astype(np.float32), mix_factor, 0.0
    )

    # 3. Blur the background mask to create a spatial alpha map (0.0 to 1.0)
    alpha_mask_blurred = cv2.GaussianBlur(background_mask.astype(np.uint8), (mask_blur_ksize, mask_blur_ksize), 0)
    alpha_mask_float = alpha_mask_blurred.astype(np.float32) / 255.0
    
    # --- >>> Apply the maximum effect limit <<< ---
    # Scale the spatial alpha map by the desired maximum effect alpha
    scaled_alpha_mask = alpha_mask_float * max_effect_alpha
    # --- >>> End change <<< ---

    # Add channel dimension for broadcasting (h, w) -> (h, w, 1)
    scaled_alpha_mask_3d = scaled_alpha_mask[:, :, None] 

    # 4. Perform alpha blending using the SCALED alpha mask
    # final = original * (1 - scaled_alpha) + target_background * scaled_alpha
    final_image_float = (image.astype(np.float32) * (1.0 - scaled_alpha_mask_3d) +
                         target_background_pixels * scaled_alpha_mask_3d)

    # 5. Clip values and convert back to uint8
    final_image = np.clip(final_image_float, 0, 255).astype(np.uint8)

    return final_image
    
# --- Function to generate random bright colors ---
# (Keep the get_random_bright_color function as before)
def get_random_bright_color():
    hue = random.random()
    saturation = random.uniform(0.7, 1.0)
    value = random.uniform(0.8, 1.0) 
    rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
    bgr_int = tuple(int(c * 255) for c in reversed(rgb_float))
    return bgr_int

# --- Main Script Logic ---

input_folder = "test"
# Updated folder name to reflect the change
output_folder = "change_background_test" 

# --- Parameters to Tune ---
# Heuristic Masking Parameters:
PARAM_corner_fraction = 0.05 
PARAM_color_threshold = 60  # *** Tune this first! ***
PARAM_blur_ksize = 5        
PARAM_morph_ksize = 7       
# Background Change Parameters:
PARAM_mix_factor = 0.8      # How much extreme color is mixed into the *target* bg
PARAM_noise_level = 25      
PARAM_mask_blur_ksize = 21  # Smoothness of spatial transition
PARAM_max_effect_alpha = 0.2 # <<< NEW/MODIFIED: Overall strength of the effect
PARAM_use_random_color = True 
PARAM_fixed_color = (0, 0, 255) # BGR: Red (if not using random)
# --- End Parameters ---

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output folder: {output_folder}")
else:
    print(f"Output folder already exists: {output_folder}")

try:
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"Warning: No .jpg files found in {input_folder}")
except FileNotFoundError:
    print(f"Error: Input folder not found: {input_folder}")
    exit() 

random.seed(45) # Yet another seed
np.random.seed(45)

print(f"\nStarting SMOOTH background color change augmentation (Max Alpha = {PARAM_max_effect_alpha})...")
processed_count = 0
skipped_count = 0
for img_file in image_files:
    input_path = os.path.join(input_folder, img_file)
    output_path = os.path.join(output_folder, img_file)
    
    image = cv2.imread(input_path)
    if image is None:
        print(f"Warning: Could not load image {input_path}. Skipping.")
        skipped_count += 1
        continue
    
    # 1. Create the initial binary background mask
    background_mask_binary, _ = create_background_mask_heuristic(
        image,
        corner_fraction=PARAM_corner_fraction,
        color_threshold=PARAM_color_threshold,
        blur_ksize=PARAM_blur_ksize,
        morph_ksize=PARAM_morph_ksize
    )
    
    if PARAM_use_random_color:
        extreme_color = get_random_bright_color()
    else:
        extreme_color = PARAM_fixed_color
        
    # 2. Apply the background color change using the LIMITED smooth function
    augmented_image = change_background_color_smooth_limited(
        image, 
        background_mask_binary, 
        extreme_color=extreme_color, 
        mix_factor=PARAM_mix_factor, 
        noise_level=PARAM_noise_level,
        mask_blur_ksize=PARAM_mask_blur_ksize,
        max_effect_alpha=PARAM_max_effect_alpha # Pass the max alpha limit
    )
    
    # 3. Save the augmented image
    success = cv2.imwrite(output_path, augmented_image)
    if success:
        print(f"Processed {img_file} with color {extreme_color} -> {output_path}")
        processed_count += 1
    else:
        print(f"Error: Failed to save image {output_path}")
        skipped_count += 1

print(f"\nProcessing complete.")
print(f"Successfully processed: {processed_count} images.")
print(f"Skipped/Failed: {skipped_count} images.")
print(f"--- NOTE: Max effect alpha set to {PARAM_max_effect_alpha}. Tune 'PARAM_color_threshold' for mask accuracy. ---")