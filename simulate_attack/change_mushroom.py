import os
import random
import numpy as np
import cv2
import colorsys 

# --- Helper Function: Create Background/Foreground Mask (Heuristic) ---
# (Keep the create_background_mask_heuristic function exactly as before)
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
        fg_mask = np.full((h,w), 255, dtype=np.uint8) # Assume all foreground if corners fail
        bg_mask = cv2.bitwise_not(fg_mask)
        return bg_mask, fg_mask
        
    avg_bg_color = np.mean(corner_pixels, axis=0)
    blurred_image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    diff = blurred_image.astype(np.float32) - avg_bg_color.astype(np.float32)
    dist_sq = np.sum(diff**2, axis=2)
    # Background is close to corner color
    background_mask = np.where(dist_sq < color_threshold**2, 255, 0).astype(np.uint8)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Foreground is the inverse
    foreground_mask = cv2.bitwise_not(background_mask)
    
    return background_mask, foreground_mask # Return both

# --- Function: Colorize Foreground with Smooth transition & MAX ALPHA ---
# (Keep the colorize_foreground_smooth_limited function exactly as before)
def colorize_foreground_smooth_limited(image, foreground_mask, target_color, 
                                       mix_factor=0.5, noise_level=20, 
                                       mask_blur_ksize=15, max_effect_alpha=0.3): # Default max effect alpha
    h, w = image.shape[:2]
    if mask_blur_ksize <= 0: mask_blur_ksize = 1
    elif mask_blur_ksize % 2 == 0: mask_blur_ksize += 1
    mix_factor = np.clip(mix_factor, 0.0, 1.0)
    max_effect_alpha = np.clip(max_effect_alpha, 0.0, 1.0)
    color_layer = np.full_like(image, target_color, dtype=np.uint8)
    if noise_level > 0:
        noise = np.random.randint(-noise_level, noise_level + 1, color_layer.shape, dtype=np.int16)
        color_layer = np.clip(color_layer.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    fully_colorized_pixels = cv2.addWeighted(
        image.astype(np.float32), 1.0 - mix_factor,
        color_layer.astype(np.float32), mix_factor, 0.0
    )
    alpha_mask_blurred = cv2.GaussianBlur(foreground_mask.astype(np.uint8), (mask_blur_ksize, mask_blur_ksize), 0)
    alpha_mask_float = alpha_mask_blurred.astype(np.float32) / 255.0
    scaled_alpha_mask = alpha_mask_float * max_effect_alpha
    scaled_alpha_mask_3d = scaled_alpha_mask[:, :, None] 
    final_image_float = (image.astype(np.float32) * (1.0 - scaled_alpha_mask_3d) +
                         fully_colorized_pixels * scaled_alpha_mask_3d)
    final_image = np.clip(final_image_float, 0, 255).astype(np.uint8)
    return final_image
    
# --- Function to generate random colors (can reuse or make specific ones) ---
# (Keep the get_random_color_for_tint function exactly as before)
def get_random_color_for_tint():
    hue = random.random() 
    saturation = random.uniform(0.4, 0.9)
    value = random.uniform(0.5, 0.9) 
    rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
    bgr_int = tuple(int(c * 255) for c in reversed(rgb_float))
    return bgr_int

# --- Main Script Logic ---

input_folder = "test"
output_folder = "colorize_mushroom_test" # Keeping the folder name

# --- Parameters to Tune ---
# Mask Generation (Crucial for identifying the mushroom):
PARAM_corner_fraction = 0.05 
PARAM_color_threshold = 50  # *** Tune this based on mask results! ***
PARAM_blur_ksize = 5        
PARAM_morph_ksize = 7       
# Mushroom Colorization Parameters:
PARAM_mushroom_mix_factor = 0.6      
PARAM_mushroom_noise_level = 30      
PARAM_mushroom_mask_blur_ksize = 11  
PARAM_mushroom_max_effect_alpha = 0.35 
PARAM_mushroom_use_random_color = True 
PARAM_mushroom_fixed_color = (0, 255, 255) # BGR: Yellow (if not using random)
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
        image_files = [] # Ensure it's an empty list if none found
except FileNotFoundError:
    print(f"Error: Input folder not found: {input_folder}")
    exit() 

random.seed(46) 
np.random.seed(46)

print(f"\nStarting mushroom colorization augmentation (Max Alpha = {PARAM_mushroom_max_effect_alpha})...")
processed_count = 0 # Counts successfully augmented images
saved_original_count = 0 # Counts images saved as original due to mask failure
load_fail_count = 0 # Counts images that couldn't be loaded
save_fail_count = 0 # Counts images that failed to save

for img_file in image_files:
    input_path = os.path.join(input_folder, img_file)
    output_path = os.path.join(output_folder, img_file)
    
    image = cv2.imread(input_path)
    if image is None:
        print(f"Warning: Could not load image {input_path}. Skipping.")
        load_fail_count += 1
        continue # Skip this file entirely if loading failed
    
    # Assume augmentation will succeed initially
    image_to_save = None 
    augmentation_applied = False
    
    # 1. Create the background/foreground mask
    _, foreground_mask_binary = create_background_mask_heuristic(
        image,
        corner_fraction=PARAM_corner_fraction,
        color_threshold=PARAM_color_threshold,
        blur_ksize=PARAM_blur_ksize,
        morph_ksize=PARAM_morph_ksize
    )
    
    # 2. Check if the mask seems valid
    non_zero_pixels = cv2.countNonZero(foreground_mask_binary)
    total_pixels = image.shape[0] * image.shape[1]
    
    # Check if mask is almost empty or almost full (heuristic failure)
    if non_zero_pixels < 0.01 * total_pixels or non_zero_pixels > 0.99 * total_pixels:
        print(f"Warning: Foreground mask for {img_file} seems unusual. Saving original image.")
        image_to_save = image # Set image_to_save to the original
        augmentation_applied = False
        # Optionally save the problematic mask for debugging:
        # mask_output_path = os.path.join(output_folder, f"mask_fail_{img_file}")
        # cv2.imwrite(mask_output_path, foreground_mask_binary)
    else:
        # Mask seems okay, proceed with augmentation
        # --- DEBUG: Save the *foreground* mask if needed ---
        # mask_output_path = os.path.join(output_folder, f"mask_foreground_{img_file}")
        # cv2.imwrite(mask_output_path, foreground_mask_binary)
        # --- End Debug ---

        # Choose the target tint color
        if PARAM_mushroom_use_random_color:
            target_color = get_random_color_for_tint()
        else:
            target_color = PARAM_mushroom_fixed_color
            
        # Apply the foreground colorization
        image_to_save = colorize_foreground_smooth_limited(
            image, 
            foreground_mask_binary, 
            target_color=target_color, 
            mix_factor=PARAM_mushroom_mix_factor, 
            noise_level=PARAM_mushroom_noise_level,
            mask_blur_ksize=PARAM_mushroom_mask_blur_ksize,
            max_effect_alpha=PARAM_mushroom_max_effect_alpha 
        )
        augmentation_applied = True # Mark that augmentation was applied
    
    # 3. Save the result (either augmented or original)
    success = cv2.imwrite(output_path, image_to_save)
    if success:
        if augmentation_applied:
            print(f"Processed and saved augmented: {output_path}")
            processed_count += 1
        else:
            print(f"Saved original (due to mask issue): {output_path}")
            saved_original_count += 1
    else:
        print(f"Error: Failed to save image {output_path}")
        save_fail_count += 1

# --- Final Summary ---
total_input_files = len(image_files)
print(f"\nProcessing complete.")
print(f"Total input files: {total_input_files}")
print(f"Successfully augmented: {processed_count} images.")
print(f"Saved as original (due to mask issue): {saved_original_count} images.")
print(f"Failed to load: {load_fail_count} images.")
print(f"Failed to save: {save_fail_count} images.")
total_output = processed_count + saved_original_count
print(f"Total files written to output folder: {total_output}")
print(f"--- NOTE: Check results carefully! Mask generation (PARAM_color_threshold={PARAM_color_threshold}) impacts results. ---")