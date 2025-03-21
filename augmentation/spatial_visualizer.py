import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from augmentation.transforms import AdvancedSpatialTransforms

def visualize_crop_and_zoom(image_path, output_dir, crop_scales=None):
    """Visualize the crop and zoom augmentation with different crop scales."""
    if crop_scales is None:
        crop_scales = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_name = Path(image_path).stem
    
    # Create a grid for visualization
    rows = 2
    cols = len(crop_scales) // 2 + (1 if len(crop_scales) % 2 else 0)
    
    fig, ax = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    fig.suptitle(f'Crop and Zoom Effect: {image_name}', fontsize=16)
    
    # Display original image in first position
    ax[0, 0].imshow(image)
    ax[0, 0].set_title('Original', fontsize=12)
    ax[0, 0].axis('off')
    
    # Display augmented images
    idx = 1
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                continue  # Skip the first position (original image)
                
            if idx < len(crop_scales) + 1:
                scale = crop_scales[idx-1]
                
                # Apply crop and zoom
                augmenter = AdvancedSpatialTransforms.RandomCropAndZoom(crop_scale=scale, p=1.0)
                augmented = augmenter(image)
                
                # Display image
                ax[i, j].imshow(augmented)
                ax[i, j].set_title(f'Crop scale: {scale}', fontsize=12)
                ax[i, j].axis('off')
                
                # Save individual augmented images
                output_file = os.path.join(output_dir, f"{image_name}_crop_scale_{scale:.2f}.jpg")
                augmented.save(output_file)
                
                idx += 1
            else:
                # Hide empty subplot
                ax[i, j].axis('off')
    
    # Save the grid figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    grid_output = os.path.join(output_dir, f"{image_name}_crop_zoom_grid.jpg")
    plt.savefig(grid_output, dpi=150)
    plt.close()
    
    print(f"Crop and zoom visualization saved to {output_dir}")
    return grid_output

if __name__ == "__main__":
    # Get the example image path
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example.jpg")
    
    # If the example image doesn't exist, create a placeholder
    if not os.path.exists(image_path):
        print(f"Example image not found at {image_path}. Creating a placeholder...")
        placeholder = Image.new('RGB', (256, 256), color='white')
        
        # Draw something on the placeholder
        from PIL import ImageDraw
        
        # Create a drawing context
        draw = ImageDraw.Draw(placeholder)
        
        # Draw a colorful pattern with a centered element
        for i in range(0, 256, 32):
            color = (
                np.random.randint(50, 200),
                np.random.randint(50, 200),
                np.random.randint(50, 200)
            )
            draw.rectangle([(i, i), (256-i, 256-i)], outline=color, width=3)
        
        # Add a recognizable center element to better show the zoom effect
        draw.ellipse([100, 100, 156, 156], fill=(255, 0, 0))
        
        placeholder.save(image_path)
        print(f"Created placeholder image at {image_path}")
    
    # Set the output directory for visualizations
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spatial_examples")
    
    # Visualize crop and zoom augmentations
    crop_scales = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6]
    grid_path = visualize_crop_and_zoom(image_path, output_dir, crop_scales)
    
    # Try to open the saved grid image
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(grid_path)}")
    except:
        print(f"Saved grid image to: {grid_path}")
