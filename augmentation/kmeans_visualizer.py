import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from augmentation.transforms import KMeansColorAugmentation

def visualize_kmeans_augmentation(image_path, output_dir, k_values=None, blend_factors=None):
    """Visualize K-Means augmentation with different k values and blend factors."""
    if k_values is None:
        k_values = [7, 10, 15, 20]
    
    if blend_factors is None:
        blend_factors = [1.0, 0.7, 0.4]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_name = Path(image_path).stem
    
    # Create a grid of images for visualization
    rows = len(k_values)
    cols = len(blend_factors) + 1  # +1 for original image
    
    fig, ax = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    fig.suptitle(f'K-Means Color Augmentation: {image_name}', fontsize=16)
    
    # Add column and row labels
    for i, k in enumerate(k_values):
        ax[i, 0].set_ylabel(f'k={k}', fontsize=12)
    
    for j, blend in enumerate(blend_factors):
        ax[0, j+1].set_title(f'blend={blend}', fontsize=12)
    
    # Display original image in first column
    for i in range(rows):
        ax[i, 0].imshow(image)
        ax[i, 0].set_title('Original' if i == 0 else '')
        ax[i, 0].axis('off')
    
    # Display augmented images
    for i, k in enumerate(k_values):
        for j, blend in enumerate(blend_factors):
            # Apply K-Means augmentation
            augmenter = KMeansColorAugmentation(k=k, p=1.0, blend_factor=blend)
            augmented = augmenter(image)
            
            # Display image
            ax[i, j+1].imshow(augmented)
            ax[i, j+1].axis('off')
            
            # Save individual augmented images
            output_file = os.path.join(output_dir, f"{image_name}_k{k}_blend{blend:.1f}.jpg")
            augmented.save(output_file)
    
    # Save the grid figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    grid_output = os.path.join(output_dir, f"{image_name}_kmeans_grid.jpg")
    plt.savefig(grid_output, dpi=150)
    plt.close()
    
    print(f"K-Means augmentation visualization saved to {output_dir}")
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
        
        # Draw a colorful pattern
        for i in range(0, 256, 16):
            for j in range(0, 256, 16):
                color = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )
                draw.rectangle([(i, j), (i+15, j+15)], fill=color)
        
        placeholder.save(image_path)
        print(f"Created placeholder image at {image_path}")
    
    # Set the output directory for visualizations
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kmeans_examples")
    
    # Visualize K-Means augmentations
    k_values = [3, 8, 16, 32]
    blend_factors = [1.0, 0.8, 0.5, 0.3]
    grid_path = visualize_kmeans_augmentation(image_path, output_dir, k_values, blend_factors)
    
    # Try to open the saved grid image
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(grid_path)}")
    except:
        print(f"Saved grid image to: {grid_path}")
