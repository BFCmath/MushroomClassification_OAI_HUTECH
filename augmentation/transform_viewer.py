import os
import sys
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tv_transforms  # Import torchvision transforms explicitly
from torchvision.utils import save_image
from pathlib import Path
import webbrowser
from datetime import datetime

# Add parent directory to path so we can import custom transforms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transforms import (
    get_transforms, 
    get_enhanced_transforms, 
    get_albumentation_transforms,
    AdvancedColorTransforms
)

# Check if Albumentations is available
ALBUMENTATIONS_AVAILABLE = False

def apply_transform(image, transform, seed=None):
    """Apply a transform to an image with optional seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # For PIL-based transforms
    if hasattr(transform, '__module__') and 'torchvision' in transform.__module__:
        return transform(image)
    
    # For custom transforms from our module
    if hasattr(transform, '__call__'):
        try:
            return transform(image)
        except Exception as e:
            print(f"Error applying transform {transform.__class__.__name__}: {e}")
            return image
    
    return image  # Fallback if transform can't be applied

def transforms_to_pil_image(tensor):
    """Convert a tensor to a PIL Image."""
    # Clone the tensor to avoid modifying the original
    tensor = tensor.clone().detach()
    
    # Handle different tensor formats
    if tensor.ndim == 3:
        # If tensor is [C, H, W]
        if tensor.shape[0] == 3:
            # De-normalize if the tensor seems to be normalized
            if tensor.min() < 0 or tensor.max() > 1:
                # Approximately reverse the normalization with ImageNet mean/std
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor = tensor * std + mean
            
            # Ensure values are in [0, 1]
            tensor = torch.clamp(tensor, 0, 1)
            
            # Convert to PIL Image using torchvision's ToPILImage
            return tv_transforms.ToPILImage()(tensor)
    
    # Fallback: return a blank image
    return Image.new('RGB', (32, 32), color='gray')

def generate_transform_grid(image_path, output_dir, num_examples=5):
    """Generate a grid of transformed images and save them with an HTML viewer."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the example image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image with size: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Create a dictionary to store all our transform types
    transform_sets = {
        "Standard": get_transforms(image_size=32)[0],
        "Enhanced": get_enhanced_transforms(image_size=32, pixel_percent=0.02)[0],
        "Multi-Scale": get_enhanced_transforms(multi_scale=True, image_size=32, pixel_percent=0.02)[0]
    }
    
    # Try to add Albumentations transforms if available
    if ALBUMENTATIONS_AVAILABLE:
        try:
            transform_sets["Albumentations Low"] = get_albumentation_transforms(
                aug_strength="low", image_size=32)[0]
            transform_sets["Albumentations High"] = get_albumentation_transforms(
                aug_strength="high", image_size=32)[0]
        except Exception as e:
            print(f"Albumentations transforms not available: {str(e)}")
    
    # Add individual advanced transforms
    advanced_transforms = {
        "RandomPixelNoise": AdvancedColorTransforms.RandomPixelNoise(p=1.0, percent_pixels=0.02),
        "RandomColorDrop": AdvancedColorTransforms.RandomColorDrop(p=1.0),
        "RandomChannelSwap": AdvancedColorTransforms.RandomChannelSwap(p=1.0),
        "RandomGamma": AdvancedColorTransforms.RandomGamma(p=1.0),
        "SimulateLighting": AdvancedColorTransforms.SimulateLightingCondition(p=1.0),
        "SimulateHSVNoise": AdvancedColorTransforms.SimulateHSVNoise(p=1.0),
        "RandomGrayscale": AdvancedColorTransforms.RandomGrayscale(p=1.0)
    }
    
    # Create a results dictionary to store images
    results = {
        "Original": {"image": image, "samples": [image]}
    }
    
    # Apply each transform set multiple times
    for name, transform in transform_sets.items():
        samples = []
        for i in range(num_examples):
            try:
                # Apply the transform
                transformed = apply_transform(image, transform, seed=i+42)
                
                # If it's a tensor, convert to PIL
                if isinstance(transformed, torch.Tensor):
                    transformed = transforms_to_pil_image(transformed)
                    
                samples.append(transformed)
            except Exception as e:
                print(f"Error applying {name} transform: {e}")
                samples.append(image)  # Use original on error
        
        results[name] = {
            "image": samples[0] if samples else image,
            "samples": samples
        }
    
    # Apply each individual advanced transform
    for name, transform in advanced_transforms.items():
        samples = []
        for i in range(num_examples):
            try:
                # Apply the transform
                transformed = apply_transform(image, transform, seed=i+42)
                samples.append(transformed)
            except Exception as e:
                print(f"Error applying {name} transform: {e}")
                samples.append(image)  # Use original on error
        
        results[name] = {
            "image": samples[0] if samples else image,
            "samples": samples
        }
    
    # Save images and generate HTML
    html_content = generate_html(results, output_dir)
    
    # Save HTML file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"transforms_preview_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Transform preview saved to {html_path}")
    
    # Open in browser
    webbrowser.open(f"file://{os.path.abspath(html_path)}")
    
    return html_path

def generate_html(results, output_dir):
    """Generate HTML to display the transform results."""
    # Create images directory inside output_dir
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Save images
    img_paths = {}
    for transform_name, data in results.items():
        img_paths[transform_name] = []
        
        # Save main image
        safe_name = transform_name.replace(" ", "_")
        main_path = f"images/{safe_name}_main.jpg"
        data["image"].save(os.path.join(output_dir, main_path))
        
        # Save sample images
        for i, sample in enumerate(data["samples"]):
            sample_path = f"images/{safe_name}_sample_{i}.jpg"
            sample.save(os.path.join(output_dir, sample_path))
            img_paths[transform_name].append(sample_path)
    
    # Generate HTML content
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Transform Preview</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .transform-section {{
                margin-bottom: 30px;
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .sample-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 10px;
                margin-top: 15px;
            }}
            .sample-image {{
                width: 100%;
                border-radius: 4px;
                border: 1px solid #ddd;
            }}
            .original-section {{
                background-color: #f0f8ff;
            }}
            .transform-name {{
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .timestamp {{
                color: #666;
                font-size: 0.8em;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>Image Transform Preview</h1>
        <p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    """
    
    # Add original image section
    html += f"""
        <div class="transform-section original-section">
            <h2>Original Image</h2>
            <img src="{img_paths['Original'][0]}" alt="Original Image" style="max-width: 100%; max-height: 300px;">
        </div>
    """
    
    # Add transform group sections
    html += """
        <h2>Standard Transform Groups</h2>
    """
    
    for transform_name in ["Standard", "Enhanced", "Multi-Scale", "Albumentations Low", "Albumentations High"]:
        if transform_name not in img_paths:
            continue
            
        html += f"""
        <div class="transform-section">
            <h3>{transform_name} Transforms</h3>
            <div class="sample-grid">
        """
        
        for img_path in img_paths[transform_name]:
            html += f"""
                <div>
                    <img src="{img_path}" alt="{transform_name} sample" class="sample-image">
                </div>
            """
        
        html += """
            </div>
        </div>
        """
    
    # Add individual transform sections
    html += """
        <h2>Individual Advanced Transforms</h2>
    """
    
    for transform_name, paths in img_paths.items():
        if transform_name in ["Original", "Standard", "Enhanced", "Multi-Scale", "Albumentations Low", "Albumentations High"]:
            continue
            
        html += f"""
        <div class="transform-section">
            <h3>{transform_name}</h3>
            <div class="sample-grid">
        """
        
        for img_path in paths:
            html += f"""
                <div>
                    <img src="{img_path}" alt="{transform_name} sample" class="sample-image">
                </div>
            """
        
        html += """
            </div>
        </div>
        """
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    return html

if __name__ == "__main__":
    # Get the example image path
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example.jpg")
    
    # If the example image doesn't exist, create a placeholder
    if not os.path.exists(image_path):
        print(f"Example image not found at {image_path}. Creating a placeholder...")
        placeholder = Image.new('RGB', (32, 32), color='white')
        
        # Draw something on the placeholder
        from PIL import ImageDraw
        
        # Create a drawing context
        draw = ImageDraw.Draw(placeholder)
        
        # Draw a pattern
        for i in range(0, 32, 4):
            color = (
                np.random.randint(0, 200),
                np.random.randint(0, 200),
                np.random.randint(0, 200)
            )
            draw.rectangle([(i, i), (32-i, 32-i)], outline=color)
        
        placeholder.save(image_path)
        print(f"Created placeholder image at {image_path}")
    
    # Set the output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transform_previews")
    
    # Generate the transform grid
    generate_transform_grid(image_path, output_dir, num_examples=5)
