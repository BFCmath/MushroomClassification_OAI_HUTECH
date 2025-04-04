import os
import sys
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from pathlib import Path
import webbrowser
from datetime import datetime
from torchvision import transforms

# Import our advanced spatial transforms
from advanced_spatial_transforms import MushroomSpatialTransforms

def apply_transform(image, transform, seed=None):
    """Apply a transform to an image with optional seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    return transform(image)

def generate_transform_grid(image_path, output_dir, num_examples=5):
    """Generate a grid of transformed images and save them with an HTML viewer."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Load the example image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image with size: {image.size}")
        
        # Resize to ensure we have a good size for demo
        if min(image.size) > 200:
            image = image.resize((128, 128), Image.BICUBIC)
            print(f"Resized to {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        # Create a dummy image
        image = Image.new('RGB', (128, 128), color=(200, 200, 200))
        # Add a simple pattern to make transformations visible
        for i in range(0, 128, 20):
            for j in range(0, 128, 20):
                color = (100, 150, 100) if (i+j) % 40 == 0 else (150, 100, 100)
                for x in range(i, min(i+10, 128)):
                    for y in range(j, min(j+10, 128)):
                        image.putpixel((x, y), color)
        print("Created dummy image")
    
    # Create a dictionary of transforms to demo
    mushroom_transforms = {
        "RadialDistortion": MushroomSpatialTransforms.RadialDistortion(p=1.0),
        "ElasticDeformation": MushroomSpatialTransforms.ElasticDeformation(p=1.0),
        "CentralFocusZoom": MushroomSpatialTransforms.CentralFocusZoom(p=1.0),
        "AspectRatioVariation": MushroomSpatialTransforms.AspectRatioVariation(p=1.0),
        "GridShuffle": MushroomSpatialTransforms.GridShuffle(p=1.0),
        "PolarTransform": MushroomSpatialTransforms.PolarTransform(p=1.0),
        "StructuredOcclusion": MushroomSpatialTransforms.StructuredOcclusion(p=1.0),
        "ThinPlateSpline": MushroomSpatialTransforms.ThinPlateSpline(p=1.0)
    }
    
    # Apply each transform multiple times
    results = {
        "Original": {"image": image, "samples": [image] * num_examples}
    }
    
    for transform_name, transform_fn in mushroom_transforms.items():
        samples = []
        for i in range(num_examples):
            try:
                # Apply the transform with a different seed each time
                transformed = apply_transform(image, transform_fn, seed=i+42)
                samples.append(transformed)
            except Exception as e:
                print(f"Error applying {transform_name} transform: {e}")
                samples.append(image)  # Use original on error
        
        results[transform_name] = {
            "image": samples[0] if samples else image,
            "samples": samples
        }
    
    # Generate HTML to display results
    html_content = generate_html(results, output_dir)
    
    # Save HTML file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"mushroom_transforms_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Transform preview saved to {html_path}")
    
    # Open in browser
    webbrowser.open(f"file://{os.path.abspath(html_path)}")
    
    return html_path

def generate_html(results, output_dir):
    """Generate HTML to display the transform results."""
    # Save images
    img_paths = {}
    for transform_name, data in results.items():
        img_paths[transform_name] = []
        
        # Save sample images
        for i, sample in enumerate(data["samples"]):
            sample_path = f"images/{transform_name}_sample_{i}.jpg"
            sample.save(os.path.join(output_dir, sample_path))
            img_paths[transform_name].append(sample_path)
    
    # Generate HTML content
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mushroom Spatial Transform Demo</title>
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
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
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
            .transform-description {{
                color: #555;
                margin: 10px 0 20px;
                font-size: 0.9em;
                line-height: 1.4;
            }}
            .timestamp {{
                color: #666;
                font-size: 0.8em;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>Mushroom Spatial Transform Demo</h1>
        <p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="transform-section original-section">
            <h2>Original Image</h2>
            <img src="{img_paths['Original'][0]}" alt="Original Image" style="max-width: 100%; max-height: 300px;">
        </div>
    """
    
    # Add descriptions for each transform
    transform_descriptions = {
        "RadialDistortion": "Applies barrel or pincushion distortion to simulate lens effects and natural cap curvature variations.",
        "ElasticDeformation": "Applies elastic warping to simulate natural growth variations while preserving mushroom structure.",
        "CentralFocusZoom": "Selectively zooms on the center region to emphasize the cap-stem junction while maintaining periphery.",
        "AspectRatioVariation": "Applies subtle changes to height/width ratio to simulate different growth stages.",
        "GridShuffle": "Divides image into grid cells and randomly shuffles them, forcing the model to focus on local patterns.",
        "PolarTransform": "Converts image to polar coordinates and applies transformations, effective for circular caps with radial features.",
        "StructuredOcclusion": "Adds realistic occlusions mimicking forest debris or overlapping mushrooms, more contextually appropriate than random erasing.",
        "ThinPlateSpline": "Applies smooth non-linear warping to simulate realistic mushroom shape deformations while preserving continuity."
    }
    
    # Add each transform section
    for transform_name, paths in img_paths.items():
        if transform_name == "Original":
            continue
            
        description = transform_descriptions.get(transform_name, "")
        
        html += f"""
        <div class="transform-section">
            <h2>{transform_name}</h2>
            <p class="transform-description">{description}</p>
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
    # Look for the existing example.jpg file
    default_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example.jpg")
    
    # Use command line argument if provided, otherwise use default
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        image_path = sys.argv[1]
    else:
        # If default example.jpg doesn't exist, look in parent directories
        if not os.path.exists(default_image_path):
            # Look for example.jpg in parent directory
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            alt_path = os.path.join(parent_dir, "example.jpg")
            if os.path.exists(alt_path):
                default_image_path = alt_path
                print(f"Found example.jpg in parent directory: {default_image_path}")
            else:
                print(f"Warning: example.jpg not found. Using a default sample image.")
                # Just use any available test image from the script's directory
                test_images = [f for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if test_images:
                    default_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_images[0])
                    print(f"Using available image: {default_image_path}")
        
        image_path = default_image_path
        print(f"Using image: {image_path}")
    
    # Set the output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spatial_transform_demos")
    
    # Generate the transform grid
    generate_transform_grid(image_path, output_dir, num_examples=4)
