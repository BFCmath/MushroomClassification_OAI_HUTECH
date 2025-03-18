import os
import json
import csv
import pandas as pd
from pathlib import Path
import argparse

def create_single_cv_csv(json_path, image_dir, output_csv, class_map=None, kaggle_path_prefix='/kaggle/input/aio-hutech/train'):
    """
    Create a single cross-validation CSV file based on the provided JSON classification file.
    
    Args:
        json_path: Path to the JSON file containing validation set entries
        image_dir: Directory containing all image files
        output_csv: Path to save the output CSV file
        class_map: Optional dictionary mapping class names to class IDs
        kaggle_path_prefix: Path prefix to use in the output CSV for Kaggle compatibility
    """
    print(f"Creating single cross-validation CSV...")
    print(f"Reading validation set from: {json_path}")
    print(f"Scanning image directory: {image_dir}")
    print(f"Using Kaggle path prefix: {kaggle_path_prefix}")
    
    # Read the JSON file with validation set entries
    with open(json_path, 'r') as f:
        val_set = json.load(f)
    
    print(f"Found {len(val_set)} validation set entries")
    
    # Get all image files from the directory
    image_files = []
    image_path = Path(image_dir)
    
    # Look for images in the main directory and subdirectories
    for img_path in image_path.glob("**/*"):
        if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.append(img_path)
    
    print(f"Found {len(image_files)} total image files")
    
    # List to store CSV rows
    csv_data = []
    val_count = 0
    train_count = 0
    
    # Process each image
    for img_path in image_files:
        # Get image ID from filename (without extension)
        img_id = img_path.stem
        
        # Default class from parent directory name
        class_name = img_path.parent.name
        
        # Determine if this image is in validation set
        is_val = img_id in val_set
        
        # Get specific class label from validation set if available
        if is_val:
            # Extract class from validation entry (e.g., "val_bn" -> "bn")
            val_class = val_set[img_id]
            if val_class.endswith('_val'):
                # Remove 'val_' prefix to get the actual class
                class_name = val_class[:-4]
            else:
                class_name = val_class
            val_count += 1
        else:
            # For training images, extract class from filename prefix or directory
            # If img_id starts with letters (like "BN" in "BN001"), use that as class
            prefix = ''.join([c for c in img_id if c.isalpha()]).lower()
            if prefix:
                class_name = prefix
            train_count += 1
        
        # Create the Kaggle-compatible image path
        # Format: /kaggle/input/aio-hutech/train/nấm mỡ/NM001.jpg
        relative_path = img_path.relative_to(image_path)
        kaggle_image_path = f"{kaggle_path_prefix}/{str(relative_path).replace('\\', '/')}"
        
        # Create CSV entry
        entry = {
            'image_path': kaggle_image_path,
            'class_name': class_name,
            'fold': 0 if is_val else 1,  # 0 for validation, 1 for training
            'split': 'val' if is_val else 'train'
        }
        
        # Add class ID if class map is provided
        if class_map and class_name in class_map:
            entry['class_id'] = class_map[class_name]
        
        csv_data.append(entry)
    
    print(f"Generated {len(csv_data)} entries: {train_count} training, {val_count} validation")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv, index=False)
    
    print(f"CSV file saved to: {output_csv}")
    print(f"Class distribution:")
    print(df['class_name'].value_counts())
    print("\nSplit distribution:")
    print(df['split'].value_counts())
    
    # Print some example paths to verify
    if len(csv_data) > 0:
        print("\nExample image paths:")
        for i in range(min(5, len(csv_data))):
            print(f"  {csv_data[i]['image_path']}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Create single cross-validation CSV file')
    parser.add_argument('--json', default='single_cv_classifications.json', 
                        help='Path to JSON file with validation set entries')
    parser.add_argument('--image_dir', default='../train', 
                        help='Directory containing image files')
    parser.add_argument('--output', default='train_single_cv.csv', 
                        help='Output CSV file path')
    parser.add_argument('--kaggle_prefix', default='/kaggle/input/aio-hutech/train',
                        help='Path prefix to use for Kaggle compatibility')
    
    args = parser.parse_args()
    
    # Define the base paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, args.json)
    image_dir = os.path.abspath(os.path.join(base_dir, args.image_dir))
    output_csv = os.path.join(base_dir, args.output)
    
    # You can define a class map here if needed
    class_map = {
        'bn': 1,  # bào ngư
        'dg': 2,  # đùi gà
        'lc': 3,  # linh chi
        'nm': 0   # nấm mỡ
    }
    
    create_single_cv_csv(json_path, image_dir, output_csv, class_map, args.kaggle_prefix)

if __name__ == '__main__':
    main()
