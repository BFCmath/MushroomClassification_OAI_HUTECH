import os 
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import torch 
from PIL import Image
from pathlib import Path
import random
from transforms import AlbumentationsWrapper
class MushroomDataset(Dataset):
    """Dataset class for loading images with robust error handling."""
    def __init__(self, csv_file, transform=None):
        # Validate CSV file
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file does not exist: {csv_file}")
        self.data = pd.read_csv(csv_file)
        
        # Check required columns and empty data
        required_columns = ['image_path', 'class_name']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError("CSV must contain 'image_path' and 'class_name' columns")
        if len(self.data) == 0:
            raise ValueError("CSV file is empty")

        self.transform = transform
        self.classes = sorted(self.data['class_name'].unique())
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.error_count = 0  # Track error count
        print(f"Loaded dataset with {len(self.data)} samples and {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Try loading with retries
        max_retries = 1
        for attempt in range(max_retries):
            try:
                # Get image path with correct platform handling
                current_idx = (idx + attempt) % len(self.data)
                img_path = str(Path(self.data.iloc[current_idx]['image_path'])).replace('\\', '/')
                
                # Load image and label
                image = Image.open(img_path).convert('RGB')
                label = self.class_to_idx[self.data.iloc[current_idx]['class_name']]
                
                # Apply transformations
                if self.transform:
                    if isinstance(self.transform, AlbumentationsWrapper):
                        image = np.array(image)
                    image = self.transform(image)
                
                return image, label
                
            except Exception as e:
                self.error_count += 1
                print(f"Error loading image at index {current_idx}: {str(e)}")
        
        # If all retries failed, use a placeholder
        print(f"All {max_retries} attempts to load valid image failed, starting at idx {idx}")
        # Create a recognizable pattern with correct image size
        placeholder = torch.ones((3, 32, 32)) * 0.1 if self.transform else Image.new('RGB', (32, 32), color=(25, 25, 25))
        # Use a random valid label to avoid biasing the model
        random_label = random.randint(0, len(self.classes) - 1)
        return placeholder, random_label
