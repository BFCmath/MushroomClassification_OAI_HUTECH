import os
import pandas as pd
import numpy as np
import random
from dataclasses import dataclass
from datetime import datetime
import time
from pathlib import Path
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from PIL import Image

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ### Config Class ###
@dataclass
class Config:
    """Configuration for hyperparameters."""
    debug = False 
    version: str = "v1.0"  # Version for organizing outputs
    data_dir: str = '/kaggle/input/oai-cv/'
    csv_path: str = os.path.join(data_dir, 'train_group_cv.csv')
    output_dir: str = '/kaggle/working/'
    image_size: int = 32
    batch_size: int = 256  # Reduced from 512 for better generalization
    num_epochs: int = 500 if not debug else 2
    learning_rate: float = 0.005  # Reduced from 0.005
    dropout_rate: float = 0.2  # Increased from 0.2 for better regularization
    l2_reg: float = 0.0001  # Increased from 0.0001
    num_folds: int = 6  # Folds 0 to 6
    train_folds = [0]
    early_stopping_patience: int = 15  # Increased from 10
    mixup_alpha: float = 0.5
    mixup_prob: float = 0.4  # Increased from 0.2
    scheduler_factor: float = 0.5  # Increased from 0.1 for faster LR reduction
    scheduler_patience: int = 2  # Increased from 2
    layer_decay_rate: float = 0.95
    pretrained: bool = True  # Changed to True - using pretrained weights helps
    num_workers: int = 4
    pin_memory: bool = True        # Pin memory for faster GPU transfer
    use_amp: bool = False           # Use automatic mixed precision
    grad_clip_val: float = 10.0     # Gradient clipping value
    scheduler_type: str = "plateau" # "plateau" or "cosine"
    seed: int = 42                 # Random seed

# ### Custom Dataset with Error Handling ###
class MushroomDataset(Dataset):
    """Dataset class for loading images from train_cv.csv."""
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.classes = sorted(self.data['class_name'].unique())
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        print(f"Loaded dataset with {len(self.data)} samples and {len(self.classes)} classes")
        # Track failed images
        self.error_count = 0
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Try loading the image at the given index, if it fails try the next one
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Get the image path and handle path separators
                current_idx = (idx + attempt) % len(self.data)
                img_path = str(Path(self.data.iloc[current_idx]['image_path'])).replace('\\', '/')
                # Open the image
                image = Image.open(img_path).convert('RGB')
                # Get the label
                label = self.class_to_idx[self.data.iloc[current_idx]['class_name']]
                
                # Apply transformations
                if self.transform:
                    image = self.transform(image)
                    
                return image, label
            except Exception as e:
                self.error_count += 1
                if self.error_count % 100 == 1:  # Limit logging to avoid console spam
                    print(f"Error loading image at index {current_idx}: {str(e)}")
                # Continue to next retry attempt
        
        # If all retries failed, use a safer fallback approach
        print(f"All {max_retries} attempts to load valid image failed, starting at idx {idx}")
        # Create a recognizable pattern instead of zeros
        placeholder = torch.ones((3, 224, 224)) * 0.1 if self.transform else Image.new('RGB', (224, 224), color=(25, 25, 25))
        # Use a random valid label to avoid biasing the model
        random_label = random.randint(0, len(self.classes) - 1)
        return placeholder, random_label

# ### Enhanced Transformations ###
def get_transforms():
    # Change resize dimensions to match the actual image size
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Changed from 224x224 to 32x32
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0.1, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Changed from 224x224 to 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# ### Model Definition ###
def get_model(num_classes, config):
    """
    Use a model better suited for small images instead of MobileNetV3
    which is designed for 224x224 inputs.
    """
    if config.image_size <= 32:
        # Custom smaller model for tiny images
        model = nn.Sequential(
            # Input: 32x32x3
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            
            nn.Flatten(),  # 4x4x128 = 2048
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(512, num_classes)
        )
        return model
    else:
        # Use original MobileNetV3 model for larger images
        try:
            model = models.mobilenet_v3_small(pretrained=config.pretrained)
            model.classifier[2] = nn.Dropout(p=config.dropout_rate)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            
            # Initialize the new layer properly
            nn.init.xavier_uniform_(model.classifier[3].weight)
            nn.init.zeros_(model.classifier[3].bias)
            
            return model
        except Exception as e:
            print(f"Model initialization failed: {str(e)}")
            raise

# ### Optimized Layer-wise Learning Rate Optimizer ###
def get_layer_wise_lr_optimizer(model, config):
    """Creates an Adam optimizer with layer-wise learning rates or a regular optimizer for custom models."""
    # Print model structure for debugging
    print("Model Structure:")
    
    # Check if model is a Sequential model (our custom CNN)
    if isinstance(model, nn.Sequential):
        print("  Using standard optimizer for Sequential model")
        # For Sequential models, use standard optimizer without layer-wise LR
        return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
    
    # For models with named modules like MobileNetV3
    # Group parameters by module type for better organization
    feature_params = []
    classifier_params = []
    other_params = []  # Add a catch-all for other parameters
    
    # Organize parameters by layer type
    for name, module in model.named_children():
        print(f"Layer group: {name}")
        
        if name == 'features':
            # Apply gradually decreasing LR to feature layers
            for i, layer in enumerate(module):
                layer_lr = config.learning_rate * (config.layer_decay_rate ** i)
                print(f"  Feature block {i}: lr = {layer_lr:.6f}")
                feature_params.append({'params': layer.parameters(), 'lr': layer_lr})
        elif name == 'classifier':
            # Use base LR for classifier (final layers)
            print(f"  Classifier: lr = {config.learning_rate}")
            classifier_params.append({'params': module.parameters(), 'lr': config.learning_rate})
        else:
            # Other layers - collect their parameters too
            print(f"  {name}: lr = {config.learning_rate}")
            other_params.append({'params': module.parameters(), 'lr': config.learning_rate})
    
    # Combine all parameter groups
    param_groups = feature_params + classifier_params + other_params
    
    # If param_groups is still empty, fall back to using all model parameters
    if len(param_groups) == 0:
        print("  No parameters found in named children, using all model parameters")
        return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
    
    # Create optimizer with parameter groups
    return optim.Adam(param_groups, lr=config.learning_rate, weight_decay=config.l2_reg)

# ### Training Function with AMP and Gradient Clipping ###
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device):
    """Train with mixup, early stopping, scheduler, and optimizations."""
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # History tracking for plotting
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    # Create output directory if it doesn't exist
    version_dir = os.path.join(config.output_dir, config.version)
    os.makedirs(version_dir, exist_ok=True)
  
    # Initialize gradient scaler for AMP
    scaler = GradScaler() if config.use_amp else None
    
    # Training loop
    for epoch in range(config.num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Mixup augmentation
            use_mixup = np.random.rand() < config.mixup_prob
            if use_mixup:
                lam = np.random.beta(config.mixup_alpha, config.mixup_alpha)
                index = torch.randperm(inputs.size(0)).to(device)
                inputs_mix = lam * inputs + (1 - lam) * inputs[index]
                labels_a, labels_b = labels, labels[index]
                inputs = inputs_mix
            
            # Forward pass with optional AMP
            if config.use_amp:
                with autocast():
                    outputs = model(inputs)
                    if use_mixup:
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    else:
                        loss = criterion(outputs, labels)
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if config.grad_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_val)
                    
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training flow without AMP
                outputs = model(inputs)
                if use_mixup:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)
                
                loss.backward()
                
                # Gradient clipping
                if config.grad_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_val)
                    
                optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate training accuracy for non-mixup batches
            if not use_mixup:
                _, predicted_train = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train if total_train > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted_val = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()
        
        # Calculate validation metrics
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_val / total_val if total_val > 0 else 0
        
        # Update scheduler
        if config.scheduler_type == "plateau":
            scheduler.step(val_loss)
        elif config.scheduler_type == "cosine":
            scheduler.step()
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        history['lr'].append(current_lr)
        
        # Print epoch results
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.num_epochs} [{elapsed_time:.1f}s]")
        print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"  Valid Loss: {val_loss:.4f}, Valid Acc: {val_accuracy:.4f}")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.8f}")
        
        # Save best model (by validation accuracy)
        if val_accuracy > best_val_acc:
            print(f"  Validation accuracy improved from {best_val_acc:.4f} to {val_accuracy:.4f}")
            best_val_acc = val_accuracy
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': best_val_acc,
                'history': history,
            }, os.path.join(config.output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # Early stopping check
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation accuracy: {best_val_acc:.4f}")
    
    return model, best_val_acc, history


# ### New function to plot and save training history ###
def plot_training_history(history, save_path):
    """Plot and save training metrics history."""
    plt.figure(figsize=(15, 10))
    
    # Plot training & validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    
    # Also save history as JSON for future reference
    json_path = os.path.splitext(save_path)[0] + '.json'
    with open(json_path, 'w') as f:
        json.dump(history, f)
    
    print(f"Training history saved to {save_path} and {json_path}")
    plt.close()

# ### Cross-Validation ###
def cross_validate(config, device):
    """Perform cross-validation with enhanced features."""
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Create version directory
    version_dir = os.path.join(config.output_dir, config.version)
    os.makedirs(version_dir, exist_ok=True)
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Initialize dataset for fold information
    dataset = MushroomDataset(config.csv_path, transform=None)
    val_metrics = []
    fold_results = {}
    all_histories = {}
    
    if config.train_folds == None:
        config.train_folds = range(config.num_folds)
    
    for fold in config.train_folds:
        print(f"\n===== Fold {fold+1}/{config.num_folds} =====")
        
        # Create train/val datasets for this fold with appropriate transforms
        train_indices = dataset.data[dataset.data['fold'] != fold].index.tolist()
        val_indices = dataset.data[dataset.data['fold'] == fold].index.tolist()
        
        # Create fold directory
        fold_dir = os.path.join(version_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Create train/val datasets with proper transforms
        train_dataset = MushroomDataset(config.csv_path, transform=train_transform)
        val_dataset = MushroomDataset(config.csv_path, transform=val_transform)
        
        # Create subsets for this fold
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        
        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # Initialize model
        model = get_model(len(dataset.class_to_idx), config).to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = get_layer_wise_lr_optimizer(model, config)
        
        # Create scheduler
        if config.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=config.scheduler_factor, 
                patience=config.scheduler_patience, 
                verbose=True
            )
        elif config.scheduler_type == "cosine":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
        else:
            raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
        
        # Train model
        model, val_accuracy, history = train_model(
            model,
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            scheduler, 
            config, 
            device
        )

        plot_training_history(history, os.path.join(fold_dir, 'training_history.png'))

        # Store results
        val_metrics.append(val_accuracy)
        fold_results[fold] = val_accuracy
        all_histories[fold] = history
        
        # Save fold-specific model
        torch.save(model.state_dict(), os.path.join(fold_dir, 'model_weights.pth'))
    
    # Report cross-validation results
    avg_val_accuracy = np.mean(val_metrics)
    std_val_accuracy = np.std(val_metrics)
    print("\n===== Cross-Validation Summary =====")
    for fold, acc in fold_results.items():
        print(f"Fold {fold+1}: {acc:.4f}")
    print(f"Average: {avg_val_accuracy:.4f} ± {std_val_accuracy:.4f}")
    
    # Save overall CV results
    cv_results = {
        'fold_accuracies': fold_results,
        'mean_accuracy': float(avg_val_accuracy),
        'std_accuracy': float(std_val_accuracy),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': {k: str(v) for k, v in config.__dict__.items()},
    }
    
    with open(os.path.join(version_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=4)
    
    return avg_val_accuracy, fold_results, all_histories

# ### Helper class for applying transforms to subsets ###
class CustomSubset(Dataset):
    """Custom subset that applies transform to the original dataset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# ### Main Execution ###
def main():
    """Run the pipeline."""
    # Initialize configuration
    config = Config()
    if(config.debug):
        print("WARNING: THIS IS DEBUG MODE")
    
    # Create output directory
    version_dir = os.path.join(config.output_dir, config.version)
    os.makedirs(version_dir, exist_ok=True)    

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Running experiment version: {config.version}")
    
    # Save config to JSON for reproducibility
    with open(os.path.join(version_dir, 'config.json'), 'w') as f:
        config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}
        json.dump(config_dict, f, indent=4)
    
    # Save start time for benchmarking
    start_time = time.time()
    
    try:
        # Cross-validation
        print("\n=== Starting cross-validation ===")
        avg_val_accuracy, fold_results, cv_histories = cross_validate(config, device)
        
        # Report total execution time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise