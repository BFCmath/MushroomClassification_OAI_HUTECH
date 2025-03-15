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
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)', 'linh chi trắng']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
# ### Config Class ###
@dataclass
class Config:
    """Configuration for hyperparameters."""
    debug = False 
    version: str = "v1.0"  # Version for organizing outputs
    data_dir: str = '/kaggle/input/oai-cv/'
    csv_path: str = os.path.join(data_dir, 'train_group_cv.csv')
    output_dir: str = '/kaggle/working/'
    inference_input_path: str = '/kaggle/input/aio-hutech/test'  # Directory or file for inference
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
    run_inference_after_training: bool = True  # Whether to run inference after training
    inference_batch_size: int = 16  # Batch size for inference
    submission_path: str = None  # Will be auto-set to output_dir/submission.csv if None
    logits_path: str = None  # Will be auto-set to output_dir/logits.csv if None
    ensemble_method: str = "mean"  # How to combine multiple model predictions: "mean" or "vote"
    save_per_fold_results: bool = True  # Save results for each fold separately
    use_class_weights: bool = True  # Whether to use class weights in loss function
    weight_multiplier: float = 2.0  # How much extra weight to give to the prioritized class
    prioritized_class: str = "Đùi gà Baby (cắt ngắn)"  # Class to prioritize

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
    
    # Add these lines before returning from the function
    # Validate the best model to get predictions for analysis
    model.eval()
    all_true_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_true_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Get dataset class names for analysis
    # This assumes val_loader.dataset is a Subset of MushroomDataset
    if hasattr(val_loader.dataset, 'dataset'):  # For Subset
        classes = val_loader.dataset.dataset.classes
    else:
        classes = val_loader.dataset.classes
    
    # Analyze false predictions
    analysis = analyze_false_predictions(all_true_labels, all_predictions, classes)
    
    # Print report
    print_false_prediction_report(analysis)
    
    # Save analysis to JSON
    analysis_path = os.path.join(config.output_dir, 'false_prediction_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(convert_to_json_serializable(analysis), f, indent=2)
    
    # Plot confusion matrix
    cm_path = os.path.join(config.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_true_labels, all_predictions, classes, cm_path)
    
    return model, best_val_acc, history, analysis

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
    all_analyses = {}
    
    if config.train_folds == None:
        config.train_folds = range(config.num_folds)
    
    # Create class weights if enabled
    class_weights = None
    if config.use_class_weights:
        # Create a tensor of ones for all classes
        class_weights = torch.ones(len(dataset.classes))
        
        # If the prioritized class exists in our classes, give it more weight
        if config.prioritized_class in dataset.classes:
            priority_idx = dataset.classes.index(config.prioritized_class)
            class_weights[priority_idx] = config.weight_multiplier
            print(f"Applying weight {config.weight_multiplier} to class '{config.prioritized_class}' (index {priority_idx})")
        else:
            print(f"Warning: Prioritized class '{config.prioritized_class}' not found in dataset classes: {dataset.classes}")
            
        # Move class weights to the correct device
        class_weights = class_weights.to(device)
    
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
        
        # Loss function with class weights if enabled
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using weighted CrossEntropyLoss with weights: {class_weights.cpu().numpy()}")
        else:
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
        
        # Train model - updated to also return analysis
        model, val_accuracy, history, analysis = train_model(
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
        all_analyses[fold] = analysis
        
        # Save fold-specific model
        torch.save(model.state_dict(), os.path.join(fold_dir, 'model_weights.pth'))
        
        # Save analysis to fold directory
        analysis_path = os.path.join(fold_dir, 'false_prediction_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(convert_to_json_serializable(analysis), f, indent=2)
    
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
    
    return avg_val_accuracy, fold_results, all_histories, all_analyses

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

# ### Inference Functions ###
def load_model_from_checkpoint(checkpoint_path, num_classes, config, device):
    """Load a model from a checkpoint file."""
    model = get_model(num_classes, config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, transform=None):
    """Preprocess an image for model inference."""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Match the model's expected input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    return img_tensor

def predict_image(model, image_path, class_names, device, transform=None):
    """Run inference on a single image and return predictions."""
    img_tensor = preprocess_image(image_path, transform)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence = confidence.item()
    
    # Get top-k predictions
    topk_values, topk_indices = torch.topk(probabilities, min(3, len(class_names)))
    topk_predictions = [
        (class_names[idx.item()], prob.item()) 
        for idx, prob in zip(topk_indices[0], topk_values[0])
    ]
    
    return {
        'class': predicted_class,
        'confidence': confidence,
        'top_predictions': topk_predictions
    }

def batch_inference(model, image_dir, class_names, device, transform=None, batch_size=16):
    """Run inference on multiple images in a directory."""
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_paths.extend(list(Path(image_dir).glob(f'*.{ext}')))
        image_paths.extend(list(Path(image_dir).glob(f'*.{ext.upper()}')))
    
    # Sort paths for consistent ordering
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return []
    
    results = []
    all_probabilities = []
    
    print(f"Running inference on {len(image_paths)} images in batches of {batch_size}")
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = [preprocess_image(str(img_path), transform) for img_path in batch_paths]
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_indices = torch.max(probabilities, 1)
        
        # Store the full probabilities tensor for later processing
        all_probabilities.append(probabilities.cpu())
        
        for j, (img_path, pred_idx, conf) in enumerate(zip(batch_paths, predicted_indices, confidence)):
            predicted_class = class_names[pred_idx.item()]
            
            # Get top-k predictions for this image
            topk_values, topk_indices = torch.topk(probabilities[j], min(3, len(class_names)))
            topk_predictions = [
                (class_names[idx.item()], prob.item()) 
                for idx, prob in zip(topk_indices, topk_values)
            ]
            
            # Get image filename without extension for CSV
            filename = Path(img_path).stem
            
            results.append({
                'image_path': str(img_path),
                'filename': filename,
                'class': predicted_class,
                'class_id': pred_idx.item(),
                'confidence': conf.item(),
                'top_predictions': topk_predictions
            })
    
    # Combine all probability tensors
    if all_probabilities:
        all_probs = torch.cat(all_probabilities, dim=0)
    else:
        all_probs = torch.tensor([])
        
    return results, all_probs, class_names

def save_inference_results(results, output_file):
    """Save inference results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

def save_submission_csv(results, output_file):
    """Save prediction results to a submission CSV file in the format id,type."""
    df = pd.DataFrame([{
        'id': result['filename'],
        'type': result['class'] # still class, mapping to correct class_id later
    } for result in results])
    
    df['type'] = df['type'].map(CLASS_MAP)
    
    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")

def save_logits_csv(filenames, probabilities, class_names, output_file):
    """Save all class probabilities to a logits CSV file."""
    # Create a DataFrame with one row per image
    data = {'filename': filenames}
    
    # Add one column per class with the probability values
    for i, class_name in enumerate(class_names):
        data[class_name] = probabilities[:, i].numpy()
    
    # Create and save DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Logits saved to {output_file}")

def run_inference(config, model_path, input_path, output_dir, device):
    """Run the inference pipeline on a directory of images."""
    # Get the class names
    dataset = MushroomDataset(config.csv_path, transform=None)
    class_names = dataset.classes
    
    # Load the model
    model = load_model_from_checkpoint(model_path, len(class_names), config, device)
    
    # Create validation transform
    _, val_transform = get_transforms()
    
    # Ensure input_path is a Path object
    input_path = Path(input_path)
    
    # Generate output paths
    output_json_path = os.path.join(output_dir, "inference_results.json")
    output_submission_path = os.path.join(output_dir, "submission.csv")
    output_logits_path = os.path.join(output_dir, "logits.csv")

    if not input_path.is_dir():
        print(f"Error: {input_path} is not a directory. Please provide a directory of images.")
        return []
    
    # Run batch inference
    print(f"Running batch inference on directory: {input_path}")
    results, all_probabilities, class_names = batch_inference(
        model, input_path, class_names, device, val_transform, 
        batch_size=config.inference_batch_size
    )
    
    if not results:
        print("No results generated. Check if the input directory contains valid images.")
        return []
    
    # Get filenames in the same order as probabilities
    filenames = [result['filename'] for result in results]
    
    # Save results in all formats in both locations
    save_inference_results(results, output_json_path)
    save_submission_csv(results, output_submission_path)
    save_logits_csv(filenames, all_probabilities, class_names, output_logits_path)
    
    # If we have ground truth labels (for validation/test sets)
    if 'true_label' in results[0]:
        true_labels = [result['true_label'] for result in results]
        predicted_labels = [result['class_id'] for result in results]
        
        # Analyze false predictions
        analysis = analyze_false_predictions(true_labels, predicted_labels, class_names)
        
        # Save analysis
        analysis_path = os.path.join(output_dir, "false_prediction_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(convert_to_json_serializable(analysis), f, indent=2)
        
        # Print report
        print_false_prediction_report(analysis)
        
        # Plot confusion matrix
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(true_labels, predicted_labels, class_names, cm_path)
    
    print(f"Results also saved to output directory: {output_dir}")
    
    return results

def combine_fold_predictions(fold_predictions, class_names, ensemble_method="mean"):
    """Combine predictions from multiple folds using voting or averaging."""
    if not fold_predictions:
        return []
    
    # Group predictions by filename
    combined_results = {}
    all_filenames = set()
    
    for fold_preds in fold_predictions:
        for pred in fold_preds:
            filename = pred['filename']
            all_filenames.add(filename)
            
            if filename not in combined_results:
                combined_results[filename] = {
                    'filename': filename,
                    'image_path': pred['image_path'],
                    'fold_predictions': [],
                    'fold_probabilities': []
                }
            
            combined_results[filename]['fold_predictions'].append({
                'fold': pred.get('fold', -1),
                'class': pred['class'],
                'class_id': pred['class_id'],
                'confidence': pred['confidence']
            })
    
    # Process each image's multiple predictions
    final_results = []
    for filename, result in combined_results.items():
        if ensemble_method == "vote":
            # Use majority voting to select the final class
            votes = {}
            for pred in result['fold_predictions']:
                class_name = pred['class']
                if class_name not in votes:
                    votes[class_name] = 0
                votes[class_name] += 1
            
            # Find class with most votes
            final_class = max(votes.items(), key=lambda x: x[1])[0]
            
            # Find class_id and average confidence for this class
            final_class_id = next(p['class_id'] for p in result['fold_predictions'] if p['class'] == final_class)
            confidences = [p['confidence'] for p in result['fold_predictions'] if p['class'] == final_class]
            final_confidence = sum(confidences) / len(confidences) if confidences else 0
            
        else:  # Default is "mean"
            # Create a mapping of class_id to class_name
            id_to_name = {p['class_id']: p['class'] for p in result['fold_predictions']}
            
            # Get average confidence per class
            class_scores = {}
            for pred in result['fold_predictions']:
                class_id = pred['class_id']
                if class_id not in class_scores:
                    class_scores[class_id] = []
                class_scores[class_id].append(pred['confidence'])
            
            # Average the scores
            avg_scores = {cid: sum(scores)/len(scores) for cid, scores in class_scores.items()}
            
            # Find class with highest average score
            final_class_id = max(avg_scores.items(), key=lambda x: x[1])[0]
            final_class = id_to_name[final_class_id]
            final_confidence = avg_scores[final_class_id]
        
        final_results.append({
            'filename': filename,
            'image_path': result['image_path'],
            'class': final_class,
            'class_id': final_class_id,
            'confidence': final_confidence,
            'fold_predictions': result['fold_predictions']
        })
    
    return final_results

def analyze_false_predictions(true_labels, predicted_labels, class_names):
    """
    Analyze and report false predictions per class.
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Model predictions
        class_names: List of class names
        
    Returns:
        Dictionary with false prediction statistics
    """
    # Convert numpy arrays to lists if needed
    if hasattr(true_labels, 'tolist'):
        true_labels = true_labels.tolist()
    if hasattr(predicted_labels, 'tolist'):
        predicted_labels = predicted_labels.tolist()
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Calculate false predictions per class
    false_pred_per_class = {}
    
    # For each true class, count misclassifications
    for true_idx, class_name in enumerate(class_names):
        # Total samples of this class
        total = int(np.sum(cm[true_idx, :]))
        # Correct predictions (true positives)
        correct = int(cm[true_idx, true_idx])
        # False predictions (should be this class but predicted as something else)
        false = int(total - correct)
        
        # Store the statistics
        false_pred_per_class[class_name] = {
            'total': int(total),
            'correct': int(correct),
            'false': int(false),
            'accuracy': float(correct / total) if total > 0 else 0.0
        }
        
        # Store which classes this class was confused with
        confused_with = {}
        for pred_idx, pred_class in enumerate(class_names):
            if pred_idx != true_idx and cm[true_idx, pred_idx] > 0:
                confused_with[pred_class] = int(cm[true_idx, pred_idx])
        
        false_pred_per_class[class_name]['confused_with'] = confused_with
    
    # Calculate overall statistics
    total_samples = len(true_labels)
    total_correct = sum(1 for i in range(len(true_labels)) if true_labels[i] == predicted_labels[i])
    overall_accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0
    
    result = {
        'per_class': false_pred_per_class,
        'overall': {
            'total_samples': int(total_samples),
            'correct_predictions': int(total_correct),
            'false_predictions': int(total_samples - total_correct),
            'accuracy': float(overall_accuracy)
        }
    }
    
    return result

def plot_confusion_matrix(true_labels, predicted_labels, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_false_prediction_report(analysis):
    """Print a human-readable report of false prediction analysis."""
    print("\n=== False Prediction Analysis ===")
    print(f"Overall Accuracy: {analysis['overall']['accuracy']:.4f}")
    print(f"Total Samples: {analysis['overall']['total_samples']}")
    print(f"Correct Predictions: {analysis['overall']['correct_predictions']}")
    print(f"False Predictions: {analysis['overall']['false_predictions']}")
    
    print("\nPer-Class Analysis (sorted by accuracy):")
    
    # Sort classes by accuracy (ascending)
    sorted_classes = sorted(analysis['per_class'].items(), 
                           key=lambda x: x[1]['accuracy'])
    
    for class_name, stats in sorted_classes:
        print(f"\n  Class: {class_name}")
        print(f"    Accuracy: {stats['accuracy']:.4f}")
        print(f"    Total samples: {stats['total']}")
        print(f"    False predictions: {stats['false']}")
        
        if stats['confused_with']:
            print("    Confused with:")
            sorted_confusions = sorted(stats['confused_with'].items(), 
                                     key=lambda x: x[1], reverse=True)
            for confused_class, count in sorted_confusions:
                print(f"      - {confused_class}: {count}")

# Add this helper function to ensure all objects in a dictionary are JSON serializable
def convert_to_json_serializable(obj):
    """Recursively convert a nested dictionary/list with numpy types to Python standard types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj

# ### Main Execution ###
def main():
    """Run both training and inference in a single pipeline."""
    # Initialize configuration
    config = Config()
    if(config.debug):
        print("WARNING: THIS IS DEBUG MODE")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Running experiment version: {config.version}")
    
    # Create output directory
    version_dir = os.path.join(config.output_dir, config.version)
    os.makedirs(version_dir, exist_ok=True)    
    
    # Save config to JSON for reproducibility
    with open(os.path.join(version_dir, 'config.json'), 'w') as f:
        config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}
        json.dump(config_dict, f, indent=4)
    
    # Save start time for benchmarking
    start_time = time.time()
    
    try:
        # Training phase - Cross-validation
        print("\n=== Starting Training Phase (Cross-validation) ===")
        avg_val_accuracy, fold_results, cv_histories, analyses = cross_validate(config, device)
        
        # Save combined analysis
        combined_analysis_path = os.path.join(version_dir, 'combined_analysis.json')
        with open(combined_analysis_path, 'w') as f:
            json.dump(convert_to_json_serializable(analyses), f, indent=2)
        
        # Print summary of problematic classes across folds
        print("\n=== Summary of Problematic Classes Across Folds ===")
        class_problem_scores = {}
        
        for fold, analysis in analyses.items():
            for class_name, stats in analysis['per_class'].items():
                if class_name not in class_problem_scores:
                    class_problem_scores[class_name] = {'total_false': 0, 'count': 0}
                
                class_problem_scores[class_name]['total_false'] += stats['false']
                class_problem_scores[class_name]['count'] += 1
        
        # Calculate average false predictions per class
        for class_name, stats in class_problem_scores.items():
            stats['avg_false'] = stats['total_false'] / stats['count']
        
        # Sort by average false predictions (descending)
        sorted_problems = sorted(class_problem_scores.items(), 
                               key=lambda x: x[1]['avg_false'], reverse=True)
        
        print("\nClasses sorted by average false predictions:")
        for class_name, stats in sorted_problems:
            print(f"  {class_name}: {stats['avg_false']:.2f} avg false predictions")
        
        # Report training time
        train_time = time.time() - start_time
        hours, remainder = divmod(train_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        # Inference phase (if enabled)
        if config.run_inference_after_training and config.inference_input_path:
            print("\n=== Starting Inference Phase ===")
            
            # Run inference for each trained fold model
            all_fold_results = []
            
            for fold in config.train_folds:
                print(f"\n--- Running inference with fold {fold+1} model ---")
                fold_dir = os.path.join(version_dir, f'fold_{fold}')
                model_path = os.path.join(fold_dir, 'model_weights.pth')
                
                # Check if model exists
                if not os.path.exists(model_path):
                    print(f"Warning: Model for fold {fold+1} not found at {model_path}, skipping.")
                    continue
                
                # Run inference with this fold's model
                fold_results = run_inference(
                    config, 
                    model_path, 
                    config.inference_input_path, 
                    fold_dir,  # Save in fold directory
                    device
                )
                
                # Add fold information to results
                for result in fold_results:
                    result['fold'] = fold
                
                all_fold_results.append(fold_results)
            
            # Combine predictions from all folds (if enabled and we have multiple folds)
            if config.ensemble_method and len(all_fold_results) > 1:
                print(f"\n--- Creating ensemble using method: {config.ensemble_method} ---")
                dataset = MushroomDataset(config.csv_path, transform=None)
                class_names = dataset.classes
                
                combined_results = combine_fold_predictions(
                    all_fold_results, 
                    class_names,
                    ensemble_method=config.ensemble_method
                )
                
                # Create ensemble directory
                ensemble_dir = os.path.join(version_dir, "ensemble")
                os.makedirs(ensemble_dir, exist_ok=True)
                
                # Save combined results
                combined_json_path = os.path.join(ensemble_dir, "inference_results.json")
                combined_submission_path = os.path.join(ensemble_dir, "submission.csv")
                combined_version_submission_path = os.path.join(version_dir, "submission.csv")
                
                save_inference_results(combined_results, combined_json_path)
                save_submission_csv(combined_results, combined_submission_path)
                save_submission_csv(combined_results, combined_version_submission_path)  # Also at version level
                
                print(f"Ensemble predictions from {len(all_fold_results)} models saved to {ensemble_dir}")
                print(f"Ensemble submission also saved to {combined_version_submission_path}")
            elif not config.ensemble_method and len(config.train_folds) > 1:
                print("Ensemble is disabled. Each fold's predictions are saved separately.")
            
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