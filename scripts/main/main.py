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
import cv2
import torchvision.transforms.functional as TF

# Import LowNet from the same directory
from datasets import MushroomDataset
from utils import *
from transforms import *
from datasets import *
from scripts.cnn.SmallResNet import *
from scripts.multi_branch.DualBranch import *
from scripts.multi_branch.InceptionFSD import *
from scripts.cnn.DilatedGroup import *
from scripts.cnn.DenseNet7x7 import *
from scripts.cnn.AdaptedLowNet import *
from scripts.cnn.SPDResNet import *
from scripts.multi_branch.SPDDualBranch import SPDDualBranchNetwork  # Import the new model
from scripts.cnn.MiniXception import MiniXception  # Import the new models
from scripts.mixmodel.MixModel1 import MixModel1
from scripts.mixmodel.MixModel2 import MixModel2
from scripts.mixmodel.MixModel3 import MixModel3  # Import the new MixModel3
from scripts.mixmodel.MixModel4 import MixModel4  # Import the new MixModel4
from scripts.transformer.PixT import create_pixt_model, TransformerConfig, PixelTransformer, MemoryEfficientPixT
from scripts.transformer.VT import create_vt_model, VisualTransformer  # Import the new model
from scripts.transformer.PatchPixT import create_patchpixt_model, PatchPixTConfig, PatchPixT
from scripts.transformer.CNNMultiPatchPixT import create_cnn_multipatch_pixt_model, CNNMultiPatchPixTConfig, CNNMultiPatchPixT
from scripts.transformer.TaylorIR import create_taylorir_model, TaylorConfig, TaylorIRClassifier  # Add TaylorIR import

# Add Albumentations for advanced augmentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("Albumentations not available. Using torchvision transforms only.")
    ALBUMENTATIONS_AVAILABLE = False


# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)', 'linh chi trắng'] 
CLASS_NAMES = ['nm', 'bn', 'dg', 'lc']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
# ### Config Class ###
@dataclass
class Config:
    """Configuration for hyperparameters."""
    debug = False
    version: str = "exp1.0"  # Version for organizing outputs
    data_dir: str = '/kaggle/input/oai-cv/'
    csv_path: str = os.path.join(data_dir, 'train_cv.csv')
    output_dir: str = '/kaggle/working/'
    inference_input_path: str = '/kaggle/input/aio-hutech/test'  # Directory or file for inference
    image_size: int = 32
    batch_size: int = 256  # Reduced from 512 for better generalization
    num_epochs: int = 500 if not debug else 2
    learning_rate: float = 0.001  # Reduced from 0.005
    dropout_rate: float = 0.2  # Increased from 0.2 for better regularization
    l2_reg: float = 0.00001  # Increased from 0.0001
    num_folds: int = 6  # Folds 0 to 5
    train_folds = None
    early_stopping_patience: int = 15  # Increased from 10
    early_stopping_warmup: int = 25
    mixup_alpha: float = 0.5
    mixup_prob: float = 0.0  # Increased from 0.2
    scheduler_factor: float = 0.5  # Increased from 0.1 for faster LR reduction
    scheduler_patience: int = 2  # Increased from 2
    layer_decay_rate: float = 0.95
    pretrained: bool = True  # Changed to True - using pretrained weights helps
    num_workers: int = 4
    pin_memory: bool = True        # Pin memory for faster GPU transfer
    use_amp: bool = False           # Use automatic mixed precision
    grad_clip_val: float = 10.0     # Gradient clipping value
    scheduler_type: str = "cosine" # "plateau" or "cosine"
    seed: int = 42                 # Random seed
    run_inference_after_training: bool = True  # Whether to run inference after training
    inference_batch_size: int = 256  # Batch size for inference
    submission_path: str = None  # Will be auto-set to output_dir/submission.csv if None
    logits_path: str = None  # Will be auto-set to output_dir/logits.csv if None
    ensemble_method: str = "mean"  # How to combine multiple model predictions: "mean" or "vote"
    save_per_fold_results: bool = True  # Save results for each fold separately
    use_class_weights: bool = False  # Whether to use class weights in loss function
    weight_multiplier: float = 2.0  # How much extra weight to give to the prioritized class
    prioritized_class: str = "Đùi gà Baby (cắt ngắn)"  # Class to prioritize
    ensemble_methods: list = ["mean", "vote"]  # List of methods to combine predictions: "mean", "vote", "weighted"
    # Backward compatibility - will be set to ["mean"] if None

# Update Config class with new parameters
@dataclass
class EnhancedConfig(Config):
    """Enhanced configuration with additional parameters for advanced techniques."""
    model_type: str = "dual_branch"  # Options: dual_branch, densenet, smallresnet, mobilenet, inceptionfsd, dilatedgroupconv, dilatedgroupconv
    use_multi_scale: bool = False  # Whether to use multi-scale training
    use_albumentations: bool = True  # Whether to use Albumentations augmentation library
    aug_strength: str = "high"  # Options: "low", "medium", "high"
    pixel_percent: float = 0.15 
    crop_scale: float = 0.9
    
    # transformer - direct parameters for manual configuration
    transformer_size: str = None  # Set to None to use manual configuration instead of presets
    transformer_d_model: int = 128  # Embedding dimension
    transformer_nhead: int = 8  # Number of attention heads
    transformer_num_layers: int = 6  # Number of transformer layers
    transformer_dim_feedforward: int = 512  # Size of feedforward layer in transformer
    transformer_dropout_rate: float = 0.1
    transformer_type: str = "pixt"  # Options: "pixt", "vt", "patchpixt", "multiscale_patchpixt", "cnn_multipatch_pixt"
    transformer_patch_size: int = 4  # Patch size for PatchPixT (2, 4, or 8)
    transformer_patch_sizes: list = None  # List of patch sizes for MultiPatchPixT models, defaults to [1, 2, 4]
    transformer_fusion_type: str = "concat"  # How to fuse features: "concat", "weighted_sum", "attention"
    transformer_growth_rate: int = 12  # Growth rate for CNN in CNNMultiPatchPixT
    
    # Memory efficiency options for transformers
    transformer_use_gradient_checkpointing: bool = False
    transformer_sequence_reduction_factor: int = 1
    transformer_share_layer_params: bool = False
    transformer_use_amp: bool = True  # Use automatic mixed precision specifically for transformers
    
    # Multi-GPU support
    use_multi_gpu: bool = True  # Whether to use multiple GPUs if available
    gpu_ids: list = None  # Specific GPU IDs to use, None means use all available

if Config.csv_path in ["/kaggle/input/oai-cv/train_cv.csv"]:
    CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)', 'linh chi trắng'] 
    CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    Config.num_folds = 6
if Config.csv_path in ['/kaggle/input/oai-cv/train_group_cv.csv']:
    CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)', 'linh chi trắng'] 
    CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    Config.num_folds = 5
if Config.train_folds is None:
    Config.train_folds = list(range(Config.num_folds))
    print(f"No train_folds specified, using all {Config.num_folds} folds")

# Fix the get_model function which was corrupted
def get_model(num_classes, config, device):
    """
    Use a model suited for the image size with architecture specified in config.
    Wrap with DataParallel if multiple GPUs are to be used.
    """
    model_type = getattr(config, 'model_type', 'densenet')
    
    # Create the base model based on model_type
    if model_type == 'lownet':
        print("Creating LowNet model for low-resolution image feature extraction...")
        model = AdaptedLowNet(num_classes=num_classes, 
                             dropout_rate=config.dropout_rate)
    elif model_type == 'spdresnet':
        print("Creating SPDResNet model with Space-to-Depth downsampling...")
        print("Using SPDResNet-18 architecture for small images")
        model = spdresnet18(num_classes=num_classes, dropout_rate=config.dropout_rate)
    elif model_type == 'inceptionfsd':
        print("Creating InceptionFSD model with multi-scale feature extraction...")
        model = InceptionFSD(num_classes=num_classes, 
                            dropout_rate=config.dropout_rate)
    elif model_type == 'spddualbranch':
        print("Creating SPDDualBranch model with Space-to-Depth downsampling in both branches...")
        model = SPDDualBranchNetwork(num_classes=num_classes,
                                   dropout_rate=config.dropout_rate)
    elif model_type == 'dilatedgroupconv':
        print("Creating DilatedGroupConvNet with dilated convolutions...")
        model = DilatedGroupConvNet(num_classes=num_classes, 
                                dropout_rate=config.dropout_rate)
    elif model_type == 'dual_branch':
        print("Creating Dual-Branch Network with common feature subspace...")
        model = DualBranchNetwork(num_classes=num_classes, 
                                 dropout_rate=config.dropout_rate)
    elif model_type == 'densenet':
        # Use DenseNet explicitly when requested
        print("Creating DenseNet7x7 model with 7x7 kernels...")
        model = DenseNet7x7(growth_rate=16, 
                            block_config=(3, 6, 12, 8),
                            num_classes=num_classes, 
                            dropout_rate=config.dropout_rate)
    elif model_type == 'smallresnet':
        print("Creating SmallResNet model...")
        model = SmallResNet(num_classes=num_classes, 
                           dropout_rate=config.dropout_rate)
    elif model_type == 'minixception':
        print("Creating MiniXception model with SPD downsampling and separable convolutions...")
        model = MiniXception(num_classes=num_classes, 
                           dropout_rate=config.dropout_rate)
    elif model_type == 'mixmodel1':
        print("Creating MixModel1 with SPDConv, Residual Inception blocks, and SE attention...")
        model = MixModel1(num_classes=num_classes, 
                         dropout_rate=config.dropout_rate)
    elif model_type == 'mixmodel2':
        print("Creating MixModel2 with multi-pathways and skip connections...")
        model = MixModel2(num_classes=num_classes, 
                         dropout_rate=config.dropout_rate)
    elif model_type == 'mixmodel3':
        print("Creating MixModel3 that maintains 32x32 resolution throughout the network...")
        model = MixModel3(num_classes=num_classes, 
                         dropout_rate=config.dropout_rate)
    elif model_type == 'mixmodel4':
        print("Creating MixModel4 with multi-branch spatial feature extraction at different scales...")
        model = MixModel4(num_classes=num_classes, 
                         dropout_rate=config.dropout_rate)
    elif model_type == 'trans':
        print("Creating Transformer model for image classification...")
        transformer_type = getattr(config, 'transformer_type', 'pixt')
        
        # Create transformer configuration using direct parameters
        transformer_config = TransformerConfig(
            img_size=config.image_size,
            d_model=config.transformer_d_model,
            nhead=config.transformer_nhead,
            num_layers=config.transformer_num_layers,
            dim_feedforward=getattr(config, 'transformer_dim_feedforward', config.transformer_d_model * 4),
            dropout=config.transformer_dropout_rate,
            use_gradient_checkpointing=config.transformer_use_gradient_checkpointing,
            sequence_reduction_factor=config.transformer_sequence_reduction_factor,
            share_layer_params=config.transformer_share_layer_params,
            use_sequence_downsampling=getattr(config, 'transformer_use_sequence_downsampling', False)
        )
        
        # Create the appropriate transformer model based on type
        if transformer_type == 'vt':
            model = create_vt_model(
                num_classes=num_classes,
                img_size=config.image_size,
                config=transformer_config
            )
        elif transformer_type == "patchpixt":
            # Create PatchPixT with directly specified parameters
            patch_size = getattr(config, 'transformer_patch_size', 4)
            
            # Create PatchPixT config with manual parameters
            patch_config = PatchPixTConfig(
                img_size=config.image_size,
                patch_size=patch_size,
                d_model=config.transformer_d_model,
                nhead=config.transformer_nhead,
                num_layers=config.transformer_num_layers,
                dim_feedforward=getattr(config, 'transformer_dim_feedforward', config.transformer_d_model * 4),
                dropout=config.transformer_dropout_rate,
                use_gradient_checkpointing=config.transformer_use_gradient_checkpointing,
                sequence_reduction_factor=config.transformer_sequence_reduction_factor,
                share_layer_params=config.transformer_share_layer_params,
                use_sequence_downsampling=getattr(config, 'transformer_use_sequence_downsampling', False)
            )
            
            model = create_patchpixt_model(
                num_classes=num_classes,
                img_size=config.image_size,
                patch_size=patch_size,
                config=patch_config
            )
        elif transformer_type == "cnn_multipatch_pixt":
            # Create CNNMultiPatchPixT model with CNN backbone and multiple patch sizes
            patch_sizes = getattr(config, 'transformer_patch_sizes', [1, 2, 4])
            fusion_type = getattr(config, 'transformer_fusion_type', 'concat')
            growth_rate = getattr(config, 'transformer_growth_rate', 12)
            
            # Create CNNMultiPatchPixT config
            cnn_multipatch_config = CNNMultiPatchPixTConfig(
                img_size=config.image_size,
                patch_sizes=patch_sizes,
                d_model=config.transformer_d_model,
                nhead=config.transformer_nhead,
                num_layers=config.transformer_num_layers,
                dim_feedforward=getattr(config, 'transformer_dim_feedforward', config.transformer_d_model * 4),
                dropout=config.transformer_dropout_rate,
                fusion_type=fusion_type,
                growth_rate=growth_rate,
                use_gradient_checkpointing=config.transformer_use_gradient_checkpointing,
                share_layer_params=config.transformer_share_layer_params,
                cnn_dropout=config.dropout_rate
            )
            
            model = create_cnn_multipatch_pixt_model(
                num_classes=num_classes,
                img_size=config.image_size,
                patch_sizes=patch_sizes,
                config=cnn_multipatch_config
            )
        elif transformer_type == "taylorir":
            # Create TaylorIR model with specified parameters
            taylor_config = TaylorConfig(
                img_size=config.image_size,
                embed_dim=config.transformer_d_model,
                num_heads=config.transformer_nhead,
                num_layers=config.transformer_num_layers,
                dim_feedforward=getattr(config, 'transformer_dim_feedforward', config.transformer_d_model * 4),
                dropout=config.transformer_dropout_rate,
                use_gradient_checkpointing=config.transformer_use_gradient_checkpointing
            )
            
            model = create_taylorir_model(
                num_classes=num_classes,
                img_size=config.image_size,
                config=taylor_config
            )
        elif transformer_type == "pixt":  # Default to 'pixt'
            model = create_pixt_model(
                num_classes=num_classes,
                config=transformer_config
            )
        else: 
            print("Unknown transformer type, stop to save your time")
            raise ValueError(f"Unknown transformer type: {transformer_type}")
    else:
        print("Unknown model type, stop to save your time")
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move model to device first
    model = model.to(device)
    
    # Wrap with DataParallel if using multiple GPUs
    if hasattr(config, 'use_multi_gpu') and config.use_multi_gpu:
        # Get available GPU devices
        if hasattr(config, 'gpu_ids') and config.gpu_ids:
            gpu_ids = config.gpu_ids
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        
        if len(gpu_ids) > 1:
            print(f"Using DataParallel with GPUs: {gpu_ids}")
            model = nn.DataParallel(model, device_ids=gpu_ids)
    
    return model

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
    
    # Keep track of parameters we've added to groups
    param_set = set()
    
    # Organize parameters by layer type
    for name, module in model.named_children():
        print(f"Layer group: {name}")
        
        # Skip modules with no parameters
        if sum(1 for _ in module.parameters()) == 0:
            print(f"  {name}: Skipping - no trainable parameters")
            continue
            
        if name == 'features':
            # Apply gradually decreasing LR to feature layers
            for i, layer in enumerate(module):
                # Skip layers with no parameters
                if sum(1 for p in layer.parameters() if p.requires_grad) == 0:
                    continue
                    
                layer_lr = config.learning_rate * (config.layer_decay_rate ** i)
                print(f"  Feature block {i}: lr = {layer_lr:.6f}")
                
                # Only include parameters that require gradients
                params = [p for p in layer.parameters() if p.requires_grad]
                if params:  # Only add if there are actually parameters
                    feature_params.append({'params': params, 'lr': layer_lr})
                    param_set.update(params)  # Add to our tracking set
                    
        elif name == 'classifier':
            # Use base LR for classifier (final layers)
            # Only include parameters that require gradients
            params = [p for p in module.parameters() if p.requires_grad]
            if params:  # Check if there are any trainable parameters
                print(f"  Classifier: lr = {config.learning_rate}")
                classifier_params.append({'params': params, 'lr': config.learning_rate})
                param_set.update(params)  # Add to our tracking set
        else:
            # Other layers - collect their parameters too
            params = [p for p in module.parameters() if p.requires_grad]
            if params:  # Check if there are any trainable parameters
                print(f"  {name}: lr = {config.learning_rate}")
                other_params.append({'params': params, 'lr': config.learning_rate})
                param_set.update(params)  # Add to our tracking set
    
    # Check for parameters that weren't included in any group
    all_params = set(p for p in model.parameters() if p.requires_grad)
    missed_params = all_params - param_set
    
    if missed_params:
        print(f"  Found {len(missed_params)} parameters not assigned to any group, adding with base LR")
        other_params.append({'params': list(missed_params), 'lr': config.learning_rate})
    
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
    
    # Add early stopping warmup default if not in config
    early_stopping_warmup = getattr(config, 'early_stopping_warmup', 5)
    
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
    
    # Check if using a transformer model with AMP
    is_transformer = isinstance(model, (PixelTransformer, MemoryEfficientPixT, VisualTransformer)) or 'PixT' in str(type(model)) or 'VT' in str(type(model))
    use_transformer_amp = is_transformer and hasattr(config, 'transformer_use_amp') and config.transformer_use_amp
    
    if use_transformer_amp and not config.use_amp:
        print("Enabling AMP specifically for transformer model")
        scaler = GradScaler()
    
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
            if config.use_amp or use_transformer_amp:
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
        model.eval()  # Explicitly set model to evaluation mode
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
        
        # Get current learning rate (just once)
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
        print(f"  Learning Rate: {current_lr:.8f}")
        
        # Save best model (by validation accuracy)
        if val_accuracy > best_val_acc:
            print(f"  Validation accuracy improved from {best_val_acc:.4f} to {val_accuracy:.4f}")
            best_val_acc = val_accuracy
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint to version directory instead of output_dir
            checkpoint_path = os.path.join(version_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': best_val_acc,
                'history': history,
            }, checkpoint_path)
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # Early stopping check with warm-up period
        if epoch >= early_stopping_warmup:
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs (after {early_stopping_warmup} warm-up epochs)")
                break
        else:
            print(f"  Warm-up phase: {epoch+1}/{early_stopping_warmup} epochs")
    
    # Load best model before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation accuracy: {best_val_acc:.4f}")
    
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
    
    # Save analysis to version directory instead of output_dir
    analysis_path = os.path.join(version_dir, 'false_prediction_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(convert_to_json_serializable(analysis), f, indent=2)
    
    # Plot confusion matrix to version directory
    cm_path = os.path.join(version_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_true_labels, all_predictions, classes, cm_path)
    
    return model, best_val_acc, history, analysis

# ### Cross-Validation ###
def cross_validate(config, device):
    """Perform cross-validation with enhanced features."""
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Check for multi-GPU setup
    if hasattr(config, 'use_multi_gpu') and config.use_multi_gpu:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Multi-GPU training enabled with {num_gpus} GPUs")
            # Adjust batch size if using multiple GPUs to utilize them effectively
            effective_batch_size = config.batch_size * num_gpus
            print(f"Effective batch size: {effective_batch_size} (per GPU: {config.batch_size})")
        else:
            print("Multi-GPU training requested but only one GPU found.")
    
    # Create version directory
    version_dir = os.path.join(config.output_dir, config.version)
    os.makedirs(version_dir, exist_ok=True)
    
    # Get appropriate transforms based on configuration with consistent parameters
    if getattr(config, 'use_albumentations', False) and ALBUMENTATIONS_AVAILABLE:
        print(f"Using Albumentations transforms with strength: {config.aug_strength}")
        train_transform, val_transform = get_albumentation_transforms(
            aug_strength=getattr(config, 'aug_strength', 'high'),
            image_size=config.image_size,
            multi_scale=getattr(config, 'use_multi_scale', False)
        )
    elif getattr(config, 'use_multi_scale', False):
        print("Using enhanced multi-scale transforms")
        train_transform, val_transform = get_enhanced_transforms(
            multi_scale=True,
            image_size=config.image_size,
            pixel_percent = config.pixel_percent,
            crop_scale = config.crop_scale
        )
    else:
        print("Using standard transforms")
        train_transform, val_transform = get_transforms(
            image_size=config.image_size,
            aug_strength=getattr(config, 'aug_strength', 'standard')
        )
    
    # Initialize dataset once for fold information
    try:
        dataset = MushroomDataset(config.csv_path, transform=None)
        if 'fold' not in dataset.data.columns:
            raise ValueError("CSV file must contain a 'fold' column for cross-validation")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise
        
    val_metrics = []
    fold_results = {}
    all_histories = {}
    all_analyses = {}
    
    # Create a local copy of train_folds to avoid modifying the config object
    train_folds = config.train_folds

    
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
    
    # Create a base dataset that will be reused across folds to save memory and loading time
    base_train_dataset = MushroomDataset(config.csv_path, transform=None)
    base_val_dataset = MushroomDataset(config.csv_path, transform=None)
    
    for fold_idx, fold in enumerate(train_folds):
        print(f"\n===== Fold {fold+1}/{config.num_folds} ({fold_idx+1}/{len(train_folds)}) =====")
        
        try:
            # Create train/val datasets for this fold with appropriate transforms
            train_indices = dataset.data[dataset.data['fold'] != fold].index.tolist()
            val_indices = dataset.data[dataset.data['fold'] == fold].index.tolist()
            
            if len(train_indices) == 0 or len(val_indices) == 0:
                print(f"Warning: Empty train or validation set for fold {fold}. Skipping...")
                continue
            
            # Create fold directory
            fold_dir = os.path.join(version_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # Create subsets for this fold using CustomSubset for more efficient transform application
            train_dataset = CustomSubset(Subset(base_train_dataset, train_indices), train_transform)
            val_dataset = CustomSubset(Subset(base_val_dataset, val_indices), val_transform)
            
            # Create data loaders with optimized settings and error handling
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config.batch_size, 
                shuffle=True, 
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                drop_last=False  # Don't drop last batch even if smaller than batch_size
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config.batch_size, 
                shuffle=False, 
                num_workers=config.num_workers,
                pin_memory=config.pin_memory
            )
            
            print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
            
            # Initialize model with proper error handling
            try:
                model = get_model(len(dataset.class_to_idx), config, device)
            except Exception as e:
                print(f"Error initializing model: {str(e)}")
                print("Skipping fold due to model initialization failure")
                continue
            
            # Loss function with class weights if enabled
            if class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                print(f"Using weighted CrossEntropyLoss with weights: {class_weights.cpu().numpy()}")
            else:
                criterion = nn.CrossEntropyLoss()
                
            optimizer = get_layer_wise_lr_optimizer(model, config)
            
            # Create scheduler with proper error handling
            try:
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
            except Exception as e:
                print(f"Error creating scheduler: {str(e)}")
                print("Using default scheduler (ReduceLROnPlateau)")
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            
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

            # Generate and save visualizations
            plot_training_history(history, os.path.join(fold_dir, 'training_history.png'))

            # Store results
            val_metrics.append(val_accuracy)
            fold_results[fold] = val_accuracy
            all_histories[fold] = history
            all_analyses[fold] = analysis
            
            # Save fold-specific model with error handling
            try:
                torch.save(model.state_dict(), os.path.join(fold_dir, 'model_weights.pth'))
            except Exception as e:
                print(f"Error saving model: {str(e)}")
            
            # Save analysis to fold directory with error handling
            try:
                analysis_path = os.path.join(fold_dir, 'false_prediction_analysis.json')
                with open(analysis_path, 'w') as f:
                    json.dump(convert_to_json_serializable(analysis), f, indent=2)
            except Exception as e:
                print(f"Error saving analysis: {str(e)}")
                
            # Clear some memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error processing fold {fold}: {str(e)}")
            print("Continuing to next fold...")
            continue
    
    # Only compute summary statistics if we have results
    if val_metrics:
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
            'config': {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v 
                      for k, v in config.__dict__.items()},
        }
        
        with open(os.path.join(version_dir, 'cv_results.json'), 'w') as f:
            json.dump(cv_results, f, indent=4)
        
        return avg_val_accuracy, fold_results, all_histories, all_analyses
    else:
        print("No valid results from any fold. Cross-validation failed.")
        return 0.0, {}, {}, {}

# ### Inference Functions ###
def load_model_from_checkpoint(checkpoint_path, num_classes, config, device):
    """Load a model from a checkpoint file with support for DataParallel."""
    # Create base model on specified device
    model = get_model(num_classes, config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats and DataParallel prefixes
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle case where model was saved with DataParallel but is now loaded without it
    if not isinstance(model, nn.DataParallel) and all(k.startswith('module.') for k in state_dict.keys()):
        # Remove 'module.' prefix from state_dict keys
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix (7 characters)
        state_dict = new_state_dict
    
    # Handle case where model was saved without DataParallel but is now loaded with it
    elif isinstance(model, nn.DataParallel) and not all(k.startswith('module.') for k in state_dict.keys()):
        # Add 'module.' prefix to state_dict keys
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict['module.' + k] = v
        state_dict = new_state_dict
    
    # Load the state dict
    model.load_state_dict(state_dict)
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
    try:
        img_tensor = preprocess_image(image_path, transform)
        img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        with torch.no_grad():
            outputs = model(img_tensor)  # Fixed: was using 'inputs' variable that didn't exist
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted_idx.item()]
        confidence = confidence.item()
        
        # Get top-k predictions
        k = min(3, len(class_names))
        topk_values, topk_indices = torch.topk(probabilities, k)
        topk_predictions = [
            (class_names[idx.item()], prob.item()) 
            for idx, prob in zip(topk_indices[0], topk_values[0])
        ]
        
        # Add full probability distribution
        class_probabilities = {class_name: probabilities[0, idx].item() 
                              for idx, class_name in enumerate(class_names)}
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'top_predictions': topk_predictions,
            'class_probabilities': class_probabilities,  # Add full probability distribution
            'success': True
        }
    except Exception as e:
        print(f"Error predicting image {image_path}: {str(e)}")
        return {
            'class': None,
            'confidence': 0.0,
            'top_predictions': [],
            'error': str(e),
            'success': False
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
        return [], torch.tensor([]), class_names  # Return consistent tuple structure
    
    results = []
    all_probabilities = []
    processed_count = 0
    failed_count = 0
    
    print(f"Running inference on {len(image_paths)} images in batches of {batch_size}")
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        valid_tensors = []
        valid_paths = []
        
        # Process each image with error handling
        for img_path in batch_paths:
            try:
                tensor = preprocess_image(str(img_path), transform)
                valid_tensors.append(tensor)
                valid_paths.append(img_path)
            except Exception as e:
                failed_count += 1
                print(f"Error processing image {img_path}: {str(e)}")
                # Continue to next image instead of failing entire batch
        
        if not valid_tensors:
            print(f"No valid images in current batch, skipping")
            continue
        
        # Stack successful tensors and run inference
        try:
            batch_tensor = torch.stack(valid_tensors).to(device)
            
            with torch.no_grad():
                outputs = model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_indices = torch.max(probabilities, 1)
            
            # Store the full probabilities tensor for later processing
            all_probabilities.append(probabilities.cpu())
            
            # Process each result
            for j, (img_path, pred_idx, conf) in enumerate(zip(valid_paths, predicted_indices, confidence)):
                predicted_class = class_names[pred_idx.item()]
                
                # Get top-k predictions for this image
                k = min(3, len(class_names))
                topk_values, topk_indices = torch.topk(probabilities[j], k)
                topk_predictions = [
                    (class_names[idx.item()], prob.item()) 
                    for idx, prob in zip(topk_indices, topk_values)
                ]
                
                # Get image filename without extension for CSV
                filename = Path(img_path).stem
                
                # Add full probability distribution to results
                class_probs = {class_name: probabilities[j, idx].item() 
                              for idx, class_name in enumerate(class_names)}
                
                results.append({
                    'image_path': str(img_path),
                    'filename': filename,
                    'class': predicted_class,
                    'class_id': pred_idx.item(),
                    'confidence': conf.item(),
                    'top_predictions': topk_predictions,
                    'class_probabilities': class_probs  # Store all class probabilities
                })
                processed_count += 1
                
        except Exception as e:
            print(f"Error during batch inference: {str(e)}")
            # Continue to next batch
    
    # Print summary
    print(f"Processed {processed_count} images successfully, {failed_count} images failed")
    
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
    try:
        # Get the class names
        dataset = MushroomDataset(config.csv_path, transform=None)
        class_names = dataset.classes
        
        # Load the model with proper error handling
        try:
            model = load_model_from_checkpoint(model_path, len(class_names), config, device)
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            return [], None, None
        
        # Create validation transform with correct image size from config
        _, val_transform = get_transforms(image_size=config.image_size)
        
        # Ensure input_path is a Path object
        input_path = Path(input_path)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output paths
        output_json_path = os.path.join(output_dir, "inference_results.json")
        output_submission_path = os.path.join(output_dir, "submission.csv")
        output_logits_path = os.path.join(output_dir, "logits.csv")

        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory. Please provide a directory of images.")
            return [], None, None
        
        # Run batch inference
        print(f"Running batch inference on directory: {input_path}")
        results, all_probabilities, class_names = batch_inference(
            model, input_path, class_names, device, val_transform, 
            batch_size=config.inference_batch_size
        )
        
        if not results or len(all_probabilities) == 0:
            print("No valid results generated. Check if the input directory contains valid images.")
            return [], torch.tensor([]), class_names
        
        # Get filenames in the same order as probabilities
        filenames = [result['filename'] for result in results]
        
        # Save results in all formats
        try:
            save_inference_results(results, output_json_path)
            save_submission_csv(results, output_submission_path)
            save_logits_csv(filenames, all_probabilities, class_names, output_logits_path)
            print(f"Results saved to {output_dir}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
        
        # Check if any results have ground truth labels
        has_true_labels = any('true_label' in result for result in results)
        
        if has_true_labels:
            try:
                # Extract true labels and predictions where available
                true_labels = []
                predicted_labels = []
                for result in results:
                    if 'true_label' in result:
                        true_labels.append(result['true_label'])
                        predicted_labels.append(result['class_id'])
                
                # Only perform analysis if we have ground truth labels
                if true_labels:
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
            except Exception as e:
                print(f"Error during analysis: {str(e)}")
        
        return results, all_probabilities, class_names
        
    except Exception as e:
        print(f"Inference pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], None, None

# Fix combine_fold_predictions to properly average full probability distributions
def combine_fold_predictions(fold_predictions, class_names, ensemble_method="mean"):
    """Combine predictions from multiple folds using voting or averaging of full probability distributions."""
    if not fold_predictions:
        return []
    
    # Group predictions by filename
    combined_results = {}
    
    # Process predictions from all folds
    for fold_preds in fold_predictions:
        for pred in fold_preds:
            filename = pred['filename']
            
            if filename not in combined_results:
                combined_results[filename] = {
                    'filename': filename,
                    'image_path': pred['image_path'],
                    'fold_predictions': [],
                    'fold_class_probabilities': []  # Store full probability distributions
                }
            
            combined_results[filename]['fold_predictions'].append({
                'fold': pred.get('fold', -1),
                'class': pred['class'],
                'class_id': pred['class_id'],
                'confidence': pred['confidence']
            })
            
            # Store full class probability distributions if available
            if 'class_probabilities' in pred:
                combined_results[filename]['fold_class_probabilities'].append(
                    pred['class_probabilities']
                )
    
    # Process each image's multiple predictions
    final_results = []
    for filename, result in combined_results.items():
        # Initialize to None to avoid variable scope issues later
        class_probabilities = None
        
        try:
            if ensemble_method == "vote":
                # Use majority voting to select the final class
                votes = {}
                for pred in result['fold_predictions']:
                    class_name = pred['class']
                    votes[class_name] = votes.get(class_name, 0) + 1
                
                # Find class with most votes
                if votes:
                    final_class = max(votes.items(), key=lambda x: x[1])[0]
                    
                    # Find class_id and average confidence for this class
                    matching_preds = [p for p in result['fold_predictions'] if p['class'] == final_class]
                    if matching_preds:
                        final_class_id = matching_preds[0]['class_id']  # Use first matching prediction's class_id
                        confidences = [p['confidence'] for p in matching_preds]
                        final_confidence = sum(confidences) / len(confidences)
                    else:
                        # This should not happen, but handle it just in case
                        print(f"Warning: No matching predictions found for voted class '{final_class}' in {filename}")
                        final_class_id = 0
                        final_confidence = 0.5
                else:
                    # No votes (should never happen)
                    print(f"Warning: No votes for {filename}, using first prediction")
                    final_class = result['fold_predictions'][0]['class']
                    final_class_id = result['fold_predictions'][0]['class_id']
                    final_confidence = result['fold_predictions'][0]['confidence']
                
            else:  # Default is "mean" - properly average full probability distributions
                # Check if we have full probability distributions
                if result['fold_class_probabilities'] and len(result['fold_class_probabilities']) > 0:
                    # Initialize an averaged probability distribution
                    avg_probs = {class_name: 0.0 for class_name in class_names}
                    
                    # Sum probabilities for each class across all folds
                    for probs in result['fold_class_probabilities']:
                        for class_name, prob in probs.items():
                            if class_name in avg_probs:  # Ensure class exists in dictionary
                                avg_probs[class_name] += prob
                    
                    # Average the summed probabilities
                    num_folds = len(result['fold_class_probabilities'])
                    for class_name in avg_probs:
                        avg_probs[class_name] /= num_folds
                    
                    # Find the class with highest average probability
                    if avg_probs:
                        final_class = max(avg_probs.items(), key=lambda x: x[1])[0]
                        final_confidence = avg_probs[final_class]
                        
                        # Find the class_id for the final class (safely)
                        try:
                            final_class_id = class_names.index(final_class)
                        except ValueError:
                            print(f"Warning: Class '{final_class}' not found in class_names. Using first class.")
                            final_class_id = 0
                    else:
                        # Empty probabilities (should never happen)
                        print(f"Warning: Empty probability distribution for {filename}")
                        final_class = class_names[0]
                        final_class_id = 0
                        final_confidence = 0.0
                    
                    # Store the full averaged distribution
                    class_probabilities = avg_probs
                else:
                    # Fall back to the old method if no full distributions are available
                    print(f"Warning: No probability distributions for {filename}. Using confidence averaging.")
                    
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
                    if avg_scores:
                        final_class_id = max(avg_scores.items(), key=lambda x: x[1])[0]
                        final_confidence = avg_scores[final_class_id]
                        final_class = id_to_name.get(final_class_id, class_names[final_class_id] if 0 <= final_class_id < len(class_names) else "unknown")
                    else:
                        # No scores (should never happen)
                        print(f"Warning: No valid scores for {filename}")
                        final_class = class_names[0]
                        final_class_id = 0
                        final_confidence = 0.0
                    
                    # Create an probability distribution as a fallback
                    class_probabilities = {class_name: 0.0 for class_name in class_names}
                    for cid, score in avg_scores.items():
                        class_name = id_to_name.get(cid, "")
                        if class_name in class_probabilities:
                            class_probabilities[class_name] = score
            
            # Create the final result
            result_entry = {
                'filename': filename,
                'image_path': result['image_path'],
                'class': final_class,
                'class_id': final_class_id,
                'confidence': final_confidence,
                'fold_predictions': result['fold_predictions']
            }
            
            # Add full probability distribution if available (mean method)
            if ensemble_method == "mean" and class_probabilities is not None:
                result_entry['class_probabilities'] = class_probabilities
            
            final_results.append(result_entry)
            
        except Exception as e:
            print(f"Error processing ensemble for {filename}: {str(e)}")
            # Add a basic entry so we don't lose this image in results
            if result['fold_predictions']:
                first_pred = result['fold_predictions'][0]
                final_results.append({
                    'filename': filename,
                    'image_path': result['image_path'],
                    'class': first_pred['class'],
                    'class_id': first_pred['class_id'],
                    'confidence': first_pred['confidence'],
                    'fold_predictions': result['fold_predictions'],
                    'error': str(e)
                })
    
    return final_results

def evaluate_model(model, data_loader, criterion, device, num_classes=None):
    """Evaluate model performance on a dataset with comprehensive metrics."""
    # Set model to evaluation mode
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Prepare storage for detailed metrics
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    # Track progress for large datasets
    total_batches = len(data_loader)
    print_interval = max(1, total_batches // 10)
    
    try:
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(data_loader):
                # Progress indication
                if batch_idx % print_interval == 0:
                    print(f"Evaluating batch {batch_idx}/{total_batches}")
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Compute probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and targets for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
        
        # Combine all probabilities
        if all_probabilities:
            all_probabilities = np.vstack(all_probabilities)
        
        # Calculate basic metrics
        avg_loss = running_loss / total if total > 0 else float('inf')
        accuracy = correct / total if total > 0 else 0.0
        
        # Prepare return value as a comprehensive results dictionary
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total,
        }
        
        # If class names are available, add detailed metrics
        if num_classes is not None:
            # Create confusion matrix and get per-class metrics
            analysis = analyze_false_predictions(all_targets, all_predictions, 
                                              list(range(num_classes)))
            results['analysis'] = analysis
            results['per_class_accuracy'] = {
                i: stats['accuracy'] for i, stats in analysis['per_class'].items()
            }
        
        # Add predictions and probabilities for external analysis
        results['predictions'] = np.array(all_predictions)
        results['targets'] = np.array(all_targets)
        results['probabilities'] = all_probabilities
        
        return results
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'error': str(e)
        }

def main():
    """Run both training and inference in a single pipeline."""
    try:
        # Initialize enhanced configuration
        config = EnhancedConfig()
        if config.debug:
            print("WARNING: THIS IS DEBUG MODE")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Running experiment version: {config.version}")
        
        # Create output directory
        version_dir = os.path.join(config.output_dir, config.version)
        os.makedirs(version_dir, exist_ok=True)    
        
        # Save config to JSON for reproducibility with better error handling
        try:
            with open(os.path.join(version_dir, 'config.json'), 'w') as f:
                # Handle all potentially non-serializable types, not just Path
                config_dict = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None), list, dict)) else v 
                              for k, v in config.__dict__.items()}
                json.dump(config_dict, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save configuration to JSON: {str(e)}")
        
        # Save start time for benchmarking
        start_time = time.time()
        
        # Configure transforms at the global scope rather than locally
        if getattr(config, 'use_albumentations', False) and ALBUMENTATIONS_AVAILABLE:
            print(f"Using Albumentations augmentation with {config.aug_strength} strength")
            # Use global variables to properly override transformations
            train_transform, val_transform = get_albumentation_transforms(
                aug_strength=getattr(config, 'aug_strength', 'high'), 
                image_size=config.image_size,
                multi_scale=getattr(config, 'use_multi_scale', False)
            )
        elif getattr(config, 'use_multi_scale', False):
            print("Using multi-scale training transforms")
            train_transform, val_transform = get_enhanced_transforms(
                multi_scale=True,
                image_size=config.image_size,
                pixel_percent = config.pixel_percent,
                crop_scale = config.crop_scale
            )
        else:
            print("Using standard transforms")
            train_transform, val_transform = get_transforms(
                image_size=config.image_size,
                aug_strength="standard"
            )
        
        # === Training Phase ===
        print("\n=== Starting Training Phase (Cross-validation) with Enhanced Features ===")
        
        # Run cross-validation with the enhanced model architecture
        avg_val_accuracy, fold_results, cv_histories, analyses = cross_validate(config, device)

        # Save combined analysis if we have results
        if analyses:
            try:
                combined_analysis_path = os.path.join(version_dir, 'combined_analysis.json')
                with open(combined_analysis_path, 'w') as f:
                    json.dump(convert_to_json_serializable(analyses), f, indent=2)
                    
                # Print summary of problematic classes across folds
                print("\n=== Summary of Problematic Classes Across Folds ===")
                class_problem_scores = {}
                
                for fold, analysis in analyses.items():
                    if not analysis or 'per_class' not in analysis:
                        continue
                        
                    for class_name, stats in analysis['per_class'].items():
                        if class_name not in class_problem_scores:
                            class_problem_scores[class_name] = {'total_false': 0, 'count': 0}
                        
                        class_problem_scores[class_name]['total_false'] += stats.get('false', 0)
                        class_problem_scores[class_name]['count'] += 1
                
                # Calculate average false predictions per class
                if class_problem_scores:
                    for class_name, stats in class_problem_scores.items():
                        stats['avg_false'] = stats['total_false'] / max(stats['count'], 1)  # Avoid division by zero
                    
                    # Sort by average false predictions (descending)
                    sorted_problems = sorted(class_problem_scores.items(), 
                                          key=lambda x: x[1]['avg_false'], reverse=True)
                    
                    print("\nClasses sorted by average false predictions:")
                    for class_name, stats in sorted_problems:
                        print(f"  {class_name}: {stats['avg_false']:.2f} avg false predictions")
            except Exception as e:
                print(f"Error generating analysis summary: {str(e)}")
        else:
            print("No analysis results available from cross-validation")
        
        # Report training time
        train_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(train_time)}")
        
        # === Inference Phase ===
        if config.run_inference_after_training:
            if not config.inference_input_path or not os.path.exists(config.inference_input_path):
                print(f"Warning: Inference path not found or not specified: {config.inference_input_path}")
            else:
                print("\n=== Starting Inference Phase ===")
                
                # Load class names from dataset once for efficiency
                try:
                    dataset = MushroomDataset(config.csv_path, transform=None)
                    class_names = dataset.classes
                except Exception as e:
                    print(f"Error loading dataset for class names: {str(e)}")
                    print("Using default class names for inference")
                    class_names = CLASS_NAMES  # Fall back to global constants
                
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
                    results, probs, _ = run_inference(
                        config, 
                        model_path, 
                        config.inference_input_path, 
                        fold_dir,  # Save in fold directory
                        device
                    )
                    
                    # Add fold information to results if we got valid results
                    if results:
                        for result in results:
                            result['fold'] = fold
                        all_fold_results.append(results)
                    else:
                        print(f"No valid results from fold {fold+1}, skipping in ensemble")
                
                # Combine predictions from all folds using multiple ensemble methods
                if all_fold_results and len(all_fold_results) > 1:  # Make sure we have at least some results from multiple folds
                    # Get ensemble methods (ensure it's a list)
                    ensemble_methods = config.ensemble_methods if isinstance(config.ensemble_methods, list) else [config.ensemble_methods]
                    
                    # Create results for each ensemble method
                    for method in ensemble_methods:
                        if not method:  # Skip empty/None methods
                            continue
                            
                        print(f"\n--- Creating ensemble using method: {method} ---")
                        
                        combined_results = combine_fold_predictions(
                            all_fold_results, 
                            class_names,
                            ensemble_method=method
                        )
                        
                        if combined_results:  # Make sure ensemble produced valid results
                            # Create ensemble directory with method name
                            ensemble_dir = os.path.join(version_dir, f"ensemble_{method}")
                            os.makedirs(ensemble_dir, exist_ok=True)
                            
                            # Save combined results
                            try:
                                combined_json_path = os.path.join(ensemble_dir, "inference_results.json")
                                combined_submission_path = os.path.join(ensemble_dir, "submission.csv")
                                
                                # Save ensemble results
                                save_inference_results(combined_results, combined_json_path)
                                save_submission_csv(combined_results, combined_submission_path)
                                
                                # If this is the primary method (first in list), also save at version level
                                if method == ensemble_methods[0]:
                                    combined_version_submission_path = os.path.join(version_dir, "submission.csv")
                                    save_submission_csv(combined_results, combined_version_submission_path)
                                    print(f"Primary ensemble (method={method}) also saved to {combined_version_submission_path}")
                                
                                print(f"Ensemble '{method}' predictions from {len(all_fold_results)} models saved to {ensemble_dir}")
                            except Exception as e:
                                print(f"Error saving ensemble ({method}) results: {str(e)}")
                        else:
                            print(f"Error: Ensemble method '{method}' produced no valid results")
                    
                    # Generate comparison report of ensemble methods if multiple methods used
                    if len(ensemble_methods) > 1:
                        try:
                            # Create comparison directory
                            comparison_dir = os.path.join(version_dir, "ensemble_comparison")
                            os.makedirs(comparison_dir, exist_ok=True)
                            
                            # Generate and save comparison report
                            comparison_path = os.path.join(comparison_dir, "methods_comparison.json")
                            with open(comparison_path, 'w') as f:
                                json.dump({
                                    "methods": ensemble_methods,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "config": {k: str(v) if not isinstance(v, (int, float, bool, str, type(None), list, dict)) else v 
                                             for k, v in config.__dict__.items()}
                                }, f, indent=4)
                            
                            print(f"\nEnsemble methods comparison saved to {comparison_path}")
                        except Exception as e:
                            print(f"Error generating ensemble comparison: {str(e)}")
                            
                elif not config.ensemble_methods or (len(config.ensemble_methods) == 1 and not config.ensemble_methods[0]):
                    print("Ensemble is disabled. Each fold's predictions are saved separately.")
                else:
                    print("No valid inference results from any fold, skipping ensemble")
                
        # Report total execution time
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {format_time(total_time)}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 1
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0  # Return success code
