import os
import random
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from datetime import datetime

# Import local modules
from model_utils import get_model, get_layer_wise_lr_optimizer
from transforms import get_transforms, get_enhanced_transforms, get_albumentation_transforms, ALBUMENTATIONS_AVAILABLE, CustomSubset
from utils import analyze_false_predictions, print_false_prediction_report, plot_confusion_matrix, plot_training_history, convert_to_json_serializable
from datasets import MushroomDataset, MixupDataset
from scripts.transformer.PixT import PixelTransformer, MemoryEfficientPixT
from scripts.transformer.VT import VisualTransformer
from scripts.main.poly_loss import create_poly_loss


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, output_dir):
    """Train with mixup, early stopping, scheduler, and optimizations."""
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # Add early stopping warmup default if not in config
    early_stopping_warmup = getattr(config, 'early_stopping_warmup', 5)
    
    # Check saving behavior configuration
    save_last_model = getattr(config, 'save_last_model', False)
    save_only_at_end = getattr(config, 'save_only_at_end', False)
    
    if save_only_at_end:
        print("Note: Configured to save model only at the end of training to reduce I/O operations")
    elif save_last_model:
        print("Note: Configured to save the last model instead of the best validation model")
    
    # History tracking for plotting
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        # Track the best model for final return value
        if val_accuracy > best_val_acc:
            print(f"  Validation accuracy improved from {best_val_acc:.4f} to {val_accuracy:.4f}")
            best_val_acc = val_accuracy
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save best model checkpoint only if we're not saving only at the end
            if not save_only_at_end and not save_last_model:
                checkpoint_path = os.path.join(output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_accuracy': best_val_acc,
                    'history': history,
                }, checkpoint_path)
                print(f"  Saved best model checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # If save_last_model is True, save the current model at each epoch (unless save_only_at_end is True)
        if save_last_model and not save_only_at_end:
            checkpoint_path = os.path.join(output_dir, 'model_weights.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'history': history,
            }, checkpoint_path)
            print(f"  Saved last epoch model to {checkpoint_path}")
        
        # Early stopping check with warm-up period
        if epoch >= early_stopping_warmup:
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs (after {early_stopping_warmup} warm-up epochs)")
                
                # Save model when early stopping is triggered, if save_only_at_end is True
                if save_only_at_end:
                    if save_last_model:
                        # Save the current model if save_last_model is True
                        checkpoint_path = os.path.join(output_dir, 'model_weights.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_accuracy': val_accuracy,
                            'history': history,
                        }, checkpoint_path)
                        print(f"Early stopping - saved last model to {checkpoint_path}")
                    else:
                        # Save the best model if save_last_model is False
                        checkpoint_path = os.path.join(output_dir, 'best_model.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': best_model_state,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': best_val_loss,
                            'val_accuracy': best_val_acc,
                            'history': history,
                        }, checkpoint_path)
                        print(f"Early stopping - saved best model to {checkpoint_path}")
                
                break
        else:
            print(f"  Warm-up phase: {epoch+1}/{early_stopping_warmup} epochs")
    
    # Save final model state if we reached the end of training
    # and haven't saved it yet due to save_only_at_end=True
    if save_only_at_end and epoch == config.num_epochs - 1:
        if save_last_model:
            # Save the last model if save_last_model is True
            checkpoint_path = os.path.join(output_dir, 'model_weights.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'history': history,
            }, checkpoint_path)
            print(f"End of training - saved last model to {checkpoint_path}")
        else:
            # Save the best model if save_last_model is False
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': best_val_acc,
                'history': history,
            }, checkpoint_path)
            print(f"End of training - saved best model to {checkpoint_path}")
    
    # Always load the best model for validation and analysis, even if we saved the last model
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
    if hasattr(val_loader.dataset, 'dataset'):  # For Subset
        classes = val_loader.dataset.dataset.classes
    else:
        classes = val_loader.dataset.classes
    
    # Analyze false predictions
    analysis = analyze_false_predictions(all_true_labels, all_predictions, classes)
    
    # Print report
    print_false_prediction_report(analysis)
    
    # Save analysis to directory passed as parameter 
    analysis_path = os.path.join(output_dir, 'false_prediction_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(convert_to_json_serializable(analysis), f, indent=2)
    
    # Plot confusion matrix to directory passed as parameter
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_true_labels, all_predictions, classes, cm_path)
    
    return model, best_val_acc, history, analysis


def cross_validate(config, device):
    """Perform cross-validation with enhanced features."""
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Check for multi-GPU setup
    if hasattr(config, 'use_multi_gpu') and config.use_multi_gpu:
        num_gpus = torch.cuda.device_count()
        if (num_gpus > 1):
            print(f"Multi-GPU training enabled with {num_gpus} GPUs")
            # Adjust batch size if using multiple GPUs to utilize them effectively
            effective_batch_size = config.batch_size * num_gpus
            print(f"Effective batch size: {effective_batch_size} (per GPU: {config.batch_size})")
        else:
            print("Multi-GPU training requested but only one GPU found.")
    
    # Create version directory
    version_dir = os.path.join(config.output_dir, config.version)
    os.makedirs(version_dir, exist_ok=True)
    
    # Create mushroom transform parameters dictionary from config
    mushroom_params = {
        'radial_strength': getattr(config, 'radial_distortion_strength', 0.15),
        'radial_p': getattr(config, 'radial_distortion_p', 0.3),
        'elastic_alpha': getattr(config, 'elastic_deform_alpha', 2.0),
        'elastic_sigma': getattr(config, 'elastic_deform_sigma', 1.5),
        'elastic_p': getattr(config, 'elastic_deform_p', 0.15),
        'focus_zoom_strength': getattr(config, 'focus_zoom_strength', 0.2),
        'focus_zoom_p': getattr(config, 'focus_zoom_p', 0.3),
        'aspect_ratio_p': getattr(config, 'aspect_ratio_p', 0.3),
        'grid_shuffle_p': getattr(config, 'grid_shuffle_p', 0.2),
        'polar_p': getattr(config, 'polar_transform_p', 0.2),
        'tps_strength': getattr(config, 'tps_strength', 0.05),
        'tps_p': getattr(config, 'tps_p', 0.1)
    }
    
    # Get appropriate transforms based on configuration with consistent parameters
    if getattr(config, 'use_albumentations', False) and ALBUMENTATIONS_AVAILABLE:
        print(f"Using Albumentations transforms with strength: {config.aug_strength}")
        train_transform, val_transform = get_albumentation_transforms(
            aug_strength=getattr(config, 'aug_strength', 'high'),
            image_size=config.image_size,
            multi_scale=getattr(config, 'use_multi_scale', False),
            pixel_percent=getattr(config, 'pixel_percent', 0.05),
            crop_scale=getattr(config, 'crop_scale', 0.9)
        )
    elif getattr(config, 'use_multi_scale', False):
        print("Using enhanced multi-scale transforms")
        train_transform, val_transform = get_enhanced_transforms(
            multi_scale=True,
            image_size=config.image_size,
            pixel_percent=getattr(config, 'pixel_percent', 0.05),
            crop_scale=getattr(config, 'crop_scale', 0.9),
            advanced_spatial_transforms=getattr(config, 'use_advanced_spatial_transforms', True),
            mushroom_transform_params=mushroom_params
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
    
    for fold_idx, fold in enumerate(config.train_folds):
        print(f"\n===== Fold {fold+1}/{config.num_folds} ({fold_idx+1}/{len(config.train_folds)}) =====")
        
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
            train_subset = CustomSubset(Subset(base_train_dataset, train_indices), train_transform)
            val_dataset = CustomSubset(Subset(base_val_dataset, val_indices), val_transform)
            
            # Apply MixupDataset wrapper to training data if enabled
            if getattr(config, 'use_mixup_class', False):
                original_num_classes = len(dataset.classes)
                print(f"Adding mixup class to training data (original classes: {original_num_classes})")
                train_dataset = MixupDataset(
                    train_subset,
                    mixup_ratio=config.mixup_class_ratio,
                    mixup_class_name=config.mixup_class_name,
                    strategy=config.mixup_strategy
                )
                print(f"Training dataset now has {len(train_dataset.classes)} classes: {train_dataset.classes}")
                print(f"Added {train_dataset.num_mixup} mixup samples to training data")
                # Store original num_classes for model creation
                original_num_classes = train_dataset.get_original_num_classes()
            else:
                train_dataset = train_subset
                original_num_classes = len(dataset.classes)
            
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
                if getattr(config, 'use_mixup_class', False):
                    # For training, create a model with extra class for mixup
                    model = get_model(len(train_dataset.classes), config, device)
                else:
                    # Normal case - create model with the original number of classes
                    model = get_model(original_num_classes, config, device)
            except Exception as e:
                print(f"Error initializing model: {str(e)}")
                print("Skipping fold due to model initialization failure")
                continue
            
            # Get label smoothing value from config
            label_smoothing = getattr(config, 'label_smoothing', 0.0)
            
            # Determine which loss function to use
            loss_type = getattr(config, 'loss_type', 'ce')
            
            if loss_type == "poly":
                print(f"Using PolyLoss with epsilon={config.poly_loss_epsilon}")
                
                # Set up PolyLoss with class weights if needed
                if class_weights is not None:
                    if getattr(config, 'use_mixup_class', False):
                        # Add a weight of 1.0 for the mixup class
                        extended_weights = torch.cat([class_weights, torch.tensor([1.0], device=device)])
                        criterion = create_poly_loss(ce_weight=extended_weights, epsilon=config.poly_loss_epsilon)
                        print(f"Using weighted PolyLoss with extended weights: {extended_weights.cpu().numpy()}")
                    else:
                        criterion = create_poly_loss(ce_weight=class_weights, epsilon=config.poly_loss_epsilon)
                        print(f"Using weighted PolyLoss with weights: {class_weights.cpu().numpy()}")
                else:
                    criterion = create_poly_loss(epsilon=config.poly_loss_epsilon)
            else:
                # Default to CrossEntropyLoss with label smoothing
                if class_weights is not None:
                    if getattr(config, 'use_mixup_class', False):
                        # Add a weight of 1.0 for the mixup class
                        extended_weights = torch.cat([class_weights, torch.tensor([1.0], device=device)])
                        criterion = nn.CrossEntropyLoss(weight=extended_weights, label_smoothing=label_smoothing)
                        print(f"Using weighted CrossEntropyLoss with extended weights: {extended_weights.cpu().numpy()} and label smoothing: {label_smoothing}")
                    else:
                        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
                        print(f"Using weighted CrossEntropyLoss with weights: {class_weights.cpu().numpy()} and label smoothing: {label_smoothing}")
                else:
                    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                    if label_smoothing > 0:
                        print(f"Using CrossEntropyLoss with label smoothing: {label_smoothing}")
                    else:
                        print("Using standard CrossEntropyLoss without label smoothing")
            
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
            
            # Train model - updated to also return analysis, passing fold_dir as output directory
            model, val_accuracy, history, analysis = train_model(
                model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                scheduler, 
                config, 
                device,
                fold_dir  # Pass fold_dir as the output directory
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


def evaluate_model(model, data_loader, criterion, device, num_classes=None):
    """Evaluate model performance on a dataset with comprehensive metrics."""
    # Set model to evaluation model
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
                _, predicted = torch.max(outputs, 1)
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
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
