import os
from dataclasses import dataclass

# Set default class names
CLASS_NAMES = ['nm', 'bn', 'dg', 'lc']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

@dataclass
class Config:
    """Configuration for hyperparameters."""
    debug = False
    version: str = "exp1.19"  # Version for organizing outputs
    data_dir: str = '/kaggle/input/oai-cv/'
    csv_path: str = os.path.join(data_dir, 'train_cv.csv')
    output_dir: str = '/kaggle/working/'
    inference_input_path: str = '/kaggle/input/aio-hutech/test'  # Directory or file for inference
    image_size: int = 32
    batch_size: int = 256  # Reduced from 512 for better generalization
    num_epochs: int = 500 if not debug else 2
    learning_rate: float = 0.01  # Reduced from 0.005
    dropout_rate: float = 0.0  # Increased from 0.2 for better regularization
    l2_reg: float = 0.00001  # Increased from 0.0001
    num_folds: int = 6  # Folds 0 to 5
    train_folds = None
    early_stopping_patience: int = 50  # Increased from 10
    early_stopping_warmup: int = 75
    mixup_alpha: float = 0.5
    mixup_prob: float = 0.0  # Increased from 0.2
    scheduler_factor: float = 0.1  # Increased from 0.1 for faster LR reduction
    scheduler_patience: int = 15  # Increased from 2
    layer_decay_rate: float = 0.95
    pretrained: bool = True  # Changed to True - using pretrained weights helps
    num_workers: int = 4
    pin_memory: bool = True        # Pin memory for faster GPU transfer
    use_amp: bool = False           # Use automatic mixed precision
    grad_clip_val: float = 1.0     # Gradient clipping value
    scheduler_type: str = "plateau" # "plateau" or "cosine"
    seed: int = 42                 # Random seed
    run_inference_after_training: bool = True  # Whether to run inference after training
    inference_batch_size: int = 256  # Batch size for inference
    submission_path: str = None  # Will be auto-set to output_dir/submission.csv if None
    logits_path: str = None  # Will be auto-set to output_dir/logits.csv if None
    save_per_fold_results: bool = True  # Save results for each fold separately
    use_class_weights: bool = False  # Whether to use class weights in loss function
    weight_multiplier: float = 2.0  # How much extra weight to give to the prioritized class
    prioritized_class: str = "Đùi gà Baby (cắt ngắn)"  # Class to prioritize
    ensemble_methods = ["mean", "vote"]  # List of methods to combine predictions
    save_last_model: bool = True  # If True, save the last model instead of the best model
    save_only_at_end: bool = True  # If True, save model only at the end of training to reduce I/O operations

@dataclass 
class TransConfig:
    # Transformer parameters
    transformer_size: str = None  # Set to None to use manual configuration instead of presets
    transformer_d_model: int = 128  # Embedding dimension
    transformer_nhead: int = 8  # Number of attention heads
    transformer_num_layers: int = 6  # Number of transformer layers
    transformer_dim_feedforward: int = 512  # Size of feedforward layer in transformer
    transformer_dropout_rate: float = 0.1
    transformer_type: str = "pixt"  # Options: "pixt", "vt", "patchpixt", etc.
    transformer_patch_size: int = 4  # Patch size for PatchPixT (2, 4, or 8)
    transformer_patch_sizes = None  # List of patch sizes for MultiPatchPixT models
    transformer_fusion_type: str = "concat"  # How to fuse features
    transformer_growth_rate: int = 12  # Growth rate for CNN in CNNMultiPatchPixT
    
    # Memory efficiency options for transformers
    transformer_use_gradient_checkpointing: bool = False
    transformer_sequence_reduction_factor: int = 1
    transformer_share_layer_params: bool = False
    transformer_use_amp: bool = False  # Use automatic mixed precision for transformers
    
    
    # AstroTransformer specific parameters
    astro_expansion: int = 2  # Expansion factor for AstroTransformer
    astro_layers: list = None  # Number of blocks per stage, defaults to [2, 2, 2]
    
@dataclass 
class AttackConfig:
    # Model weights path configurations - updated with better defaults
    model_weights_pattern: str = "model_weights.pth"  # Pattern to look for when finding weights (model_weights.pth or best_model.pth)
    weights_dir: str = None # '/kaggle/input/oai-test2/exp1.20/'  # Default directory containing model weights
    
    # Attack testing parameters
    attack_base_dir: str = '/kaggle/input/oai-attack'  # Base directory containing attack datasets
    attack_datasets = [
            'change_background_test', 
            'colorize_mushroom_test', 
            'darker_shadow_test', 
            'random_blur_test', 
            'rotate_zoom_test', 
            'simulate_light_test'
        ]
    attack_weight_paths = []  # Will be populated automatically if empty

    def get_weight_paths(self):
        """
        Automatically determine model weight paths based on weights_dir and pattern.
        
        Returns:
            List of weight file paths
        """
        import glob
        import os
        
        weight_paths = []
        
        if self.weights_dir and os.path.exists(self.weights_dir):
            # Check for direct fold directories first (pattern: weights_dir/fold_*/model_weights.pth)
            fold_pattern = os.path.join(self.weights_dir, "fold_*")
            fold_dirs = sorted(glob.glob(fold_pattern))
            
            if fold_dirs:
                # Found fold directories, check each for weight files
                for fold_dir in fold_dirs:
                    # Try the specified pattern first
                    model_path = os.path.join(fold_dir, self.model_weights_pattern)
                    if os.path.exists(model_path):
                        weight_paths.append(model_path)
                    # Try alternative pattern as fallback
                    elif self.model_weights_pattern != "best_model.pth" and os.path.exists(os.path.join(fold_dir, 'best_model.pth')):
                        model_path = os.path.join(fold_dir, 'best_model.pth')
                        weight_paths.append(model_path)
                    elif self.model_weights_pattern != "model_weights.pth" and os.path.exists(os.path.join(fold_dir, 'model_weights.pth')):
                        model_path = os.path.join(fold_dir, 'model_weights.pth')
                        weight_paths.append(model_path)
            else:
                # No direct fold directories, check for experiment subdirectories
                # Get all experiment directories (pattern: exp*)
                exp_dirs = [d for d in os.listdir(self.weights_dir) 
                           if os.path.isdir(os.path.join(self.weights_dir, d)) and d.startswith("exp")]
                
                if exp_dirs:
                    # Use specified version or the first one found
                    target_exp = self.version if hasattr(self, 'version') and self.version in exp_dirs else exp_dirs[0]
                    exp_path = os.path.join(self.weights_dir, target_exp)
                    
                    # Look for fold directories in the experiment directory
                    fold_dirs = sorted(glob.glob(os.path.join(exp_path, "fold_*")))
                    
                    for fold_dir in fold_dirs:
                        # Try the specified pattern first
                        model_path = os.path.join(fold_dir, self.model_weights_pattern)
                        if os.path.exists(model_path):
                            weight_paths.append(model_path)
                        # Try alternative pattern as fallback
                        elif self.model_weights_pattern != "best_model.pth" and os.path.exists(os.path.join(fold_dir, 'best_model.pth')):
                            model_path = os.path.join(fold_dir, 'best_model.pth')
                            weight_paths.append(model_path)
                        elif self.model_weights_pattern != "model_weights.pth" and os.path.exists(os.path.join(fold_dir, 'model_weights.pth')):
                            model_path = os.path.join(fold_dir, 'model_weights.pth')
                            weight_paths.append(model_path)
        
        return weight_paths

@dataclass
class EnhancedConfig(Config, TransConfig, AttackConfig):
    """Enhanced configuration with additional parameters for advanced techniques."""
    model_type: str = "spdresnet"  # Options: dual_branch, densenet, smallresnet, etc.
    use_multi_scale: bool = True  # Whether to use multi-scale training
    use_albumentations: bool = False  # Whether to use Albumentations augmentation library
    use_advanced_spatial_transforms: bool = True  # Whether to use advanced spatial transformations
    aug_strength: str = "low"  # Options: "low", "medium", "high"
    pixel_percent: float = 0.00 
    crop_scale: float = 0.9
    label_smoothing: float = 0.0  # Label smoothing factor (0.0 means no smoothing)
    
    # Transform-related parameters
    elastic_deform_p: float = 0.15  # Probability for elastic deformation
    elastic_deform_alpha: float = 2.0  # Alpha parameter for elastic deformation
    elastic_deform_sigma: float = 1.5  # Sigma parameter for elastic deformation
    focus_zoom_strength: float = 0.2  # Strength for central focus zoom
    focus_zoom_p: float = 0.3  # Probability for central focus zoom
    aspect_ratio_p: float = 0.3  # Probability for aspect ratio variation
    grid_shuffle_p: float = 0.2  # Probability for grid shuffle
    polar_transform_p: float = 0.2  # Probability for polar transform
    tps_strength: float = 0.05  # Strength for thin plate spline
    tps_p: float = 0.1  # Probability for thin plate spline
    radial_distortion_strength: float = 0.15  # Strength for radial distortion
    radial_distortion_p: float = 0.3  # Probability for radial distortion
    
    # Mixup class parameters
    use_mixup_class: bool = True  # Whether to add a mixup class to training
    mixup_class_ratio: float = 0.2  # Ratio of mixup samples to original samples
    mixup_class_name: str = "mixup"  # Name of the mixup class
    mixup_strategy: str = "average"  # How to combine images: "average", "overlay", "mosaic"
    
    
    # Multi-GPU support
    use_multi_gpu: bool = True  # Whether to use multiple GPUs if available
    gpu_ids = None  # Specific GPU IDs to use, None means use all available

def initialize_config():
    """Initialize config settings based on data paths and defaults."""
    global CLASS_NAMES, CLASS_MAP  # Declare globals at the start
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
        
    if AttackConfig.weights_dir is None:
        AttackConfig.weights_dir = f"/kaggle/working/{Config.version}"