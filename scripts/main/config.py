import os
from dataclasses import dataclass

# Set default class names
CLASS_NAMES = ['nm', 'bn', 'dg', 'lc']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

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
    save_per_fold_results: bool = True  # Save results for each fold separately
    use_class_weights: bool = False  # Whether to use class weights in loss function
    weight_multiplier: float = 2.0  # How much extra weight to give to the prioritized class
    prioritized_class: str = "Đùi gà Baby (cắt ngắn)"  # Class to prioritize
    ensemble_methods: list = ["mean", "vote"]  # List of methods to combine predictions
    save_last_model: bool = False  # If True, save the last model instead of the best model
    save_only_at_end: bool = True  # If True, save model only at the end of training to reduce I/O operations


@dataclass
class EnhancedConfig(Config):
    """Enhanced configuration with additional parameters for advanced techniques."""
    model_type: str = "dual_branch"  # Options: dual_branch, densenet, smallresnet, etc.
    use_multi_scale: bool = False  # Whether to use multi-scale training
    use_albumentations: bool = True  # Whether to use Albumentations augmentation library
    use_advanced_spatial_transforms: bool = True  # Whether to use advanced spatial transformations
    aug_strength: str = "high"  # Options: "low", "medium", "high"
    pixel_percent: float = 0.15 
    crop_scale: float = 0.9
    label_smoothing: float = 0.1  # Label smoothing factor (0.0 means no smoothing)
    
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
    use_mixup_class: bool = False  # Whether to add a mixup class to training
    mixup_class_ratio: float = 0.2  # Ratio of mixup samples to original samples
    mixup_class_name: str = "mixup"  # Name of the mixup class
    mixup_strategy: str = "average"  # How to combine images: "average", "overlay", "mosaic"
    
    # Transformer parameters
    transformer_size: str = None  # Set to None to use manual configuration instead of presets
    transformer_d_model: int = 128  # Embedding dimension
    transformer_nhead: int = 8  # Number of attention heads
    transformer_num_layers: int = 6  # Number of transformer layers
    transformer_dim_feedforward: int = 512  # Size of feedforward layer in transformer
    transformer_dropout_rate: float = 0.1
    transformer_type: str = "pixt"  # Options: "pixt", "vt", "patchpixt", etc.
    transformer_patch_size: int = 4  # Patch size for PatchPixT (2, 4, or 8)
    transformer_patch_sizes: list = None  # List of patch sizes for MultiPatchPixT models
    transformer_fusion_type: str = "concat"  # How to fuse features
    transformer_growth_rate: int = 12  # Growth rate for CNN in CNNMultiPatchPixT
    
    # Memory efficiency options for transformers
    transformer_use_gradient_checkpointing: bool = False
    transformer_sequence_reduction_factor: int = 1
    transformer_share_layer_params: bool = False
    transformer_use_amp: bool = True  # Use automatic mixed precision for transformers
    
    # Multi-GPU support
    use_multi_gpu: bool = True  # Whether to use multiple GPUs if available
    gpu_ids: list = None  # Specific GPU IDs to use, None means use all available

    # AstroTransformer specific parameters
    astro_expansion: int = 2  # Expansion factor for AstroTransformer
    astro_layers: list = None  # Number of blocks per stage, defaults to [2, 2, 2]


def initialize_config():
    """Initialize config settings based on data paths and defaults."""
    if Config.csv_path in ["/kaggle/input/oai-cv/train_cv.csv"]:
        global CLASS_NAMES
        CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)', 'linh chi trắng']
        global CLASS_MAP
        CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        Config.num_folds = 6
    
    if Config.csv_path in ['/kaggle/input/oai-cv/train_group_cv.csv']:
        global CLASS_NAMES
        CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)', 'linh chi trắng']
        global CLASS_MAP
        CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        Config.num_folds = 5
    
    if Config.train_folds is None:
        Config.train_folds = list(range(Config.num_folds))
        print(f"No train_folds specified, using all {Config.num_folds} folds")
