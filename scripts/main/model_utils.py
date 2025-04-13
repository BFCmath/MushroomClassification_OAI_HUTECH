import torch
import torch.nn as nn
import torch.optim as optim

# Import model classes
from scripts.cnn.SmallResNet import *
from scripts.multi_branch.DualBranch import *
from scripts.multi_branch.InceptionFSD import *
from scripts.cnn.DilatedGroup import *
from scripts.cnn.DenseNet7x7 import *
from scripts.cnn.AdaptedLowNet import *
from scripts.cnn.SPDResNet import *
from scripts.multi_branch.SPDDualBranch import SPDDualBranchNetwork
from scripts.cnn.MiniXception import MiniXception
from scripts.mixmodel.MixModel1 import MixModel1
from scripts.mixmodel.MixModel2 import MixModel2
from scripts.mixmodel.MixModel3 import MixModel3
from scripts.mixmodel.MixModel4 import MixModel4
from scripts.transformer.PixT import create_pixt_model, TransformerConfig, PixelTransformer, MemoryEfficientPixT
from scripts.transformer.VT import create_vt_model, VisualTransformer
from scripts.transformer.PatchPixT import create_patchpixt_model, PatchPixTConfig, PatchPixT
from scripts.transformer.CNNMultiPatchPixT import create_cnn_multipatch_pixt_model, CNNMultiPatchPixTConfig, CNNMultiPatchPixT
from scripts.transformer.TaylorIR import create_taylorir_model, TaylorConfig, TaylorIRClassifier
from scripts.transformer.AstroTransformer import create_astro_model, AstroConfig, AstroTransformer
from scripts.cnn.RLNet import create_rlnet
from scripts.cnn.RLSPDNet import create_rlspdnet
from scripts.caps.HybridCapsNet import create_hybridcapsnet  # Updated import for HybridCapsNet


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
    elif model_type == 'rlnet':
        print("Creating RL-Net with Multi-Kernel blocks and multi-scale feature extraction...")
        model = create_rlnet(num_classes=num_classes, 
                           input_channels=3,
                           dropout_rate=config.dropout_rate)
    elif model_type == 'rlspdnet':
        print("Creating RLSPDNet with Multi-Kernel blocks and Space-to-Depth downsampling...")
        model = create_rlspdnet(num_classes=num_classes, 
                             input_channels=3,
                             dropout_rate=config.dropout_rate)
    elif model_type == 'hybridcapsnet':
        print("Creating HybridCapsNet with combined attention and cluster routing...")
        # Get capsule network specific parameters with defaults
        C = getattr(config, 'caps_channels', 4)
        K = getattr(config, 'caps_kernels', 10)
        D = getattr(config, 'caps_depth', 32)
        if_bias = getattr(config, 'caps_bias', True)
        reduction_ratio = getattr(config, 'caps_channel_reduction_ratio', 4)
        use_densenet_backbone = getattr(config, 'caps_use_densenet_backbone', True)
        
        # Print configuration details
        backbone_type = "DenseNet" if use_densenet_backbone else "Standard ConvNet"
        print(f"Using {backbone_type} backbone for HybridCapsNet")
        
        model = create_hybridcapsnet(
            num_classes=num_classes,
            input_img_dim=3,  # Assuming RGB images 
            input_img_size=config.image_size,
            C=C,
            K=K,
            D=D,
            if_bias=if_bias,
            dropout_rate=config.dropout_rate,
            reduction_ratio=reduction_ratio,
            use_densenet_backbone=use_densenet_backbone
        )
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
        elif transformer_type == "astro":
            # Create AstroTransformer model with Sample.py-based implementation
            astro_config = AstroConfig(
                expansion=getattr(config, 'astro_expansion', 2),
                layers=getattr(config, 'astro_layers', [2, 2, 2]),
                use_gradient_checkpointing=config.transformer_use_gradient_checkpointing
            )
            
            model = create_astro_model(
                num_classes=num_classes,
                config=astro_config
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
    
    # Combine all parameter groups
    param_groups = feature_params + classifier_params + other_params
    
    # Check for parameters that weren't included in any group
    all_params = set(p for p in model.parameters() if p.requires_grad)
    missed_params = all_params - param_set
    if missed_params:
        print(f"  Found {len(missed_params)} parameters not assigned to any group, adding with base LR")
        other_params.append({'params': list(missed_params), 'lr': config.learning_rate})
    
    # If param_groups is still empty, fall back to using all model parameters
    if len(param_groups) == 0:
        print("  No parameters found in named children, using all model parameters")
        return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
    
    # Create optimizer with parameter groups
    return optim.Adam(param_groups, lr=config.learning_rate, weight_decay=config.l2_reg)


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
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        # Try to load with strict=False which will ignore missing keys
        print("Attempting to load with strict=False")
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    return model
