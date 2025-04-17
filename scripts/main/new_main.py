import os
import time
import torch
import json
from datetime import datetime
from pathlib import Path

# Import all necessary modules
from config import EnhancedConfig, CLASS_NAMES, initialize_config
from training import cross_validate, set_seed
from inference import run_inference, combine_fold_predictions
from transforms import get_transforms, get_enhanced_transforms, get_albumentation_transforms
from utils import format_time, convert_to_json_serializable
from datasets import MushroomDataset


def main():
    """Run both training and inference in a single pipeline."""
    try:
        # Initialize enhanced configuration
        config = EnhancedConfig()
        initialize_config()  # Update config based on dataset paths
        
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
        
        # Save config to JSON for reproducibility
        try:
            with open(os.path.join(version_dir, 'config.json'), 'w') as f:
                # Handle all potentially non-serializable types
                config_dict = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None), list, dict)) else v 
                              for k, v in config.__dict__.items()}
                json.dump(config_dict, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save configuration to JSON: {str(e)}")
        
        # Save start time for benchmarking
        start_time = time.time()
        
        # Create mushroom transform parameters dictionary for consistent use
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
        
        # Configure transforms at the global scope
        if getattr(config, 'use_albumentations', False):
            print(f"Using Albumentations augmentation with {config.aug_strength} strength")
            train_transform, val_transform = get_albumentation_transforms(
                aug_strength=getattr(config, 'aug_strength', 'high'), 
                image_size=config.image_size, 
                multi_scale=getattr(config, 'use_multi_scale', False),
                pixel_percent=getattr(config, 'pixel_percent', 0.05),
                crop_scale=getattr(config, 'crop_scale', 0.9)
            )
        elif getattr(config, 'use_multi_scale', False):
            print("Using multi-scale training transforms")
            if(getattr(config, 'use_advanced_spatial_transforms', True)):
                print("Using advanced spatial transform with these below config:")
                print(mushroom_params)
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
                
                if class_problem_scores:
                    for class_name, stats in class_problem_scores.items():
                        stats['avg_false'] = stats['total_false'] / max(stats['count'], 1)
                    
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
                
                # Print TTA configuration
                if getattr(config, 'use_tta', False):
                    print(f"Test Time Augmentation (TTA) enabled with {config.tta_transforms} transforms and {config.tta_merge_mode} merge mode")
                else:
                    print("Test Time Augmentation (TTA) disabled")
                
                # Load class names from dataset
                try:
                    dataset = MushroomDataset(config.csv_path, transform=None)
                    class_names = dataset.classes
                except Exception as e:
                    print(f"Error loading dataset for class names: {str(e)}")
                    print("Using default class names for inference")
                    class_names = CLASS_NAMES
                
                # Run inference for each trained fold model
                all_fold_results = []
                
                for fold in config.train_folds:
                    print(f"\n--- Running inference with fold {fold+1} model ---")
                    fold_dir = os.path.join(version_dir, f'fold_{fold}')
                    model_path = os.path.join(fold_dir, 'model_weights.pth')
                    
                    # Check if model exists
                    if not os.path.exists(model_path):
                        print(f"Warning: Model not found at {model_path}, skipping.")
                        continue
                    
                    # Run inference with this fold's model
                    results, probs, _ = run_inference(
                        config, 
                        model_path, 
                        config.inference_input_path, 
                        fold_dir,
                        device
                    )
                    
                    # Add fold information to results if valid
                    if results:
                        for result in results:
                            result['fold'] = fold
                        all_fold_results.append(results)
                    else:
                        print(f"No valid results from fold {fold+1}, skipping in ensemble")
                
                # Combine predictions from all folds using multiple ensemble methods
                if all_fold_results and len(all_fold_results) > 1:
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
                        
                        if combined_results:
                            # Create ensemble directory with method name
                            ensemble_dir = os.path.join(version_dir, f"ensemble_{method}")
                            os.makedirs(ensemble_dir, exist_ok=True)
                            
                            # Process and save ensemble results
                            try:
                                from inference import save_inference_results, save_submission_csv
                                
                                combined_json_path = os.path.join(ensemble_dir, "inference_results.json")
                                combined_submission_path = os.path.join(ensemble_dir, "submission.csv")
                                
                                save_inference_results(combined_results, combined_json_path)
                                save_submission_csv(combined_results, combined_submission_path, config)
                                
                                # If this is the primary method, also save at version level
                                if method == ensemble_methods[0]:
                                    version_submission_path = os.path.join(version_dir, "submission.csv")
                                    save_submission_csv(combined_results, version_submission_path, config)
                                    print(f"Primary ensemble also saved to {version_submission_path}")
                                
                                print(f"Ensemble '{method}' predictions saved to {ensemble_dir}")
                            except Exception as e:
                                print(f"Error saving ensemble results: {str(e)}")
                        else:
                            print(f"Error: Ensemble method '{method}' produced no valid results")
                    
                    # Generate comparison report if multiple methods used
                    if len(ensemble_methods) > 1:
                        try:
                            # Create comparison directory
                            comparison_dir = os.path.join(version_dir, "ensemble_comparison")
                            os.makedirs(comparison_dir, exist_ok=True)
                            
                            # Save comparison report
                            comparison_path = os.path.join(comparison_dir, "methods_comparison.json")
                            with open(comparison_path, 'w') as f:
                                json.dump({
                                    "methods": ensemble_methods,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "config": {k: str(v) if not isinstance(v, (int, float, bool, str, type(None), list, dict)) else v 
                                             for k, v in config.__dict__.items()},
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


if __name__ == "__main__":
    # Only run main when executed directly (not imported)
    exit_code = main()
