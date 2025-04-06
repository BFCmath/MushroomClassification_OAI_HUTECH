import os
import json
import time
import torch
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from datetime import datetime
import traceback
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Import from parent directory modules
from config import EnhancedConfig, initialize_config, Config, CLASS_MAP
from model_utils import load_model_from_checkpoint
from transforms import get_transforms
from inference import batch_inference, combine_fold_predictions, save_inference_results, save_submission_csv, run_inference
from utils import format_time, convert_to_json_serializable, plot_confusion_matrix, analyze_false_predictions
from datasets import MushroomDataset


def run_attack_test(config: EnhancedConfig):
    """
    Run attack tests on models using the specified configuration.
    
    Args:
        config: The EnhancedConfig object containing test settings
    """
    start_time = time.time()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and getattr(config, 'use_cuda', True) else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory for attack results
    attack_output_dir = os.path.join(config.output_dir, config.version, "attack_results")
    os.makedirs(attack_output_dir, exist_ok=True)
    
    # Save attack config - silently
    try:
        with open(os.path.join(attack_output_dir, 'attack_config.json'), 'w') as f:
            config_dict = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None), list, dict)) else v 
                          for k, v in config.__dict__.items()}
            json.dump(config_dict, f, indent=4)
    except Exception:
        pass
    
    # Process each attack dataset
    attack_datasets = getattr(config, 'attack_datasets', [])
    if not attack_datasets:
        attack_datasets = ['change_background_test', 'colorize_mushroom_test', 
                           'darker_shadow_test', 'random_blur_test', 
                           'rotate_zoom_test', 'simulate_light_test']
        
    # Store overall attack results for final summary
    overall_attack_results = {}
        
    for attack_name in attack_datasets:
        print(f"\n=== Running attack: {attack_name} ===")
        
        # Create full path to attack dataset
        attack_path = os.path.join(getattr(config, 'attack_base_dir', '/kaggle/input/oai-attack'), attack_name)
        
        # Check if attack dataset exists
        if not os.path.exists(attack_path):
            print(f"Warning: Attack dataset not found at {attack_path}, skipping.")
            continue
            
        dataset_output_dir = os.path.join(attack_output_dir, attack_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Set up ground truth for this attack
        # Each attack subfolder has 50 images for each class in the order [1,2,3,0]
        ground_truth = [1]*50 + [2]*50 + [3]*50 + [0]*50
        
        # Find model weights from all folds - quietly
        all_fold_results = []
        weight_paths = getattr(config, 'attack_weight_paths', [])
        
        # If weights_dir is provided, look for model files in that directory
        weights_dir = getattr(config, 'weights_dir', None)
        if not weight_paths and weights_dir and os.path.exists(weights_dir):
            # Check if weights_dir points directly to an experiment folder
            fold_pattern = os.path.join(weights_dir, "fold_*")
            fold_dirs = sorted(glob.glob(fold_pattern))
            
            if fold_dirs:
                for fold_dir in fold_dirs:
                    model_path = os.path.join(fold_dir, 'model_weights.pth')
                    if os.path.exists(model_path):
                        weight_paths.append(model_path)
                    elif os.path.exists(os.path.join(fold_dir, 'best_model.pth')):
                        model_path = os.path.join(fold_dir, 'best_model.pth')
                        weight_paths.append(model_path)
            else:
                # Maybe weights_dir is parent folder containing multiple experiment versions?
                exp_dirs = [d for d in os.listdir(weights_dir) 
                           if os.path.isdir(os.path.join(weights_dir, d)) and d.startswith("exp")]
                
                if exp_dirs:
                    # If experiment version is specified, prioritize that one
                    if hasattr(config, 'version') and config.version in exp_dirs:
                        target_exp = config.version
                    else:
                        target_exp = exp_dirs[0]
                    
                    exp_path = os.path.join(weights_dir, target_exp)
                    fold_dirs = sorted(glob.glob(os.path.join(exp_path, "fold_*")))
                    
                    for fold_dir in fold_dirs:
                        model_path = os.path.join(fold_dir, 'model_weights.pth')
                        if os.path.exists(model_path):
                            weight_paths.append(model_path)
                        elif os.path.exists(os.path.join(fold_dir, 'best_model.pth')):
                            model_path = os.path.join(fold_dir, 'best_model.pth')
                            weight_paths.append(model_path)
        
        # If still no specific paths, search in version directory as last resort
        if not weight_paths and (not weights_dir or not os.path.exists(weights_dir)):
            version_dir = os.path.join(config.output_dir, config.version)
            
            for fold in config.train_folds:
                fold_dir = os.path.join(version_dir, f'fold_{fold}')
                if os.path.exists(fold_dir):
                    model_path = os.path.join(fold_dir, 'model_weights.pth')
                    if os.path.exists(model_path):
                        weight_paths.append(model_path)
                    else:
                        model_path = os.path.join(fold_dir, 'best_model.pth')
                        if os.path.exists(model_path):
                            weight_paths.append(model_path)
        
        if not weight_paths:
            print("No model weights found. Please check your config or folder structure.")
            continue
            
        print(f"Found {len(weight_paths)} model weights")
        
        attack_fold_accuracies = []
        
        # Run inference with each model weight
        for weight_path in weight_paths:
            fold_name = os.path.basename(os.path.dirname(weight_path))
            fold_output_dir = os.path.join(dataset_output_dir, fold_name)
            os.makedirs(fold_output_dir, exist_ok=True)
            
            print(f"Running {attack_name} with fold: {fold_name}")
            
            # Use run_inference function from inference.py
            # This will handle model loading, batch inference, and saving results
            results, probs, class_names = run_inference(
                config, 
                weight_path, 
                attack_path,
                fold_output_dir,
                device
            )
            
            if results:
                # Add fold information to results if valid
                for result in results:
                    result['model'] = fold_name
                all_fold_results.append(results)
                
                # Compare predictions with ground truth - using submission file for consistent handling
                submission_path = os.path.join(fold_output_dir, "submission.csv")
                if os.path.exists(submission_path):
                    try:
                        # Read the filtered predictions from the submission CSV (which properly filters mixup class)
                        df = pd.read_csv(submission_path)
                        if len(df) == len(ground_truth):
                            # The submission CSV has class IDs in the 'type' column
                            predictions = df['type'].tolist()
                            
                            # Calculate accuracy
                            correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
                            accuracy = correct / len(ground_truth)
                            print(f"Accuracy on {attack_name}: {accuracy:.4f} ({correct}/{len(ground_truth)})")
                            
                            # Store accuracy for this fold
                            attack_fold_accuracies.append(accuracy)
                            
                            # Filter class_names to only include those present in ground truth/predictions
                            # This ensures we don't include mixup class in the analysis
                            unique_classes = sorted(set(ground_truth + predictions))
                            analysis_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
                            
                            # Generate analysis and confusion matrix using filtered class names
                            analysis = analyze_false_predictions(ground_truth, predictions, analysis_class_names)
                            analysis_path = os.path.join(fold_output_dir, 'attack_analysis.json')
                            with open(analysis_path, 'w') as f:
                                json.dump(convert_to_json_serializable(analysis), f, indent=2)
                            
                            cm_path = os.path.join(fold_output_dir, 'confusion_matrix.png')
                            plot_confusion_matrix(ground_truth, predictions, analysis_class_names, cm_path)
                        else:
                            print(f"Warning: Number of predictions ({len(df)}) doesn't match ground truth ({len(ground_truth)})")
                    except Exception as e:
                        print(f"Error processing submission file: {str(e)}")
                else:
                    print(f"Warning: Submission file not found at {submission_path}")
            else:
                print(f"No valid results for fold {fold_name}")

        # Calculate and store average accuracy for this attack
        if attack_fold_accuracies:
            avg_accuracy = np.mean(attack_fold_accuracies)
            std_accuracy = np.std(attack_fold_accuracies)
            overall_attack_results[attack_name] = {
                'avg_accuracy': float(avg_accuracy),
                'std_accuracy': float(std_accuracy),
                'fold_accuracies': [float(acc) for acc in attack_fold_accuracies]
            }
            print(f"Average accuracy on {attack_name}: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Combine results from all models using ensemble methods
        if len(all_fold_results) > 1:
            ensemble_methods = getattr(config, 'ensemble_methods', ["mean", "vote"])
            if not isinstance(ensemble_methods, list):
                ensemble_methods = [ensemble_methods]
                
            for ensemble_method in ensemble_methods:
                print(f"Creating ensemble ({ensemble_method}) for {attack_name}... ", end="")
                ensemble_dir = os.path.join(dataset_output_dir, f"ensemble_{ensemble_method}")
                os.makedirs(ensemble_dir, exist_ok=True)
                
                # Use combine_fold_predictions from inference.py
                combined_results = combine_fold_predictions(
                    all_fold_results,
                    class_names,
                    ensemble_method=ensemble_method
                )
                
                if combined_results:
                    # Save results using functions from inference.py
                    json_path = os.path.join(ensemble_dir, "inference_results.json")
                    save_inference_results(combined_results, json_path)
                    
                    submission_path = os.path.join(ensemble_dir, "submission.csv")
                    save_submission_csv(combined_results, submission_path, config)
                    
                    # Calculate accuracy using the submission file for consistency
                    try:
                        df = pd.read_csv(submission_path)
                        if len(df) == len(ground_truth):
                            # Clean the predictions data - convert to integers and handle any NaN values
                            ensemble_predictions = []
                            for val in df['type'].values:
                                try:
                                    # Convert to int and validate
                                    cleaned_val = int(val)
                                    if cleaned_val not in range(len(analysis_class_names)):
                                        print(f"Warning: Invalid prediction value {cleaned_val}, using 0 instead")
                                        cleaned_val = 0
                                    ensemble_predictions.append(cleaned_val)
                                except (ValueError, TypeError):
                                    # Handle NaN or invalid values
                                    print(f"Warning: Invalid prediction value '{val}', using 0 instead")
                                    ensemble_predictions.append(0)
                            
                            # Make sure we have the right number of predictions
                            if len(ensemble_predictions) != len(ground_truth):
                                print(f"Warning: Number of cleaned predictions ({len(ensemble_predictions)}) doesn't match ground truth ({len(ground_truth)})")
                                # Pad or truncate if necessary
                                if len(ensemble_predictions) < len(ground_truth):
                                    ensemble_predictions.extend([0] * (len(ground_truth) - len(ensemble_predictions)))
                                else:
                                    ensemble_predictions = ensemble_predictions[:len(ground_truth)]
                            
                            # Now proceed with valid data
                            ensemble_correct = sum(1 for p, gt in zip(ensemble_predictions, ground_truth) if p == gt)
                            ensemble_accuracy = ensemble_correct / len(ground_truth)
                            print(f"Accuracy: {ensemble_accuracy:.4f} ({ensemble_correct}/{len(ground_truth)})")
                            
                            # Add ensemble accuracy to the attack results
                            if attack_name in overall_attack_results:
                                overall_attack_results[attack_name][f'ensemble_{ensemble_method}_accuracy'] = float(ensemble_accuracy)
                            
                            # Filter class_names for analysis
                            unique_classes = sorted(set(ground_truth + ensemble_predictions))
                            analysis_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
                            
                            # Generate analysis
                            ensemble_analysis = analyze_false_predictions(ground_truth, ensemble_predictions, analysis_class_names)
                            ensemble_analysis_path = os.path.join(ensemble_dir, 'attack_analysis.json')
                            with open(ensemble_analysis_path, 'w') as f:
                                json.dump(convert_to_json_serializable(ensemble_analysis), f, indent=2)
                            
                            # Plot confusion matrix
                            ensemble_cm_path = os.path.join(ensemble_dir, 'confusion_matrix.png')
                            plot_confusion_matrix(ground_truth, ensemble_predictions, analysis_class_names, ensemble_cm_path)
                        else:
                            print("Error: Results count mismatch")
                    except Exception as e:
                        print(f"Error processing ensemble submission: {str(e)}")
                else:
                    print("No valid results")

    # Save overall attack summary
    if overall_attack_results:
        summary_path = os.path.join(attack_output_dir, 'attack_summary.json')
        try:
            with open(summary_path, 'w') as f:
                json.dump(convert_to_json_serializable(overall_attack_results), f, indent=2)
                
            # Create a readable text summary
            text_summary_path = os.path.join(attack_output_dir, 'attack_summary.txt')
            with open(text_summary_path, 'w') as f:
                f.write("=== Attack Test Summary ===\n\n")
                f.write(f"Model: {config.model_type}\n")
                f.write(f"Version: {config.version}\n\n")
                
                # Sort attacks by average accuracy (descending)
                sorted_attacks = sorted(
                    overall_attack_results.items(),
                    key=lambda x: x[1]['avg_accuracy'],
                    reverse=True
                )
                
                f.write("Performance by attack type (sorted by accuracy):\n")
                for attack_name, stats in sorted_attacks:
                    f.write(f"\n{attack_name}:\n")
                    f.write(f"  Average Accuracy: {stats['avg_accuracy']:.4f} ± {stats['std_accuracy']:.4f}\n")
                    
                    # Write ensemble accuracies if available
                    for key, value in stats.items():
                        if key.startswith('ensemble_') and key.endswith('_accuracy'):
                            ensemble_method = key.replace('ensemble_', '').replace('_accuracy', '')
                            f.write(f"  Ensemble ({ensemble_method}) Accuracy: {value:.4f}\n")
            
            print(f"\nOverall attack summary saved to {summary_path}")
        except Exception as e:
            print(f"Error saving attack summary: {str(e)}")
    
    # Report execution time
    total_time = time.time() - start_time
    print(f"\nAttack tests completed in {format_time(total_time)}")


def analyze_model_agreement(model_results: List[List[Dict]], output_dir: str, class_names: List[str]):
    """Analyze agreement between different models on the attack dataset."""
    # Group predictions by filename
    predictions_by_file = {}
    
    for fold_idx, fold_preds in enumerate(model_results):
        fold_name = f"fold_{fold_idx}"
        if fold_preds and 'model' in fold_preds[0]:
            fold_name = fold_preds[0]['model']
            
        for pred in fold_preds:
            filename = pred['filename']
            if filename not in predictions_by_file:
                predictions_by_file[filename] = []
                
            predictions_by_file[filename].append({
                'model': fold_name,
                'class': pred['class'],
                'class_id': pred['class_id'],
                'confidence': pred['confidence']
            })
    
    # Analyze agreement statistics
    total_files = len(predictions_by_file)
    full_agreement = 0
    partial_agreement = 0
    no_agreement = 0
    
    # Track statistics by class
    class_agreement = {cls: {'count': 0, 'full_agreement': 0, 'partial_agreement': 0} 
                      for cls in class_names}
    
    # Detailed agreement analysis
    agreement_details = []
    
    for filename, preds in predictions_by_file.items():
        # Count predictions per class for this file
        class_counts = {}
        for pred in preds:
            cls = pred['class']
            if cls not in class_counts:
                class_counts[cls] = 0
            class_counts[cls] += 1
        
        # Get the most common prediction and its count
        if class_counts:
            most_common_class = max(class_counts.items(), key=lambda x: x[1])
            most_common_class_name = most_common_class[0]
            most_common_count = most_common_class[1]
            total_preds = len(preds)
            
            # Update class statistics
            if most_common_class_name in class_agreement:
                class_agreement[most_common_class_name]['count'] += 1
            
            # Determine agreement type
            if len(class_counts) == 1:  # All models agree
                agreement_type = "full"
                full_agreement += 1
                if most_common_class_name in class_agreement:
                    class_agreement[most_common_class_name]['full_agreement'] += 1
            elif most_common_count > total_preds // 2:  # Majority agreement
                agreement_type = "partial"
                partial_agreement += 1
                if most_common_class_name in class_agreement:
                    class_agreement[most_common_class_name]['partial_agreement'] += 1
            else:  # No clear agreement
                agreement_type = "none"
                no_agreement += 1
            
            # Add to detailed results
            agreement_details.append({
                'filename': filename,
                'agreement_type': agreement_type,
                'predictions': preds,
                'class_counts': class_counts,
                'most_common_class': most_common_class_name,
                'most_common_count': most_common_count,
                'total_predictions': total_preds
            })
    
    # Create agreement summary
    agreement_summary = {
        'total_files': total_files,
        'full_agreement': full_agreement,
        'partial_agreement': partial_agreement,
        'no_agreement': no_agreement,
        'full_agreement_percent': (full_agreement / total_files * 100) if total_files > 0 else 0,
        'partial_agreement_percent': (partial_agreement / total_files * 100) if total_files > 0 else 0,
        'no_agreement_percent': (no_agreement / total_files * 100) if total_files > 0 else 0,
        'class_agreement': class_agreement,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results
    try:
        # Save summary
        summary_path = os.path.join(output_dir, 'agreement_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(convert_to_json_serializable(agreement_summary), f, indent=2)
            
        # Save details (but limit size if very large)
        if len(agreement_details) <= 10000:  # Only save details if not too many files
            details_path = os.path.join(output_dir, 'agreement_details.json')
            with open(details_path, 'w') as f:
                json.dump(convert_to_json_serializable(agreement_details), f, indent=2)
        
        # Create readable text summary
        text_summary_path = os.path.join(output_dir, 'agreement_summary.txt')
        with open(text_summary_path, 'w') as f:
            f.write(f"=== Model Agreement Analysis ===\n\n")
            f.write(f"Total files analyzed: {total_files}\n")
            f.write(f"Full agreement: {full_agreement} files ({agreement_summary['full_agreement_percent']:.1f}%)\n")
            f.write(f"Partial agreement: {partial_agreement} files ({agreement_summary['partial_agreement_percent']:.1f}%)\n")
            f.write(f"No clear agreement: {no_agreement} files ({agreement_summary['no_agreement_percent']:.1f}%)\n\n")
            
            f.write(f"=== Agreement by Class ===\n")
            for cls, stats in class_agreement.items():
                if stats['count'] > 0:
                    full_percent = stats['full_agreement'] / stats['count'] * 100 if stats['count'] > 0 else 0
                    partial_percent = stats['partial_agreement'] / stats['count'] * 100 if stats['count'] > 0 else 0
                    f.write(f"Class '{cls}':\n")
                    f.write(f"  Total: {stats['count']} images\n")
                    f.write(f"  Full agreement: {stats['full_agreement']} ({full_percent:.1f}%)\n")
                    f.write(f"  Partial agreement: {stats['partial_agreement']} ({partial_percent:.1f}%)\n\n")
            
        print(f"Agreement analysis saved to {output_dir}")
    except Exception as e:
        print(f"Error saving agreement analysis: {str(e)}")


def analyze_submission_files(config, attack_path=None, detailed=True):
    """
    Analyze submission files from all folds for a specific attack or all attacks.
    
    Args:
        config: Configuration object
        attack_path: Specific attack folder to analyze, or None to analyze all
        detailed: Whether to show detailed per-class metrics
    """
    # Setup ground truth for attack datasets
    ground_truth = [1]*50 + [2]*50 + [3]*50 + [0]*50
    class_names = list(CLASS_MAP.keys())
    
    # Get list of all attack directories if not specified
    attack_output_dir = os.path.join(config.output_dir, config.version, "attack_results")
    if attack_path:
        attacks = [attack_path]
    else:
        attacks = [d for d in os.listdir(attack_output_dir) 
                  if os.path.isdir(os.path.join(attack_output_dir, d))]
    
    all_results = {}
    
    print("\n=== SUBMISSION FILE ANALYSIS ===")
    
    for attack in attacks:
        attack_dir = os.path.join(attack_output_dir, attack)
        if not os.path.isdir(attack_dir):
            print(f"Attack directory not found: {attack_dir}")
            continue
            
        print(f"\nAttack: {attack}")
        fold_accuracies = []
        fold_per_class = {}
        
        # Check each fold's submission
        for i in range(config.num_folds):
            try:
                file_path = os.path.join(attack_dir, f"fold_{i}", "submission.csv")
                if not os.path.exists(file_path):
                    continue
                    
                df = pd.read_csv(file_path)
                if len(df) != len(ground_truth):
                    print(f"  Warning: Fold {i} has {len(df)} results, expected {len(ground_truth)}")
                    continue
                
                # Calculate overall accuracy
                predictions = df['type'].tolist()
                correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
                accuracy = correct / len(ground_truth)
                fold_accuracies.append(accuracy)
                
                # Calculate per-class metrics
                for class_id in range(4):  # Assuming 4 classes
                    class_name = class_names[class_id]
                    
                    # Find indices for this class in ground truth
                    indices = [i for i, gt in enumerate(ground_truth) if gt == class_id]
                    if not indices:
                        continue
                        
                    # Get predictions for this class
                    class_preds = [predictions[i] for i in indices]
                    class_correct = sum(1 for p in class_preds if p == class_id)
                    class_accuracy = class_correct / len(indices)
                    
                    # Store per-class accuracy
                    if class_id not in fold_per_class:
                        fold_per_class[class_id] = []
                    fold_per_class[class_id].append(class_accuracy)
                
                print(f"  Fold {i}: {accuracy:.4f} ({correct}/{len(ground_truth)})")
            except Exception as e:
                print(f"  Error processing fold {i}: {str(e)}")
        
        # Calculate ensemble accuracies
        ensemble_accuracies = {}
        for method in ['mean', 'vote']:
            try:
                file_path = os.path.join(attack_dir, f"ensemble_{method}", "submission.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    if len(df) == len(ground_truth):
                        predictions = df['type'].tolist()
                        correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
                        accuracy = correct / len(ground_truth)
                        ensemble_accuracies[method] = accuracy
                        print(f"  Ensemble ({method}): {accuracy:.4f} ({correct}/{len(ground_truth)})")
            except Exception as e:
                print(f"  Error processing ensemble {method}: {str(e)}")
        
        # Compute statistics
        if fold_accuracies:
            avg_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            print(f"  Average accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
            
            # Store results for this attack
            all_results[attack] = {
                'fold_accuracies': fold_accuracies,
                'avg_accuracy': float(avg_acc),
                'std_accuracy': float(std_acc),
                'ensemble_accuracies': ensemble_accuracies,
                'per_class_accuracies': {str(k): v for k, v in fold_per_class.items()}
            }
            
            # Show per-class metrics if requested
            if detailed and fold_per_class:
                print("  Per-class metrics:")
                for class_id, accs in fold_per_class.items():
                    if accs:
                        avg_class_acc = np.mean(accs)
                        std_class_acc = np.std(accs)
                        class_name = class_names[class_id]
                        print(f"    {class_name}: {avg_class_acc:.4f} ± {std_class_acc:.4f}")
    
    # Generate comparative visualization if multiple attacks were analyzed
    if len(all_results) > 1:
        _visualize_attack_comparison(all_results, attack_output_dir)
    
    return all_results

def _visualize_attack_comparison(results, output_dir):
    """Create comparative visualizations for attack results."""
    # Extract data for plotting
    attacks = list(results.keys())
    accuracies = [results[a]['avg_accuracy'] for a in attacks]
    std_devs = [results[a]['std_accuracy'] for a in attacks]
    
    # Sort by accuracy (descending)
    sorted_data = sorted(zip(attacks, accuracies, std_devs), key=lambda x: x[1], reverse=True)
    attacks = [x[0] for x in sorted_data]
    accuracies = [x[1] for x in sorted_data]
    std_devs = [x[2] for x in sorted_data]
    
    # Create bar chart with error bars
    plt.figure(figsize=(12, 6))
    plt.bar(attacks, accuracies, yerr=std_devs, capsize=10, 
            color=sns.color_palette("muted", len(attacks)))
    plt.ylim(0, 1.0)  # Accuracy range from 0 to 1
    plt.axhline(y=0.25, color='r', linestyle='--', alpha=0.5)  # Random guess line
    
    plt.title('Attack Accuracy Comparison', fontsize=15)
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attack_comparison.png'))
    
    # Create per-class comparison if available
    has_class_data = all('per_class_accuracies' in results[a] for a in attacks)
    if has_class_data:
        classes = list(CLASS_MAP.keys())
        plt.figure(figsize=(14, 8))
        
        # Position of bars
        bar_width = 0.15
        positions = np.arange(len(attacks))
        
        # Plot bars for each class
        for i, class_id in enumerate(range(4)):
            class_accs = []
            for attack in attacks:
                per_class = results[attack].get('per_class_accuracies', {})
                class_accs.append(
                    np.mean(per_class.get(str(class_id), [0])) if str(class_id) in per_class else 0
                )
            
            plt.bar(positions + i*bar_width, class_accs, width=bar_width, 
                   label=classes[class_id])
        
        plt.title('Per-Class Accuracy Across Attacks', fontsize=15)
        plt.xlabel('Attack Type', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(positions + bar_width*1.5, attacks, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attack_perclass_comparison.png'))
    
    plt.close('all')

def attack():
    """Main function to run attack tests."""
    # Initialize config - NO ARGPARSE, only hardcoded configuration
    config = EnhancedConfig()
    initialize_config()  # Set up additional config parameters
    
    # Add attack-specific parameters, but preserve weights_dir if it's already set
    config.attack_base_dir = getattr(config, 'attack_base_dir', '/kaggle/input/oai-attack')
    config.attack_datasets = getattr(config, 'attack_datasets', [
        'change_background_test', 
        'colorize_mushroom_test', 
        'darker_shadow_test', 
        'random_blur_test', 
        'rotate_zoom_test', 
        'simulate_light_test'
    ])
    config.attack_weight_paths = getattr(config, 'attack_weight_paths', [])
    
    # Keep existing weights_dir if it's set
    if not hasattr(config, 'weights_dir') or config.weights_dir is None:
        config.weights_dir = '/kaggle/input/oai-test2/exp1.19'
    
    # Print minimal configuration
    print("\n=== Attack Test Configuration ===")
    print(f"Model type: {config.model_type}")
    print(f"Model weights directory: {config.weights_dir}")
    
    # Run attack tests
    try:
        run_attack_test(config)
        
        # Add analysis of submission files
        print("\nPerforming submission file analysis...")
        analyze_submission_files(config)
    except Exception as e:
        print(f"CRITICAL ERROR: ATTACK TEST FAILED: {str(e)}")
        traceback.print_exc()
