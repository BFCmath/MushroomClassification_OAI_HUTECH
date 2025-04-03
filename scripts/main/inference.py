import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import traceback
from PIL import Image
from torchvision import transforms

# Import local modules
from model_utils import load_model_from_checkpoint
from transforms import get_transforms
from utils import analyze_false_predictions, print_false_prediction_report, plot_confusion_matrix, convert_to_json_serializable
from datasets import MushroomDataset
from config import CLASS_MAP


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
            outputs = model(img_tensor)
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
            'class_probabilities': class_probabilities,
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
    print(f"Running inference on {len(image_paths)} images in batches of {batch_size}")
    if not image_paths:
        print(f"No images found in {image_dir}")
        return [], torch.tensor([]), class_names  # Return consistent tuple structure
    
    results = []
    all_probabilities = []
    processed_count = 0
    failed_count = 0
    
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


def save_submission_csv(results, output_file, config=None):
    """Save prediction results to a submission CSV file in the format id,type."""
    if config and getattr(config, 'use_mixup_class', False):
        # For mixup class setup, process the class probabilities without considering mixup class
        # Create DataFrame with filename and class probabilities
        data = []
        for result in results:
            if 'class_probabilities' in result:
                # Extract data with all probabilities
                row = {'filename': result['filename']}
                row.update(result['class_probabilities'])
                data.append(row)
            else:
                # Fallback if no probabilities
                data.append({
                    'filename': result['filename'],
                    'class': result['class']
                })
        
        if data and 'class_probabilities' in results[0]:  # Only if we have probability data
            # Create DataFrame and process
            df = pd.DataFrame(data)
            
            # Get the mixup class name
            mixup_class_name = getattr(config, 'mixup_class_name', 'mixup')
            
            # Get all columns except filename and mixup class
            original_class_columns = [col for col in df.columns 
                                    if col != 'filename' and col != mixup_class_name]
            
            if original_class_columns:
                # Find original class with highest probability (ignore mixup class)
                print(f"Generating submission by ignoring '{mixup_class_name}' class")
                argmax_result = df[original_class_columns].idxmax(axis=1)
                
                # Create submission with max probability class
                submission_df = pd.DataFrame({
                    'id': df['filename'],
                    'type': argmax_result
                })
                
                # Map class names to CLASS_MAP indices
                submission_df['type'] = submission_df['type'].map(CLASS_MAP)
                submission_df.to_csv(output_file, index=False)
                print(f"Submission saved to {output_file} (ignoring mixup class)")
                return
    
    # Standard path for non-mixup models or fallback
    df = pd.DataFrame([{
        'id': result['filename'],
        'type': result['class'] 
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
        
        # If we're using mixup class, add it to the class names if it's not already there
        if getattr(config, 'use_mixup_class', False) and config.mixup_class_name not in class_names:
            class_names = class_names + [config.mixup_class_name]
            print(f"Added mixup class '{config.mixup_class_name}' to class names for inference")
        
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
        
        # Generate output paths:
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
            save_submission_csv(results, output_submission_path, config)  # Pass config parameter
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
        traceback.print_exc()
        return [], None, None


def combine_fold_predictions(fold_predictions, class_names, ensemble_method="mean"):
    """Combine predictions from multiple folds using voting or averaging of full probability distributions."""
    if not fold_predictions:
        return []
    
    # Group predictions by filename
    combined_results = {}
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
        # Initialize to avoid variable scope issues
        class_probabilities = None
        
        try:
            if ensemble_method == "vote":
                # Use majority voting
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
                        final_class_id = matching_preds[0]['class_id']
                        confidences = [p['confidence'] for p in matching_preds]
                        final_confidence = sum(confidences) / len(confidences)
                    else:
                        # Shouldn't happen but handle it anyway
                        print(f"Warning: No matching predictions found for voted class '{final_class}'")
                        final_class_id = 0
                        final_confidence = 0.5
                else:
                    # No votes (should never happen)
                    print(f"Warning: No votes for {filename}, using first prediction")
                    final_class = result['fold_predictions'][0]['class']
                    final_class_id = result['fold_predictions'][0]['class_id']
                    final_confidence = result['fold_predictions'][0]['confidence']
            else:  # Default is "mean" - average probability distributions
                # Check if we have full probability distributions
                if result['fold_class_probabilities'] and len(result['fold_class_probabilities']) > 0:
                    # Initialize an averaged probability distribution
                    avg_probs = {class_name: 0.0 for class_name in class_names}
                    
                    # Sum probabilities for each class across all folds
                    for probs in result['fold_class_probabilities']:
                        for class_name, prob in probs.items():
                            if class_name in avg_probs:
                                avg_probs[class_name] += prob
                    
                    # Average the summed probabilities
                    num_folds = len(result['fold_class_probabilities'])
                    for class_name in avg_probs:
                        avg_probs[class_name] /= num_folds
                    
                    # Find the class with highest average probability
                    if avg_probs:
                        final_class = max(avg_probs.items(), key=lambda x: x[1])[0]
                        final_confidence = avg_probs[final_class]
                        
                        # Find the class_id for the final class
                        try:
                            final_class_id = class_names.index(final_class)
                        except ValueError:
                            print(f"Warning: Class '{final_class}' not found in class_names")
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
                    # Fall back to confidence averaging if no distributions
                    print(f"Warning: No probability distributions for {filename}")
                    
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
                        final_class = id_to_name.get(final_class_id, class_names[final_class_id] 
                                                   if 0 <= final_class_id < len(class_names) else "unknown")
                    else:
                        # No scores (should never happen)
                        print(f"Warning: No valid scores for {filename}")
                        final_class = class_names[0]
                        final_class_id = 0
                        final_confidence = 0.0
                    
                    # Create probability distribution as fallback
                    class_probabilities = {class_name: 0.0 for class_name in class_names}
                    for cid, score in avg_scores.items():
                        class_name = id_to_name.get(cid, "")
                        if class_name in class_probabilities:
                            class_probabilities[class_name] = score
            
            # Create the final result entry
            result_entry = {
                'filename': filename,
                'image_path': result['image_path'],
                'class': final_class,
                'class_id': final_class_id,
                'confidence': final_confidence,
                'fold_predictions': result['fold_predictions']
            }
            
            # Add full probability distribution if available
            if ensemble_method == "mean" and class_probabilities is not None:
                result_entry['class_probabilities'] = class_probabilities
            
            final_results.append(result_entry)
        except Exception as e:
            print(f"Error processing ensemble for {filename}: {str(e)}")
            # Add basic entry so we don't lose this image
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
