import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix
import os
import json
# Update main function to use enhanced features but skip distillation
def format_time(seconds):
    """Format seconds into hours, minutes, seconds string."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


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
    try:
        # Convert numpy arrays to lists if needed
        if hasattr(true_labels, 'tolist'):
            true_labels = true_labels.tolist()
        if hasattr(predicted_labels, 'tolist'):
            predicted_labels = predicted_labels.tolist()
        
        # Create confusion matrix - avoiding sklearn import inside function
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Calculate false predictions per class
        false_pred_per_class = {}
        
        # For each true class, count misclassifications
        for true_idx, class_name in enumerate(class_names):
            try:
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
            except IndexError:
                print(f"WARNING: CLASS INDEX OUT OF RANGE FOR {class_name}. SKIPPING.")
                continue
            except Exception as e:
                print(f"ERROR ANALYZING CLASS {class_name}: {str(e)}")
                continue
        
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
    except Exception as e:
        print(f"CRITICAL ERROR IN ANALYZE_FALSE_PREDICTIONS: {str(e)}")
        # Return a minimal valid structure as fallback
        return {
            'per_class': {},
            'overall': {
                'total_samples': len(true_labels) if hasattr(true_labels, '__len__') else 0,
                'correct_predictions': 0,
                'false_predictions': 0,
                'accuracy': 0.0,
                'error': str(e)
            }
        }

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
