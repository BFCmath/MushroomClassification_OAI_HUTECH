import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths and configuration
TRAIN_DIR = 'train'
OUTPUT_CSV = 'split_cv/train_cv.csv'
CV_FOLDS = 6
RANDOM_SEED = 42
PREFIX = '/kaggle/input/aio-hutech/train'
# Define class names
CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)','linh chi trắng']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def create_cv_splits():
    """
    Create cross-validation splits and save to CSV.
    Uses stratified k-fold to ensure class distribution is preserved across folds.
    """
    print(f"Creating {CV_FOLDS}-fold cross-validation splits...")
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    
    # Gather training data
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        prefix_dir = os.path.join(PREFIX, class_name)
        class_idx = CLASS_MAP[class_name]
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue
            
        print(f"Processing class {class_name} ({class_idx})...")
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(prefix_dir, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(img_path)
                labels.append(class_idx)
    
    # Convert to numpy arrays for easier handling
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Initialize fold column with -1 (not assigned)
    fold_assignments = -1 * np.ones(len(labels), dtype=int)
    
    # Assign fold IDs to each sample
    for fold_idx, (_, val_idx) in enumerate(skf.split(image_paths, labels)):
        fold_assignments[val_idx] = fold_idx
    
    # Create DataFrame with all information
    cv_df = pd.DataFrame({
        'image_path': image_paths,
        'class_id': labels, 
        'class_name': [CLASS_NAMES[label] for label in labels],
        'fold': fold_assignments
    })
    
    # Verify that all samples have been assigned a fold
    assert (cv_df['fold'] >= 0).all(), "Error: Some samples were not assigned to any fold"
    
    # Save to CSV
    cv_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Cross-validation splits saved to {OUTPUT_CSV}")
    
    return cv_df

def analyze_cv_splits(cv_df):
    """
    Analyze the class distribution in each fold.
    """
    print("\nAnalyzing cross-validation splits...")
    
    # Count samples per class per fold
    fold_class_counts = {}
    for fold in range(CV_FOLDS):
        fold_data = cv_df[cv_df['fold'] == fold]
        class_counts = fold_data['class_id'].value_counts().sort_index()
        fold_class_counts[fold] = class_counts
    
    # Convert to DataFrame for easier analysis
    distribution_df = pd.DataFrame(fold_class_counts).T
    distribution_df.columns = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    distribution_df.index.name = 'Fold'
    
    # Add total count column
    distribution_df['Total'] = distribution_df.sum(axis=1)
    
    print("\nSample count per class in each fold:")
    print(distribution_df)
    
    # Calculate class proportions in each fold
    prop_df = distribution_df.iloc[:, :-1].div(distribution_df['Total'], axis=0) * 100
    
    # Plot class distribution across folds
    plt.figure(figsize=(14, 8))
    
    # Bar plot showing counts by class for each fold
    plt.subplot(1, 2, 1)
    distribution_df.iloc[:, :-1].plot(kind='bar', ax=plt.gca())
    plt.title('Class Distribution Across Folds (Count)')
    plt.xlabel('Fold')
    plt.ylabel('Number of Samples')
    plt.legend(title='Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Heatmap showing class proportions
    plt.subplot(1, 2, 2)
    sns.heatmap(prop_df, annot=True, fmt=".1f", cmap="YlGnBu", 
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Class Distribution Across Folds (Percentage)')
    
    plt.tight_layout()
    plt.savefig('split_cv/cv_distribution.png')
    plt.show()
    
    return distribution_df

def main():
    """Main function to create and analyze CV splits."""
    # Create CV splits
    cv_df = create_cv_splits()
    
    # Analyze the splits
    distribution_df = analyze_cv_splits(cv_df)
    
if __name__ == "__main__":
    main()
