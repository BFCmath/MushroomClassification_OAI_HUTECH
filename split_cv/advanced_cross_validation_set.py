import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
from pathlib import Path

# Define paths and configuration
TRAIN_DIR = 'train'
OUTPUT_CSV = 'split_cv/train_group_cv.csv'
CV_FOLDS = 5
# 15 21 31 40 42 49
RANDOM_SEED = 15
PREFIX = '/kaggle/input/aio-hutech/train'

# Define class names
CLASS_NAMES = ['nấm mỡ', 'bào ngư xám + trắng', 'Đùi gà Baby (cắt ngắn)', 'linh chi trắng']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLASSIFICATION_FILES = {
    'nấm mỡ': 'd:\\project\\oai\\split_cv\\classifications\\classifications_nm.json',
    'bào ngư xám + trắng': 'd:\\project\\oai\\split_cv\\classifications\\classifications_bn.json',
    'linh chi trắng': 'd:\\project\\oai\\split_cv\\classifications\\classifications_lc.json',
    'Đùi gà Baby (cắt ngắn)': 'd:\\project\\oai\\split_cv/classifications/classifications_dg.json'
}

def load_classification_data():
    """
    Load classification data from JSON files and create a mapping
    from image ID to group name.
    """
    print("Loading classification data...")
    id_to_group = {}
    
    for class_name, file_path in CLASSIFICATION_FILES.items():
        if not os.path.exists(file_path):
            print(f"Warning: Classification file {file_path} not found")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                classifications = json.load(f)
            
            for img_id, group_name in classifications.items():
                # Store both the group name and the class for better tracking
                id_to_group[img_id] = {
                    'group': group_name,
                    'class_name': class_name
                }
                
            print(f"Loaded {len(classifications)} classifications from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return id_to_group

def create_group_cv_splits():
    """
    Create group-based cross-validation splits and save to CSV.
    Uses StratifiedGroupKFold to ensure class distribution is preserved while keeping groups intact.
    """
    print(f"Creating {CV_FOLDS}-fold group-based cross-validation splits...")
    
    # Load classification data for grouping
    id_to_group = load_classification_data()
    
    # Collect all image paths, labels, and groups
    image_paths = []
    labels = []
    groups = []
    image_ids = []
    
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
                # Extract image ID from filename (assuming format like "XX123.jpg" where XX123 is the ID)
                img_id = os.path.splitext(img_name)[0]
                
                # Get group information if available, otherwise use "unknown" + class name as group
                group_info = id_to_group.get(img_id, {'group': f"unknown_{class_name}", 'class_name': class_name})
                
                image_paths.append(img_path)
                labels.append(class_idx)
                groups.append(group_info['group'])
                image_ids.append(img_id)
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    groups = np.array(groups)
    image_ids = np.array(image_ids)
    
    # Create StratifiedGroupKFold splitter (or GroupKFold if scikit-learn version doesn't have StratifiedGroupKFold)
    try:
        # Try to use StratifiedGroupKFold (available in sklearn 0.24+)
        sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_generator = sgkf.split(image_paths, labels, groups)
        print("Using StratifiedGroupKFold for splitting")
    except AttributeError:
        # Fallback to GroupKFold if StratifiedGroupKFold is not available
        print("StratifiedGroupKFold not available, falling back to GroupKFold")
        gkf = GroupKFold(n_splits=CV_FOLDS)
        fold_generator = gkf.split(image_paths, labels, groups)
    
    # Initialize fold column with -1 (not assigned)
    fold_assignments = -1 * np.ones(len(labels), dtype=int)
    
    # Assign fold IDs to each sample
    for fold_idx, (_, val_idx) in enumerate(fold_generator):
        fold_assignments[val_idx] = fold_idx
    
    # Create DataFrame with all information
    cv_df = pd.DataFrame({
        'image_path': image_paths,
        'image_id': image_ids,
        'class_id': labels, 
        'class_name': [CLASS_NAMES[label] for label in labels],
        'group': groups,
        'fold': fold_assignments
    })
    
    # Verify that all samples have been assigned a fold
    assert (cv_df['fold'] >= 0).all(), "Error: Some samples were not assigned to any fold"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Save to CSV
    cv_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Group-based cross-validation splits saved to {OUTPUT_CSV}")
    
    return cv_df

def analyze_group_cv_splits(cv_df):
    """
    Analyze the class and group distribution in each fold.
    """
    print("\nAnalyzing group-based cross-validation splits...")
    
    # Count samples per class per fold
    fold_class_counts = {}
    for fold in range(CV_FOLDS):
        fold_data = cv_df[cv_df['fold'] == fold]
        class_counts = fold_data['class_id'].value_counts().sort_index()
        fold_class_counts[fold] = class_counts
    
    # Convert to DataFrame for easier analysis
    distribution_df = pd.DataFrame(fold_class_counts).T
    distribution_df.columns = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if i in distribution_df.columns]
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
    plt.savefig('split_cv/group_cv_class_distribution.png')
    
    # Group distribution analysis
    print("\nAnalyzing group distribution across folds...")
    
    # Count unique groups per fold
    fold_group_counts = {}
    for fold in range(CV_FOLDS):
        # Groups in validation set
        val_groups = set(cv_df[cv_df['fold'] == fold]['group'])
        
        # Groups in training set (all folds except current)
        train_groups = set(cv_df[cv_df['fold'] != fold]['group'])
        
        # Check for overlap (should be none with group k-fold)
        overlap = val_groups.intersection(train_groups)
        
        fold_group_counts[fold] = {
            'val_group_count': len(val_groups),
            'train_group_count': len(train_groups),
            'overlap_count': len(overlap),
            'overlap_groups': list(overlap) if overlap else "None"
        }
    
    # Convert to DataFrame for display
    group_df = pd.DataFrame(fold_group_counts).T
    group_df.index.name = 'Fold'
    
    print("\nGroup distribution across folds:")
    print(group_df)
    
    # Create visualization of group distribution
    # Count samples per group per fold
    group_fold_counts = cv_df.pivot_table(index='group', columns='fold', 
                                         aggfunc='size', fill_value=0)
    
    # Select top groups by total sample count for visualization (to keep plot readable)
    top_groups = group_fold_counts.sum(axis=1).nlargest(20).index
    top_groups_df = group_fold_counts.loc[top_groups]
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(top_groups_df, annot=True, fmt="d", cmap="YlGnBu", 
                cbar_kws={'label': 'Number of Samples'})
    plt.title('Distribution of Top 20 Groups Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Group')
    plt.tight_layout()
    plt.savefig('split_cv/group_cv_group_distribution.png')
    
    return distribution_df, group_df

def verify_group_separation(cv_df):
    """
    Verify that no group appears in both training and validation sets for any fold.
    """
    print("\nVerifying group separation across train/validation splits...")
    violations = []
    
    for fold in range(CV_FOLDS):
        val_groups = set(cv_df[cv_df['fold'] == fold]['group'])
        train_groups = set(cv_df[cv_df['fold'] != fold]['group'])
        
        overlap = val_groups.intersection(train_groups)
        if overlap:
            violations.append({
                'fold': fold,
                'overlapping_groups': list(overlap),
                'overlap_count': len(overlap)
            })
    
    if violations:
        print("WARNING: Found violations in group separation!")
        for v in violations:
            print(f"Fold {v['fold']}: {v['overlap_count']} overlapping groups")
        
        # Show details for the first few violations
        first_violation = violations[0]
        overlap_groups = first_violation['overlapping_groups'][:5]  # Show only first 5 for brevity
        print(f"Example overlapping groups in fold {first_violation['fold']}: {overlap_groups}")
        
        # Get samples for overlapping groups
        for group in overlap_groups:
            group_samples = cv_df[cv_df['group'] == group]
            print(f"\nGroup '{group}' appears in the following folds:")
            print(group_samples[['image_id', 'fold', 'class_name']].head(10))
    else:
        print("✓ All groups are properly separated between train and validation sets")
    
    return violations

def main():
    """Main function to create and analyze group-based CV splits."""
    # Create group-based CV splits
    cv_df = create_group_cv_splits()
    
    # Analyze the splits
    class_dist, group_dist = analyze_group_cv_splits(cv_df)
    
    # Verify group separation
    violations = verify_group_separation(cv_df)
    
    if not violations:
        print("\nSuccess: Group-based cross-validation completed without violations.")
    else:
        print("\nWarning: Group-based cross-validation completed with violations.")
    
if __name__ == "__main__":
    # for i in range(30,50):
    #     print(f"Run {i}")
    #     RANDOM_SEED = i
    main()
