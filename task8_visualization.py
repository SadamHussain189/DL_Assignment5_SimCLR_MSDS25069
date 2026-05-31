"""
Task 8: PCA or t-SNE Feature Visualization - 8 Marks
=====================================================
Extract 512-dimensional features from three encoders and visualize using PCA/t-SNE:
1. Random untrained encoder
2. SimCLR pretrained encoder
3. Fine-tuned encoder (once Task 7 completes)

Author: Sadam Hussain (MSDS25069)
Date: May 31, 2026
"""

import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from models import SimCLRModel, ResNet18Encoder
from utils.dataset_splits import get_cifar10_subset

# Set device and seed
DEVICE = torch.device("cpu")
SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================
BATCH_SIZE = 64
N_VAL_SAMPLES = 1000  # Use 1000 validation images
USE_TSNE = False  # Set to True for t-SNE (slower but often better)

# Dataset paths
DATA_ROOT = './data'
VAL_SPLIT = './splits/val.txt'

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_features(model, dataloader, device, max_samples=None):
    """Extract encoder features from validation dataset"""
    model.eval()
    features_list = []
    labels_list = []
    sample_count = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extracting features', leave=False):
            images = images.to(device)
            features = model(images)  # (batch_size, 512)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
            sample_count += len(images)
            if max_samples and sample_count >= max_samples:
                break
    
    features = np.vstack(features_list)[:max_samples]
    labels = np.hstack(labels_list)[:max_samples]
    
    print(f"  Extracted {len(features)} features (shape: {features.shape})")
    return features, labels

# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================
def reduce_with_pca(features, n_components=2):
    """Reduce to 2D using PCA"""
    pca = PCA(n_components=n_components, random_state=SEED)
    reduced = pca.fit_transform(features)
    print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    return reduced, pca

def reduce_with_tsne(features, n_components=2):
    """Reduce to 2D using t-SNE"""
    tsne = TSNE(n_components=n_components, random_state=SEED, perplexity=30, n_iter=1000)
    reduced = tsne.fit_transform(features)
    return reduced, tsne

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_features(reduced, labels, title, output_path):
    """Plot 2D features colored by class"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map for 10 classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for c in range(10):
        mask = labels == c
        ax.scatter(reduced[mask, 0], reduced[mask, 1], c=[colors[c]], 
                  label=class_names[c], alpha=0.6, s=30)
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def compute_class_separation(features, labels):
    """Compute within-class and between-class variance"""
    n_classes = len(np.unique(labels))
    class_centers = np.array([features[labels == c].mean(axis=0) for c in range(n_classes)])
    
    # Within-class variance
    within_class_var = 0.0
    for c in range(n_classes):
        class_points = features[labels == c]
        if len(class_points) > 0:
            within_class_var += np.mean(np.linalg.norm(class_points - class_centers[c], axis=1) ** 2)
    within_class_var /= n_classes
    
    # Between-class variance
    overall_mean = features.mean(axis=0)
    between_class_var = np.mean([np.linalg.norm(class_centers[c] - overall_mean) ** 2 
                                  for c in range(n_classes)])
    
    separation_ratio = between_class_var / (within_class_var + 1e-8)
    
    return {
        'within_class_var': float(within_class_var),
        'between_class_var': float(between_class_var),
        'separation_ratio': float(separation_ratio)
    }

# ============================================================================
# MODEL LOADING
# ============================================================================
def get_resnet18():
    """Get fresh ResNet-18 for CIFAR-10"""
    model = ResNet18Encoder().to(DEVICE)
    return model

def get_simclr_pretrained():
    """Load SimCLR pretrained encoder"""
    if not os.path.exists('models/simclr_pretrained.pth'):
        print("ERROR: models/simclr_pretrained.pth not found!")
        return None
    
    model = SimCLRModel().to(DEVICE)
    checkpoint = torch.load('models/simclr_pretrained.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint)
    return model.encoder

def get_finetuned_encoder():
    """Load fine-tuned encoder if available"""
    if not os.path.exists('models/finetuned_model.pth'):
        print("  → Fine-tuned model not available yet, using SimCLR as substitute")
        return get_simclr_pretrained()
    
    try:
        # Load the finetuned model and extract encoder
        checkpoint = torch.load('models/finetuned_model.pth', map_location=DEVICE)
        # The checkpoint should contain LinearClassifier state
        # We need to create the model structure and load it
        from models import ResNet18Encoder
        encoder = ResNet18Encoder().to(DEVICE)
        
        # Filter only encoder weights from checkpoint
        encoder_dict = {k.replace('encoder.', ''): v for k, v in checkpoint.items() if k.startswith('encoder.')}
        
        if encoder_dict:
            encoder.load_state_dict(encoder_dict, strict=False)
            print("  → Loaded fine-tuned encoder")
            return encoder
        else:
            print("  → Could not extract encoder from checkpoint, using SimCLR")
            return get_simclr_pretrained()
    except Exception as e:
        print(f"  → Error loading fine-tuned model ({e}), using SimCLR")
        return get_simclr_pretrained()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*70)
    print("Task 8: PCA/t-SNE Feature Visualization")
    print("="*70)
    
    # Data transformation (no augmentation for visualization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2470, 0.2435, 0.2616))
    ])
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = get_cifar10_subset(DATA_ROOT, VAL_SPLIT, train=True, transform=transform)
    
    # Limit to N_VAL_SAMPLES
    if len(val_dataset) > N_VAL_SAMPLES:
        indices = np.random.choice(len(val_dataset), N_VAL_SAMPLES, replace=False)
        val_dataset = Subset(val_dataset, indices)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Using {len(val_dataset)} validation samples")
    
    # Get ground truth labels from validation set
    print("\nExtracting validation labels...")
    val_labels_list = []
    for _, labels in tqdm(val_loader, desc='Reading labels', leave=False):
        val_labels_list.append(labels.numpy())
    val_labels = np.hstack(val_labels_list)
    
    # ========================================================================
    # EXTRACT FEATURES FROM THREE ENCODERS
    # ========================================================================
    print("\n" + "="*70)
    print("FEATURE EXTRACTION")
    print("="*70)
    
    # 1. Random untrained encoder
    print("\n1. Random untrained encoder")
    encoder_random = get_resnet18()
    features_random, _ = extract_features(encoder_random, val_loader, DEVICE, len(val_dataset))
    
    # 2. SimCLR pretrained encoder
    print("\n2. SimCLR pretrained encoder")
    encoder_simclr = get_simclr_pretrained()
    if encoder_simclr is None:
        print("ERROR: Cannot load SimCLR model")
        return
    features_simclr, _ = extract_features(encoder_simclr, val_loader, DEVICE, len(val_dataset))
    
    # 3. Fine-tuned encoder
    print("\n3. Fine-tuned encoder")
    encoder_finetuned = get_finetuned_encoder()
    features_finetuned, _ = extract_features(encoder_finetuned, val_loader, DEVICE, len(val_dataset))
    
    # ========================================================================
    # COMPUTE CLASS SEPARATION METRICS
    # ========================================================================
    print("\n" + "="*70)
    print("CLASS SEPARATION ANALYSIS")
    print("="*70)
    
    sep_random = compute_class_separation(features_random, val_labels)
    sep_simclr = compute_class_separation(features_simclr, val_labels)
    sep_finetuned = compute_class_separation(features_finetuned, val_labels)
    
    print(f"\nClass Separation Ratios (higher is better):")
    print(f"  Random encoder:      {sep_random['separation_ratio']:.4f}")
    print(f"  SimCLR encoder:      {sep_simclr['separation_ratio']:.4f}")
    print(f"  Fine-tuned encoder:  {sep_finetuned['separation_ratio']:.4f}")
    
    # ========================================================================
    # DIMENSIONALITY REDUCTION AND VISUALIZATION
    # ========================================================================
    print("\n" + "="*70)
    print("DIMENSIONALITY REDUCTION & VISUALIZATION")
    print("="*70)
    
    method = 't-SNE' if USE_TSNE else 'PCA'
    reduce_fn = reduce_with_tsne if USE_TSNE else reduce_with_pca
    
    print(f"\nUsing {method} for dimensionality reduction...")
    
    print("\n1. Random encoder")
    reduced_random, _ = reduce_fn(features_random)
    plot_features(reduced_random, val_labels, 
                 f'Random Encoder - {method}',
                 f'graphs/visualization_random_{method.lower()}.png')
    
    print("\n2. SimCLR encoder")
    reduced_simclr, _ = reduce_fn(features_simclr)
    plot_features(reduced_simclr, val_labels,
                 f'SimCLR Encoder - {method}',
                 f'graphs/visualization_simclr_{method.lower()}.png')
    
    print("\n3. Fine-tuned encoder")
    reduced_finetuned, _ = reduce_fn(features_finetuned)
    plot_features(reduced_finetuned, val_labels,
                 f'Fine-tuned Encoder - {method}',
                 f'graphs/visualization_finetuned_{method.lower()}.png')
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    results = {
        'task': 'Task 8: Visualization',
        'method': method,
        'n_samples': len(val_dataset),
        'class_separation': {
            'random': sep_random,
            'simclr': sep_simclr,
            'finetuned': sep_finetuned
        }
    }
    
    with open('results/task8_visualization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Saved results to results/task8_visualization_results.json")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    methods = ['Random', 'SimCLR', 'Fine-tuned']
    separations = [sep_random['separation_ratio'], 
                   sep_simclr['separation_ratio'],
                   sep_finetuned['separation_ratio']]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    
    bars = axes[0].bar(methods, separations, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Class Separation Ratio', fontsize=12)
    axes[0].set_title('Class Separation Comparison', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, separations):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot the 2D visualizations
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    colors_map = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for ax_idx, (reduced, title) in enumerate([
        (reduced_random, 'Random Encoder'),
        (reduced_simclr, 'SimCLR Encoder')
    ]):
        ax = axes[ax_idx + 1]
        for c in range(10):
            mask = val_labels == c
            ax.scatter(reduced[mask, 0], reduced[mask, 1], c=[colors_map[c]], 
                      alpha=0.5, s=20)
        ax.set_xlabel('Component 1', fontsize=10)
        ax.set_ylabel('Component 2', fontsize=10)
        ax.set_title(f'{title} - {method}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('graphs', exist_ok=True)
    plt.savefig('graphs/visualization_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comparison plot to graphs/visualization_comparison.png")
    plt.close()
    
    print("\n✅ Task 8 Complete!")
    return results

if __name__ == "__main__":
    main()
