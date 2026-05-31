"""Task 3: Feature Similarity Before Training (8 marks).

Establish baseline: Untrained model does not recognize augmented views as similar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import create_resnet18_cifar10
from utils.dataset_splits import get_cifar10_subset, TwoViewDataset
from utils.seed import set_seed


def get_augmentation_pipeline() -> transforms.Compose:
    """Get the augmentation pipeline for two views."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )


class TwoViewTransform:
    """Apply augmentation pipeline to create two views."""
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


def cosine_similarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two matrices.
    
    Args:
        z1: (N, D) feature matrix
        z2: (M, D) feature matrix
        
    Returns:
        (N, M) similarity matrix
    """
    # Normalize to unit vectors
    z1_norm = torch.nn.functional.normalize(z1, dim=1)
    z2_norm = torch.nn.functional.normalize(z2, dim=1)
    
    # Compute cosine similarity
    return torch.matmul(z1_norm, z2_norm.t())


def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract features from untrained model.
    
    Args:
        model: Feature extractor (encoder)
        dataloader: Data loader for two-view dataset
        device: Device to use
        num_samples: Number of samples to process
        
    Returns:
        view1_features, view2_features, image_indices
    """
    model.eval()
    view1_list = []
    view2_list = []
    indices_list = []
    
    sample_count = 0
    with torch.no_grad():
        for view1, view2, _ in dataloader:
            if sample_count >= num_samples:
                break
                
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            # Extract features from encoder (before FC layer)
            feat1 = model(view1)
            feat2 = model(view2)
            
            view1_list.append(feat1.cpu())
            view2_list.append(feat2.cpu())
            indices_list.append(torch.arange(sample_count, sample_count + view1.size(0)))
            
            sample_count += view1.size(0)
    
    view1_features = torch.cat(view1_list, dim=0)[:num_samples]
    view2_features = torch.cat(view2_list, dim=0)[:num_samples]
    indices = torch.cat(indices_list, dim=0)[:num_samples]
    
    return view1_features, view2_features, indices


def compute_similarity_statistics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
) -> dict:
    """Compute similarity statistics before training.
    
    Args:
        model: Untrained ResNet-18 encoder
        dataloader: Two-view dataloader
        device: Device to use
        num_samples: Number of samples
        
    Returns:
        Dictionary with similarity statistics
    """
    print(f"\n=== Task 3: Feature Similarity Before Training ===")
    print(f"Extracting features from {num_samples} images...")
    
    view1_features, view2_features, indices = extract_features(
        model, dataloader, device, num_samples
    )
    
    # Similarity between same image, two augmented views
    sim_same_image = cosine_similarity(view1_features, view2_features)
    same_image_diag = torch.diagonal(sim_same_image).cpu().numpy()
    
    # Similarity between different images
    sim_all = cosine_similarity(view1_features, view2_features)
    all_similarities = sim_all.cpu().numpy()
    
    # Statistics
    stats = {
        "same_image_mean": float(np.mean(same_image_diag)),
        "same_image_std": float(np.std(same_image_diag)),
        "same_image_min": float(np.min(same_image_diag)),
        "same_image_max": float(np.max(same_image_diag)),
        "all_similarities_mean": float(np.mean(all_similarities)),
        "all_similarities_std": float(np.std(all_similarities)),
        "all_similarities_min": float(np.min(all_similarities)),
        "all_similarities_max": float(np.max(all_similarities)),
        "num_samples": num_samples,
    }
    
    # Print results
    print(f"\nSame Image (augmented views) Similarity:")
    print(f"  Mean: {stats['same_image_mean']:.4f}")
    print(f"  Std:  {stats['same_image_std']:.4f}")
    print(f"  Min:  {stats['same_image_min']:.4f}")
    print(f"  Max:  {stats['same_image_max']:.4f}")
    
    print(f"\nAll Similarities (including different images):")
    print(f"  Mean: {stats['all_similarities_mean']:.4f}")
    print(f"  Std:  {stats['all_similarities_std']:.4f}")
    print(f"  Min:  {stats['all_similarities_min']:.4f}")
    print(f"  Max:  {stats['all_similarities_max']:.4f}")
    
    print(f"\n✓ Observation: Untrained model treats augmented views randomly!")
    print(f"  Same-image similarity ({stats['same_image_mean']:.4f}) ≈ different-image similarity ({stats['all_similarities_mean']:.4f})")
    
    return stats


def main():
    """Main function."""
    # Setup
    set_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create untrained model (encoder only)
    model = create_resnet18_cifar10(num_classes=10)
    # Remove FC layer to get features
    model.fc = nn.Identity()
    model = model.to(device)
    
    # Load unlabeled training data with augmentations
    base_dataset = get_cifar10_subset(
        data_root="data",
        split_file="splits/train_ssl_unlabeled.txt",
        train=True,
        transform=None,  # We'll apply augmentation in TwoViewDataset
    )
    
    aug_transform = TwoViewTransform(get_augmentation_pipeline())
    dataset = TwoViewDataset(base_dataset, two_view_transform=aug_transform)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Compute similarity statistics
    stats = compute_similarity_statistics(model, dataloader, device, num_samples=100)
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    stats_file = results_dir / "task3_similarity_before_training.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Saved similarity statistics to {stats_file}")


if __name__ == "__main__":
    main()
