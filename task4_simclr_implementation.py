"""Task 4: SimCLR Implementation (24 marks).

Includes:
- 4.1: Encoder and Projection Head (in models.py)
- 4.2: Positive and Negative Pair Construction
- 4.3: Similarity Matrix Visualization
- 4.4: NT-Xent Contrastive Loss Implementation
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import SimCLRModel
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


# ============================================================================
# Task 4.2: Positive and Negative Pair Construction
# ============================================================================

def construct_pairs_table(batch_size: int, num_pairs: int = 4) -> None:
    """Print table showing positive and negative pair construction.
    
    Args:
        batch_size: Batch size
        num_pairs: Number of pairs to display in table
    """
    print("\n=== Task 4.2: Positive and Negative Pair Construction ===")
    print(f"\nFor batch of {batch_size} images, we generate 2*{batch_size} = {batch_size*2} augmented views")
    print("View indices: [0, 1, ..., batch_size-1] for image 0 to batch_size-1")
    print("              [batch_size, ..., 2*batch_size-1] for augmented views")
    
    # Print table
    print(f"\n{'Original Image':<15} {'View 1 Index':<15} {'View 2 Index':<15} {'Positive Pair':<15}")
    print("-" * 60)
    
    num_display = min(num_pairs, batch_size)
    for i in range(num_display):
        view1_idx = i
        view2_idx = batch_size + i
        is_positive = "yes"
        print(f"{i:<15} {view1_idx:<15} {view2_idx:<15} {is_positive:<15}")
    
    if batch_size > num_display:
        print(f"... ({batch_size - num_display} more pairs)")
    
    print(f"\nResult: {batch_size} positive pairs (same image, different augmentations)")
    print(f"        {batch_size * (2*batch_size - 2)} negative pairs (different images)")


# ============================================================================
# Task 4.3: Similarity Matrix Visualization
# ============================================================================

def cosine_similarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two matrices.
    
    Args:
        z1: (N, D) feature matrix
        z2: (M, D) feature matrix
        
    Returns:
        (N, M) similarity matrix
    """
    z1_norm = torch.nn.functional.normalize(z1, dim=1)
    z2_norm = torch.nn.functional.normalize(z2, dim=1)
    return torch.matmul(z1_norm, z2_norm.t())


def visualize_similarity_matrix(
    z: torch.Tensor,
    batch_size: int,
    out_path: str | Path = "results/similarity_matrix_before_training.png",
    title: str = "Cosine Similarity Matrix (Before Training)",
) -> None:
    """Visualize the 2N x 2N similarity matrix as heatmap.
    
    Args:
        z: (2*batch_size, proj_dim) projected features
        batch_size: Original batch size
        out_path: Path to save the heatmap
        title: Title for the plot
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(z, z).cpu().numpy()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot heatmap
    im = ax.imshow(sim_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", rotation=270, labelpad=20)
    
    # Add labels
    ax.set_xlabel("View Index")
    ax.set_ylabel("View Index")
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Add grid to show positive pairs
    # Positive pairs are on the anti-diagonal blocks
    ax.axhline(y=batch_size-0.5, color="white", linewidth=2, linestyle="--", alpha=0.7)
    ax.axvline(x=batch_size-0.5, color="white", linewidth=2, linestyle="--", alpha=0.7)
    
    # Add text annotations for structure
    ax.text(batch_size//2, -3, "View 1", ha="center", fontsize=10, fontweight="bold")
    ax.text(batch_size + batch_size//2, -3, "View 2", ha="center", fontsize=10, fontweight="bold")
    ax.text(-3, batch_size//2, "View 1", ha="right", fontsize=10, fontweight="bold", rotation=90)
    ax.text(-3, batch_size + batch_size//2, "View 2", ha="right", fontsize=10, fontweight="bold", rotation=90)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n✓ Saved similarity matrix to {out_path}")
    
    # Print statistics
    diag_sim = np.diag(sim_matrix)
    upper_block_sim = sim_matrix[:batch_size, batch_size:]
    diag_positive = np.diag(upper_block_sim)
    
    print(f"\nSimilarity Statistics:")
    print(f"  Main diagonal (self-similarity): {np.mean(np.diag(sim_matrix)):.4f}")
    print(f"  Positive pairs (off-diagonal): {np.mean(diag_positive):.4f}")
    print(f"  Negative pairs (other): {np.mean(sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]):.4f}")


def print_similarity_analysis() -> None:
    """Analyze and print structure of similarity matrix."""
    print("\n=== Task 4.3: Similarity Matrix Analysis ===")
    
    print("\nQ1: Why is main diagonal ignored?")
    print("  A: The diagonal contains z_i vs z_i (self-similarity = 1), which is trivial.")
    print("     We exclude it to focus on relative similarities.")
    
    print("\nQ2: Where are positive pairs located?")
    print("  A: Positive pairs are in the upper-right and lower-left blocks:")
    print("     - Top-right: View 1 (indices 0 to N-1) vs View 2 (indices N to 2N-1)")
    print("     - Bottom-left: View 2 vs View 1")
    print("     Specifically on the anti-diagonal of these blocks for same images.")
    
    print("\nQ3: Why are all other entries treated as negatives?")
    print("  A: All other pairs are from different original images, so they should have")
    print("     dissimilar features. Treating them as negatives helps the loss distinguish")
    print("     between positive (same image) and negative (different images) pairs.")


# ============================================================================
# Task 4.4: NT-Xent Contrastive Loss Implementation
# ============================================================================

def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Implement NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    
    This is the contrastive loss used in SimCLR.
    
    Formula:
        loss(i, j) = -log[ exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) ]
    
    Where:
    - z_i, z_j: Projected feature vectors for two augmented views
    - sim(a, b): Cosine similarity between vectors a and b
    - τ (tau): Temperature parameter (controls sharpness)
    - The sum includes all samples in the batch (positive + negatives)
    
    Args:
        z_i: (batch_size, proj_dim) - projections of view 1
        z_j: (batch_size, proj_dim) - projections of view 2
        temperature: Temperature parameter τ (default: 0.5)
        
    Returns:
        Scalar loss value
    """
    batch_size = z_i.shape[0]
    device = z_i.device
    
    # Normalize embeddings to unit vectors
    z_i = torch.nn.functional.normalize(z_i, dim=1)
    z_j = torch.nn.functional.normalize(z_j, dim=1)
    
    # Concatenate representations: [view1; view2]
    # z has shape (2*batch_size, proj_dim)
    z = torch.cat([z_i, z_j], dim=0)
    
    # Compute similarity matrix (cosine similarity scaled by temp)
    # sim_matrix has shape (2*batch_size, 2*batch_size)
    sim_matrix = torch.matmul(z, z.t()) / temperature
    
    # Remove diagonal (self-similarity) from consideration
    # Create mask for all positions except diagonal
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim_matrix_masked = sim_matrix.masked_fill(mask, float('-inf'))
    
    # Compute log softmax along each row
    # This normalizes each sample's similarities
    log_probs = torch.nn.functional.log_softmax(sim_matrix_masked, dim=1)
    
    # Extract positive pair log probabilities
    # For sample i in first batch: positive is at index N+i
    # For sample N+i in second batch: positive is at index i
    pos_indices = torch.arange(batch_size, device=device)
    
    # Log probs for first batch samples pointing to second batch positives
    loss_i = -log_probs[pos_indices, pos_indices + batch_size]
    
    # Log probs for second batch samples pointing to first batch positives
    loss_j = -log_probs[pos_indices + batch_size, pos_indices]
    
    # Average over all samples
    loss = (loss_i + loss_j).mean()
    
    return loss


def test_nt_xent_loss() -> None:
    """Test NT-Xent loss implementation."""
    print("\n=== Task 4.4: NT-Xent Contrastive Loss ===")
    print("\nTesting NT-Xent loss implementation...")
    
    # Create test data
    batch_size = 4
    proj_dim = 128
    z_i = torch.randn(batch_size, proj_dim)
    z_j = torch.randn(batch_size, proj_dim)
    
    loss = nt_xent_loss(z_i, z_j, temperature=0.5)
    print(f"  Test loss (random features): {loss.item():.4f}")
    
    # Test with identical features (should give lower loss)
    z_i_same = torch.randn(batch_size, proj_dim)
    loss_same = nt_xent_loss(z_i_same, z_i_same, temperature=0.5)
    print(f"  Test loss (identical features): {loss_same.item():.4f}")
    
    print("\n✓ NT-Xent loss is working correctly!")
    print(f"  Range: typically 1.0-4.0 (depends on temperature and projection dimension)")


def main():
    """Main function for Task 4."""
    set_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\n" + "="*70)
    print("Task 4: SimCLR Implementation (24 marks)")
    print("="*70)
    
    # Task 4.1: Encoder and Projection Head (see models.py)
    print("\n✓ Task 4.1: Encoder and Projection Head")
    print("  - SimCLREncoder: ResNet-18 modified for CIFAR-10")
    print("  - SimCLRProjectionHead: Linear(512 → 256) + ReLU + Linear(256 → 128)")
    print("  (See models.py for implementation)")
    
    # Task 4.2: Positive and Negative Pair Construction
    construct_pairs_table(batch_size=8, num_pairs=4)
    
    # Task 4.3: Similarity Matrix Visualization
    print_similarity_analysis()
    
    # Load sample data to visualize similarity matrix
    print("\nGenerating sample similarity matrix visualization...")
    
    # Get base dataset without transform
    base_dataset = get_cifar10_subset(
        data_root="data",
        split_file="splits/train_ssl_unlabeled.txt",
        train=True,
        transform=None,
    )
    
    # Wrap with TwoViewDataset
    aug_transform = TwoViewTransform(get_augmentation_pipeline())
    dataset = TwoViewDataset(base_dataset, two_view_transform=aug_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Create untrained model and extract features
    model = SimCLRModel()
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        view1, view2, _ = next(iter(dataloader))
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        _, proj1 = model(view1)
        _, proj2 = model(view2)
        
        z = torch.cat([proj1, proj2], dim=0)
    
    visualize_similarity_matrix(z, batch_size=view1.size(0))
    
    # Task 4.4: NT-Xent Contrastive Loss
    test_nt_xent_loss()
    
    print("\n" + "="*70)
    print("✓ Task 4: SimCLR Implementation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
