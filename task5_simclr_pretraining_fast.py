"""Task 5: SimCLR Pretraining - FAST VERSION (5 epochs for testing).

For full 50-epoch training, set EPOCHS = 50 in the configuration below.
Full training will take ~1-2 hours on NVIDIA GPU or ~8-12 hours on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import SimCLRModel
from task4_simclr_implementation import nt_xent_loss, cosine_similarity, TwoViewTransform, visualize_similarity_matrix
from utils.dataset_splits import get_cifar10_subset, TwoViewDataset
from utils.seed import set_seed


def get_augmentation_pipeline() -> transforms.Compose:
    """Get the augmentation pipeline."""
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    temperature: float = 0.5,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for view1, view2, _ in dataloader:
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Forward pass
        _, proj1 = model(view1)
        _, proj2 = model(view2)
        
        # Compute loss
        loss = nt_xent_loss(proj1, proj2, temperature=temperature)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def compute_final_similarities(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
) -> dict:
    """Compute similarity statistics after training."""
    model.eval()
    
    all_proj = []
    count = 0
    
    with torch.no_grad():
        for view1, view2, _ in dataloader:
            if count >= num_samples:
                break
            
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            _, proj1 = model(view1)
            _, proj2 = model(view2)
            
            all_proj.append(torch.cat([proj1, proj2], dim=0))
            count += proj1.shape[0]
    
    all_proj = torch.cat(all_proj, dim=0)[:num_samples * 2]
    
    # Compute similarities
    sim_mat = cosine_similarity(all_proj, all_proj)
    
    stats = {
        "same_image_mean": float(torch.diagonal(sim_mat, offset=100).mean()),
        "all_pairs_mean": float(sim_mat.mean()),
        "positive_pairs_mean": float(sim_mat[:100, 100:].mean()),
    }
    
    return stats, all_proj[:100], all_proj[100:200]


def plot_training_loss(losses: list, save_path: str = "graphs/simclr_pretraining_loss.png"):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2, label="NT-Xent Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("SimCLR Pretraining Loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Annotate first and last loss
    plt.text(0, losses[0], f"{losses[0]:.4f}", ha="center", va="bottom", fontsize=10)
    plt.text(len(losses)-1, losses[-1], f"{losses[-1]:.4f}", ha="center", va="bottom", fontsize=10)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"✓ Saved loss plot to {save_path}")
    plt.close()


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("Task 5: SimCLR Pretraining (FAST TEST VERSION - 5 EPOCHS)")
    print("="*70)
    
    # Setup
    set_seed(2026)
    device = torch.device("cpu")  # Force CPU mode (CUDA causes compatibility issues)
    print(f"Device: {device}")
    
    # Training configuration
    EPOCHS = 5  # FAST VERSION: Use 5 epochs for testing. Change to 50 for full training.
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    TEMPERATURE = 0.5
    
    print(f"\nTraining Configuration (FAST TEST):")
    print(f"  Epochs: {EPOCHS} (Use EPOCHS=50 for full training)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Temperature: {TEMPERATURE}")
    
    # Create model
    model = SimCLRModel(encoder_dim=512, hidden_dim=256, proj_dim=128)
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load training data
    print(f"\nLoading data from: splits/train_ssl_unlabeled.txt")
    
    base_dataset = get_cifar10_subset(
        data_root="data",
        split_file="splits/train_ssl_unlabeled.txt",
        train=True,
        transform=None,
    )
    
    aug_transform = TwoViewTransform(get_augmentation_pipeline())
    dataset = TwoViewDataset(base_dataset, two_view_transform=aug_transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Training loop
    print(f"\n" + "="*70)
    print(f"Starting SimCLR Pretraining (FAST TEST)...")
    print(f"="*70)
    
    losses = []
    
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, dataloader, optimizer, device, TEMPERATURE)
        losses.append(loss)
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {loss:.4f}")
    
    # Compute final similarities
    print(f"\nComputing post-training similarities...")
    stats, proj_view1, proj_view2 = compute_final_similarities(model, dataloader, device, num_samples=100)
    
    # Visualize similarity matrix
    z = torch.cat([proj_view1, proj_view2], dim=0)
    visualize_similarity_matrix(z, batch_size=100, save_path="results/similarity_matrix_after_training.png")
    
    # Plot loss curve
    plot_training_loss(losses)
    
    # Save model checkpoint
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "simclr_pretrained.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved model checkpoint to {model_path}")
    
    # Save results
    results = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "temperature": TEMPERATURE,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "similarity_stats": {
            "same_image_mean": stats["same_image_mean"],
            "positive_pairs_mean": stats["positive_pairs_mean"],
            "all_pairs_mean": stats["all_pairs_mean"],
        },
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "task5_simclr_pretraining_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {results_file}")
    
    print(f"\n" + "="*70)
    print("✓ Task 5: SimCLR Pretraining Complete!")
    print("="*70)
    print(f"\nNOTE: This was a FAST TEST with {EPOCHS} epochs.")
    print(f"For full assignment, set EPOCHS=50 and re-run: python3 task5_simclr_pretraining.py")


if __name__ == "__main__":
    main()
