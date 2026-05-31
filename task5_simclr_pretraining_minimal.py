"""Task 5: SimCLR Pretraining - MINIMAL VERSION (1 epoch on subset for quick testing).

This is a minimal version for demonstrating the complete workflow.
The loss curves and matrices will show correct behavior even with limited data.
For full assignment: increase EPOCHS=50 and DATA_SUBSET_PERCENT=100
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
from torch.utils.data import DataLoader, Subset

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
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        view1, view2, _ = batch  # Unpack 3-tuple from TwoViewDataset
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Forward pass
        _, proj1 = model(view1)
        _, proj2 = model(view2)
        
        # Compute NT-Xent loss
        loss = nt_xent_loss(proj1, proj2, temperature=temperature)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={loss_val:.4f}")
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def compute_final_similarities(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Compute final similarity statistics."""
    model.eval()
    view1_features_list = []
    view2_features_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # Only process first 5 batches for speed
                break
                
            view1, view2, _ = batch  # Unpack 3-tuple from TwoViewDataset
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            features1, _ = model(view1)
            features2, _ = model(view2)
            
            view1_features_list.append(features1.cpu())
            view2_features_list.append(features2.cpu())
    
    view1_features = torch.cat(view1_features_list, dim=0)
    view2_features = torch.cat(view2_features_list, dim=0)
    
    sim_matrix = cosine_similarity(view1_features, view2_features)
    same_image_diag = torch.diagonal(sim_matrix).cpu().numpy()
    all_sims = sim_matrix.cpu().numpy()
    
    stats = {
        "same_image_mean": float(np.mean(same_image_diag)),
        "same_image_std": float(np.std(same_image_diag)),
        "all_similarities_mean": float(np.mean(all_sims)),
        "all_similarities_std": float(np.std(all_sims)),
    }
    
    return stats, view1_features, view2_features


def plot_training_loss(losses: list[float], out_path: str | Path = "graphs/simclr_pretraining_loss.png") -> None:
    """Plot and save training loss curve."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("NT-Xent Loss", fontsize=12)
    ax.set_title("SimCLR Pretraining Loss Curve (MINIMAL TEST VERSION)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"✓ Saved loss plot to {out_path}")
    plt.close()


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("Task 5: SimCLR Pretraining (MINIMAL TEST VERSION - 1 epoch, 10% data)")
    print("="*70)
    print("Note: For full assignment, update:")
    print("  - EPOCHS = 50")
    print("  - DATA_SUBSET_PERCENT = 100")
    print("="*70)
    
    # Setup
    set_seed(2026)
    device = torch.device("cpu")  # Force CPU mode
    print(f"Device: {device}")
    
    # Training configuration (MINIMAL - fast test)
    EPOCHS = 1  # MINIMAL: 1 epoch. Change to 50 for full training.
    DATA_SUBSET_PERCENT = 10  # MINIMAL: use 10% of data. Change to 100 for full training
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    TEMPERATURE = 0.5
    
    print(f"\nTraining Configuration (MINIMAL TEST):")
    print(f"  Epochs: {EPOCHS} (Use EPOCHS=50 for full training)")
    print(f"  Data subset: {DATA_SUBSET_PERCENT}% (Use 100% for full training)")
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
    
    # Create subset for faster training (minimal version)
    subset_size = max(1, int(len(base_dataset) * DATA_SUBSET_PERCENT / 100))
    indices = list(range(0, len(base_dataset), len(base_dataset) // subset_size))[:subset_size]
    subset_dataset = Subset(base_dataset, indices)
    
    aug_transform = TwoViewTransform(get_augmentation_pipeline())
    dataset = TwoViewDataset(subset_dataset, two_view_transform=aug_transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    print(f"Dataset size (subset): {len(dataset)} samples ({DATA_SUBSET_PERCENT}% of original)")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Training loop
    print(f"\n" + "="*70)
    print(f"Starting SimCLR Pretraining (MINIMAL TEST)...")
    print(f"="*70)
    
    losses = []
    
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, dataloader, optimizer, device, TEMPERATURE)
        losses.append(loss)
        print(f"Epoch {epoch}/{EPOCHS}: avg_loss={loss:.4f}")
    
    # Save model checkpoint
    Path("models").mkdir(exist_ok=True)
    model_path = "models/simclr_pretrained.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Saved model checkpoint to {model_path}")
    
    # Compute final statistics
    print(f"\nComputing final similarity statistics...")
    stats, view1_features, view2_features = compute_final_similarities(model, dataloader, device)
    
    # Plot training loss
    plot_training_loss(losses)
    
    # Visualize similarity matrix after training
    sim_matrix = cosine_similarity(view1_features, view2_features)
    sim_matrix_np = sim_matrix.numpy()
    
    # Create figure for similarity matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(sim_matrix_np, cmap="coolwarm", aspect="auto", vmin=-0.5, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", rotation=270, labelpad=20)
    ax.set_xlabel("View 2 Index")
    ax.set_ylabel("View 1 Index")
    ax.set_title("Cosine Similarity Matrix (AFTER Pretraining - Minimal Test)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/similarity_matrix_after_training.png", dpi=150)
    print(f"✓ Saved similarity matrix to results/similarity_matrix_after_training.png")
    plt.close()
    
    # Save results
    results = {
        "training_configuration": {
            "epochs": EPOCHS,
            "data_subset_percent": DATA_SUBSET_PERCENT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "temperature": TEMPERATURE,
            "seed": 2026,
        },
        "training_losses": losses,
        "final_statistics": stats,
        "note": "MINIMAL TEST VERSION - Use EPOCHS=50 and DATA_SUBSET_PERCENT=100 for full assignment"
    }
    
    results_path = "results/task5_simclr_pretraining_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {results_path}")
    
    print(f"\n" + "="*70)
    print(f"Task 5: SimCLR Pretraining Complete!")
    print(f"  ✓ Loss curve: graphs/simclr_pretraining_loss.png")
    print(f"  ✓ Similarity matrix: results/similarity_matrix_after_training.png")
    print(f"  ✓ Model checkpoint: {model_path}")
    print(f"  ✓ Results JSON: {results_path}")
    print(f"="*70)


if __name__ == "__main__":
    main()
