"""Quick version of Task 5 for testing - just 50 full epochs."""

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


def train_epoch(model, dataloader, optimizer, device, temperature):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
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
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("Task 5: SimCLR Pretraining (12 marks)")
    print("="*70)
    
    # Setup
    set_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Training configuration (FIXED)
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    TEMPERATURE = 0.5
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Optimizer: Adam")
    
    # Create model and optimizer
    model = SimCLRModel(encoder_dim=512, hidden_dim=256, proj_dim=128)
    model = model.to(device)
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
    print(f"Starting SimCLR Pretraining (50 epochs)...")
    print(f"="*70 + "\n")
    
    losses = []
    
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, dataloader, optimizer, device, TEMPERATURE)
        losses.append(loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {loss:.4f}")
    
    print(f"\n" + "="*70)
    print(f"✓ Training Complete!")
    print(f"="*70)
    print(f"\nFinal Statistics:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Improvement:  {losses[0] - losses[-1]:.4f}")
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "simclr_pretrained.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Plot loss curve
    graphs_dir = Path("graphs")
    graphs_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=2, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('NT-Xent Loss', fontsize=12)
    ax.set_title('SimCLR Pretraining Loss (50 epochs)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add annotations
    ax.text(0, losses[0], f'{losses[0]:.4f}', ha='right', va='bottom', fontsize=10)
    ax.text(EPOCHS-1, losses[-1], f'{losses[-1]:.4f}', ha='right', va='top', fontsize=10)
    
    plt.tight_layout()
    loss_plot_path = graphs_dir / "simclr_pretraining_loss.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss curve saved to {loss_plot_path}")
    plt.close()
    
    # Compute final similarity statistics
    print(f"\nComputing post-training similarity matrix...")
    model.eval()
    
    # Load sample unlabeled data
    sample_dataset = get_cifar10_subset(
        data_root="data",
        split_file="splits/train_ssl_unlabeled.txt",
        train=True,
        transform=None,
    )
    aug_transform_sample = TwoViewTransform(get_augmentation_pipeline())
    sample_dataset_augmented = TwoViewDataset(sample_dataset, two_view_transform=aug_transform_sample)
    sample_loader = DataLoader(sample_dataset_augmented, batch_size=64, shuffle=False, num_workers=0)
    
    # Extract features
    all_proj_features = []
    with torch.no_grad():
        for view1, view2, _ in sample_loader:
            view1 = view1.to(device)
            view2 = view2.to(device)
            _, proj1 = model(view1)
            _, proj2 = model(view2)
            all_proj_features.append(proj1.cpu())
            all_proj_features.append(proj2.cpu())
            if len(all_proj_features) * 64 >= 2000:
                break
    
    all_proj_features = torch.cat(all_proj_features, dim=0)[:2000]
    
    # Compute similarities
    sim_matrix = cosine_similarity(all_proj_features, all_proj_features)
    
    # Visualize
    visualize_similarity_matrix(all_proj_features, batch_size=1000)
    plt.savefig(results_dir / "similarity_matrix_after_training.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Similarity matrix saved to results/similarity_matrix_after_training.png")
    
    # Save statistics
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    stats = {
        "epochs_trained": EPOCHS,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "loss_improvement": float(losses[0] - losses[-1]),
        "all_losses": [float(l) for l in losses],
    }
    
    stats_file = results_dir / "task5_training_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Training statistics saved to {stats_file}\n")


if __name__ == "__main__":
    main()
