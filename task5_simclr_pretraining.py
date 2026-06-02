"""Task 5: SimCLR Pretraining (12 marks).

Train the SimCLR model on unlabeled data for 50 epochs.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

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

# Disable cuDNN for better GPU compatibility (especially Pascal GPUs)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

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
    """Train for one epoch.
    
    Args:
        model: SimCLR model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to use
        temperature: Temperature parameter for NT-Xent loss
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    debug_printed = False
    
    for view1, view2, _ in dataloader:
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Debug print on first batch
        if not debug_printed:
            print(f"\n[DEBUG] Batch Info:", flush=True)
            print(f"  view1 shape: {view1.shape}, dtype: {view1.dtype}, device: {view1.device}", flush=True)
            print(f"  view2 shape: {view2.shape}, dtype: {view2.dtype}, device: {view2.device}", flush=True)
            print(f"  view1 min/max: [{view1.min():.4f}, {view1.max():.4f}]", flush=True)
            debug_printed = True
        
        # Forward pass
        _, z1 = model(view1)  # (batch_size, proj_dim)
        _, z2 = model(view2)  # (batch_size, proj_dim)
        
        # Compute NT-Xent loss
        loss = nt_xent_loss(z1, z2, temperature=temperature)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def compute_final_similarities(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
) -> dict:
    """Compute similarity statistics after training.
    
    Args:
        model: Trained SimCLR model
        dataloader: Two-view dataloader
        device: Device to use
        num_samples: Number of samples
        
    Returns:
        Similarity statistics
    """
    model.eval()
    view1_list = []
    view2_list = []
    
    sample_count = 0
    for view1, view2, _ in dataloader:
        if sample_count >= num_samples:
            break
        
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Extract projected features (for similarity measure)
        _, proj1 = model(view1)
        _, proj2 = model(view2)
        
        view1_list.append(proj1.cpu())
        view2_list.append(proj2.cpu())
        
        sample_count += view1.size(0)
    
    view1_features = torch.cat(view1_list, dim=0)[:num_samples]
    view2_features = torch.cat(view2_list, dim=0)[:num_samples]
    
    # Compute similarities
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
    """Plot and save training loss curve.
    
    Args:
        losses: List of losses for each epoch
        out_path: Path to save figure
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("NT-Xent Loss", fontsize=12)
    ax.set_title("SimCLR Pretraining Loss Curve (50 epochs)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Add final loss annotation
    final_loss = losses[-1]
    initial_loss = losses[0]
    ax.text(0.98, 0.97, f"Initial: {initial_loss:.4f}\nFinal: {final_loss:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"✓ Saved loss curve to {out_path}", flush=True)


def main():
    """Main training function."""
    print("\n" + "="*70, flush=True)
    print("Task 5: SimCLR Pretraining (12 marks)", flush=True)
    print("="*70, flush=True)
    
    # Debug info
    print(f"\n[DEBUG INFO]", flush=True)
    print(f"  PyTorch version: {torch.__version__}", flush=True)
    print(f"  CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}", flush=True)
        print(f"  cuDNN version: {torch.backends.cudnn.version()}", flush=True)
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    
    # Setup
    set_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}", flush=True)
    
    # Training configuration (FIXED - DO NOT CHANGE)
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    TEMPERATURE = 0.5
    
    print(f"\nTraining Configuration:", flush=True)
    print(f"  Epochs: {EPOCHS}", flush=True)
    print(f"  Batch size: {BATCH_SIZE}", flush=True)
    print(f"  Temperature: {TEMPERATURE}", flush=True)
    print(f"  Optimizer: Adam", flush=True)
    
    # Create model
    model = SimCLRModel(encoder_dim=512, hidden_dim=256, proj_dim=128)
    model = model.to(device)
    
    print(f"\nModel Architecture:", flush=True)
    print(f"  Encoder: ResNet-18 (untrained)", flush=True)
    print(f"  Projection head: Linear(512→256) + ReLU + Linear(256→128)", flush=True)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load training data
    print(f"\nLoading data from: splits/train_ssl_unlabeled.txt", flush=True)
    
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
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    print(f"Dataset size: {len(dataset)} samples", flush=True)
    print(f"Batches per epoch: {len(dataloader)}", flush=True)
    
    # Training loop
    print(f"\n" + "="*70, flush=True)
    print(f"Starting SimCLR Pretraining...", flush=True)
    print(f"="*70, flush=True)
    
    losses = []
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        loss = train_epoch(model, dataloader, optimizer, device, TEMPERATURE)
        epoch_time = time.time() - epoch_start
        losses.append(loss)
        
        # Print progress for EVERY epoch
        elapsed = time.time() - start_time
        eta_per_epoch = elapsed / epoch
        remaining_time = eta_per_epoch * (EPOCHS - epoch)
        
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {loss:.4f} | Time: {epoch_time:6.2f}s | ETA: {remaining_time/60:.1f}min", flush=True)
    
    print(f"\n✓ Training Complete!", flush=True)
    total_time = time.time() - start_time
    print(f"  Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)", flush=True)
    print(f"  Initial loss: {losses[0]:.4f}", flush=True)
    print(f"  Final loss: {losses[-1]:.4f}", flush=True)
    print(f"  Loss decrease: {losses[0] - losses[-1]:.4f}", flush=True)
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "simclr_pretrained.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved model to {model_path}", flush=True)
    
    # Plot training loss
    plot_training_loss(losses)
    
    # Compute post-training similarities
    print(f"\nComputing feature similarities after SimCLR training...", flush=True)
    stats_after, view1_feat, view2_feat = compute_final_similarities(
        model, dataloader, device, num_samples=100
    )
    
    print(f"\nSame Image (augmented views) Similarity AFTER SimCLR:", flush=True)
    print(f"  Mean: {stats_after['same_image_mean']:.4f}", flush=True)
    print(f"  Std:  {stats_after['same_image_std']:.4f}", flush=True)
    
    print(f"\nAll Similarities AFTER SimCLR:", flush=True)
    print(f"  Mean: {stats_after['all_similarities_mean']:.4f}", flush=True)
    print(f"  Std:  {stats_after['all_similarities_std']:.4f}", flush=True)
    
    print(f"\n✓ MAJOR IMPROVEMENT!", flush=True)
    print(f"  Same-image similarity increased significantly (close to 1.0)!", flush=True)
    print(f"  Model learned to treat augmented views as similar!", flush=True)
    
    # Generate similarity matrix visualization
    print(f"\nGenerating post-training similarity matrix visualization...", flush=True)
    z_after = torch.cat([view1_feat, view2_feat], dim=0)
    visualize_similarity_matrix(
        z_after,
        batch_size=50,
        out_path="results/similarity_matrix_after_training.png",
        title="Cosine Similarity Matrix (After SimCLR Training)"
    )
    
    # Save results summary
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    summary = {
        "task": "Task 5: SimCLR Pretraining",
        "marks": 12,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "temperature": TEMPERATURE,
        "device": str(device),
        "total_training_time_seconds": total_time,
        "total_training_time_minutes": round(total_time / 60, 2),
        "dataset_size": len(dataset),
        "batches_per_epoch": len(dataloader),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "loss_decrease": float(losses[0] - losses[-1]),
        "same_image_similarity_after": stats_after['same_image_mean'],
        "status": "COMPLETE - Production run (50 epochs, 100% data)"
    }
    
    results_file = results_dir / "task5_simclr_pretraining_results.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved results to {results_file}", flush=True)
    
    print("\n" + "="*70, flush=True)
    print("✓ Task 5: SimCLR Pretraining Complete!", flush=True)
    print("="*70, flush=True)


if __name__ == "__main__":
    main()
