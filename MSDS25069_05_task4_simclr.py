"""Task 4 & 5: SimCLR Implementation and Pretraining (24 + 12 Marks).

Task 4.1 - Encoder and Projection Head (see models.py)
Task 4.2 - Positive and Negative Pair Construction
Task 4.3 - Similarity Matrix Visualization
Task 4.4 - NT-Xent Contrastive Loss Implementation
Task 5   - SimCLR Pretraining (50 epochs on unlabeled data)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from models import SimCLRModel
from utils.dataset_splits import get_cifar10_subset, TwoViewDataset
from utils.seed import set_seed

# Fixed training settings (Section 6 of assignment)
SEED = 2026
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
TEMPERATURE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def get_simclr_augmentation():
    return T.Compose([
        T.RandomResizedCrop(32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])


# ============================================================================
# Task 4.2: Positive and Negative Pair Construction
# ============================================================================

def print_pair_table(batch_size=4):
    print("\n=== Task 4.2: Positive and Negative Pair Construction ===")
    print(f"\nFor a batch of {batch_size} images -> 2*{batch_size} = {2*batch_size} augmented views")
    print(f"\n{'Original Image':<16} {'View 1 Index':<14} {'View 2 Index':<14} {'Positive Pair':<14}")
    print("-" * 58)
    for i in range(batch_size):
        print(f"image {i:<10} {i:<14} {batch_size + i:<14} {'yes':<14}")
    print(f"\nTotal positive pairs: {batch_size}")
    print(f"Total negative pairs per sample: {2 * batch_size - 2}")


# ============================================================================
# Task 4.3: Similarity Matrix Visualization
# ============================================================================

def visualize_similarity_matrix(z, batch_size, out_path, title="Cosine Similarity Matrix"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    z_norm = F.normalize(z, dim=1)
    sim = torch.mm(z_norm, z_norm.t()).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_xlabel("View Index")
    ax.set_ylabel("View Index")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(y=batch_size - 0.5, color="white", linewidth=2, linestyle="--", alpha=0.7)
    ax.axvline(x=batch_size - 0.5, color="white", linewidth=2, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================================
# Task 4.4: NT-Xent Contrastive Loss
# ============================================================================

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Args:
        z_i: (N, D) projections of view 1
        z_j: (N, D) projections of view 2
        temperature: temperature parameter tau

    Returns:
        Scalar loss value
    """
    batch_size = z_i.shape[0]
    device = z_i.device

    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    z = torch.cat([z_i, z_j], dim=0)

    sim_matrix = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

    log_probs = F.log_softmax(sim_matrix, dim=1)
    idx = torch.arange(batch_size, device=device)
    loss_i = -log_probs[idx, idx + batch_size]
    loss_j = -log_probs[idx + batch_size, idx]
    return (loss_i + loss_j).mean()


# ============================================================================
# Task 5: SimCLR Pretraining
# ============================================================================

def cosine_sim_matrix(z1, z2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return torch.mm(z1, z2.t())


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    Path("graphs").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Task 4: SimCLR Implementation (24 marks)")
    print("=" * 70)

    # --- 4.1 ---
    print("\nTask 4.1: Encoder and Projection Head (see models.py)")
    print("  Encoder: ResNet-18 modified for CIFAR-10 -> 512-dim")
    print("  Projection Head: Linear(512->256) + ReLU + Linear(256->128)")

    # --- 4.2 ---
    print_pair_table(batch_size=4)

    # --- 4.3: Similarity matrix BEFORE training ---
    base_dataset = get_cifar10_subset(
        data_root="data",
        split_file="splits/train_ssl_unlabeled.txt",
        train=True, transform=None,
    )
    two_view = TwoViewTransform(get_simclr_augmentation())
    dataset = TwoViewDataset(base_dataset, two_view_transform=two_view)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, drop_last=True)

    model = SimCLRModel().to(DEVICE)
    model.eval()

    with torch.no_grad():
        v1, v2, _ = next(iter(dataloader))
        v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
        _, proj1 = model(v1)
        _, proj2 = model(v2)
        z_before = torch.cat([proj1[:16], proj2[:16]], dim=0)

    visualize_similarity_matrix(
        z_before, batch_size=16,
        out_path="results/similarity_matrix_before_training.png",
        title="Cosine Similarity Matrix (Before Training)",
    )

    # --- 4.4: Test NT-Xent loss ---
    print("\n=== Task 4.4: NT-Xent Loss Test ===")
    z_test_i = torch.randn(4, 128)
    z_test_j = torch.randn(4, 128)
    loss_rand = nt_xent_loss(z_test_i, z_test_j, temperature=0.5)
    loss_same = nt_xent_loss(z_test_i, z_test_i, temperature=0.5)
    print(f"  Loss (random pairs):    {loss_rand.item():.4f}")
    print(f"  Loss (identical pairs): {loss_same.item():.4f}")

    # ==========================================================================
    # Task 5: SimCLR Pretraining
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Task 5: SimCLR Pretraining (12 marks)")
    print("=" * 70)

    print(f"\nTraining Configuration:")
    print(f"  Epochs:      {EPOCHS}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  LR:          {LEARNING_RATE}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Optimizer:   Adam")
    print(f"  Dataset:     {len(dataset)} unlabeled images")
    print(f"  Batches/epoch: {len(dataloader)}")

    # Re-initialise model for training from scratch
    set_seed(SEED)
    model = SimCLRModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for v1, v2, _ in dataloader:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = nt_xent_loss(z1, z2, temperature=TEMPERATURE)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        elapsed = time.time() - start
        eta = (elapsed / epoch) * (EPOCHS - epoch)
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min", flush=True)

    total_time = time.time() - start
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")

    # Save encoder (assignment requirement)
    torch.save(model.encoder.state_dict(), "models/simclr_encoder.pt")
    print("Saved: models/simclr_encoder.pt")

    # Save full model for downstream tasks
    torch.save(model.state_dict(), "models/simclr_pretrained.pth")
    print("Saved: models/simclr_pretrained.pth")

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, EPOCHS + 1), losses, linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent Loss")
    ax.set_title("SimCLR Pretraining Loss Curve", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.97, f"Initial: {losses[0]:.4f}\nFinal: {losses[-1]:.4f}",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    fig.savefig("graphs/simclr_pretraining_loss.png", dpi=150)
    plt.close(fig)
    print("Saved: graphs/simclr_pretraining_loss.png")

    # --- Post-training similarities ---
    model.eval()
    v1_feats, v2_feats = [], []
    count = 0
    with torch.no_grad():
        for v1, v2, _ in dataloader:
            if count >= 100:
                break
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            _, p1 = model(v1)
            _, p2 = model(v2)
            v1_feats.append(p1.cpu())
            v2_feats.append(p2.cpu())
            count += v1.size(0)

    f1 = torch.cat(v1_feats)[:100]
    f2 = torch.cat(v2_feats)[:100]

    sim = cosine_sim_matrix(f1, f2).numpy()
    same_sim = np.diag(sim)
    off_mask = ~np.eye(sim.shape[0], dtype=bool)
    diff_sim = sim[off_mask]

    print(f"\nPost-training similarity:")
    print(f"  Same image mean:      {np.mean(same_sim):.4f}")
    print(f"  Different image mean: {np.mean(diff_sim):.4f}")

    z_viz = torch.cat([f1[:50], f2[:50]], dim=0)
    visualize_similarity_matrix(
        z_viz, batch_size=50,
        out_path="results/similarity_matrix_after_training.png",
        title="Cosine Similarity Matrix (After SimCLR Training)",
    )

    # --- Save results ---
    summary = {
        "task": "Task 4+5: SimCLR Implementation & Pretraining",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "temperature": TEMPERATURE,
        "device": str(DEVICE),
        "total_time_minutes": round(total_time / 60, 2),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "all_losses": [float(l) for l in losses],
        "same_view_similarity_after": float(np.mean(same_sim)),
        "different_image_similarity_after": float(np.mean(diff_sim)),
    }
    with open("results/task5_simclr_pretraining_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved: results/task5_simclr_pretraining_results.json")
    print("\nTask 4+5 Complete!")


if __name__ == "__main__":
    main()
