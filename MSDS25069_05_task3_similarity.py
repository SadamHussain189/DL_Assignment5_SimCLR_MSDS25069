"""Task 3: Feature Similarity Before Training (8 Marks).

Compute cosine similarity between augmented views using an untrained encoder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from models import ResNet18Encoder
from utils.dataset_splits import get_cifar10_subset, TwoViewDataset
from utils.seed import set_seed

SEED = 2026
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 100


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


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    Path("results").mkdir(parents=True, exist_ok=True)

    # Untrained encoder
    encoder = ResNet18Encoder().to(DEVICE)
    encoder.eval()

    base_dataset = get_cifar10_subset(
        data_root="data",
        split_file="splits/train_ssl_unlabeled.txt",
        train=True, transform=None,
    )

    two_view = TwoViewTransform(get_simclr_augmentation())
    dataset = TwoViewDataset(base_dataset, two_view_transform=two_view)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    view1_feats = []
    view2_feats = []
    count = 0

    with torch.no_grad():
        for v1, v2, _ in dataloader:
            if count >= NUM_SAMPLES:
                break
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            view1_feats.append(encoder(v1).cpu())
            view2_feats.append(encoder(v2).cpu())
            count += v1.size(0)

    f1 = torch.cat(view1_feats)[:NUM_SAMPLES]
    f2 = torch.cat(view2_feats)[:NUM_SAMPLES]

    f1_norm = F.normalize(f1, dim=1)
    f2_norm = F.normalize(f2, dim=1)
    sim_matrix = torch.mm(f1_norm, f2_norm.t()).numpy()

    same_image_sim = np.diag(sim_matrix)
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    diff_image_sim = sim_matrix[mask]

    stats = {
        "same_image_mean": float(np.mean(same_image_sim)),
        "same_image_std": float(np.std(same_image_sim)),
        "different_image_mean": float(np.mean(diff_image_sim)),
        "different_image_std": float(np.std(diff_image_sim)),
        "num_samples": NUM_SAMPLES,
    }

    print(f"\nSame image (two augmented views):")
    print(f"  Mean cosine similarity: {stats['same_image_mean']:.4f}")
    print(f"  Std:                    {stats['same_image_std']:.4f}")
    print(f"\nDifferent images:")
    print(f"  Mean cosine similarity: {stats['different_image_mean']:.4f}")
    print(f"  Std:                    {stats['different_image_std']:.4f}")

    with open("results/task3_similarity_before_training.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved: results/task3_similarity_before_training.json")
    print("\nTask 3 Complete!")


if __name__ == "__main__":
    main()
