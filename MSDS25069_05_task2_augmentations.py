"""Task 2: Understanding Augmentations (8 Marks).

Visualize the SimCLR augmentation pipeline with TwoViewTransform.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10

from utils.seed import set_seed

import torchvision.transforms as T

SEED = 2026


class TwoViewTransform:
    """Apply augmentation pipeline twice to create two views."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


def main():
    set_seed(SEED)

    Path("results").mkdir(parents=True, exist_ok=True)

    # Augmentation pipeline (PIL-only, no ToTensor/Normalize for visualization)
    viz_transform = T.Compose([
        T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
    ])

    two_view = TwoViewTransform(viz_transform)

    print("Loading CIFAR-10...")
    dataset = CIFAR10(root="data", train=True, download=True, transform=None)

    np.random.seed(SEED)
    indices = np.random.choice(len(dataset), 10, replace=False)

    fig = plt.figure(figsize=(14, 15))

    for row, sample_idx in enumerate(indices):
        original_pil, label = dataset[sample_idx]
        view1_pil, view2_pil = two_view(original_pil)

        original = np.array(original_pil)
        view1 = np.array(view1_pil)
        view2 = np.array(view2_pil)

        ax = plt.subplot(10, 3, row * 3 + 1)
        ax.imshow(original)
        ax.axis("off")
        if row == 0:
            ax.set_title("Original Image", fontsize=11, fontweight="bold")

        ax = plt.subplot(10, 3, row * 3 + 2)
        ax.imshow(view1)
        ax.axis("off")
        if row == 0:
            ax.set_title("Augmented View 1", fontsize=11, fontweight="bold")

        ax = plt.subplot(10, 3, row * 3 + 3)
        ax.imshow(view2)
        ax.axis("off")
        if row == 0:
            ax.set_title("Augmented View 2", fontsize=11, fontweight="bold")

    fig.suptitle(
        "SimCLR Augmentation Pipeline\n"
        "Each row: Original Image | Augmented View 1 | Augmented View 2\n"
        "Augmentations: RandomResizedCrop, RandomHFlip, ColorJitter, RandomGrayscale",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig("results/augmentation_examples.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved: results/augmentation_examples.png")
    print("\nTask 2 Complete!")


if __name__ == "__main__":
    main()
