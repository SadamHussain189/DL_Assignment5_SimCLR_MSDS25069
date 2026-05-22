"""Task 2: Augmentations - Fast Version"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

# Load dataset
print("Loading CIFAR-10...")
dataset = CIFAR10(root="data", train=True, download=False, transform=None)

# Augmentation pipeline
transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
])

# Convert PIL to numpy for visualization
def to_numpy(img):
    return np.array(img)

# Create figure
print("Creating augmentation visualization...")
fig = plt.figure(figsize=(14, 15))

# Select 10 random samples
np.random.seed(2026)
indices = np.random.choice(len(dataset), 10, replace=False)

for idx, sample_idx in enumerate(indices):
    # Get original image
    original_pil, label = dataset[sample_idx]
    original = to_numpy(original_pil)
    
    # Generate two augmented views
    view1 = np.array(transform(original_pil))
    view2 = np.array(transform(original_pil))
    
    # Plot original
    ax = plt.subplot(10, 3, idx * 3 + 1)
    ax.imshow(original)
    ax.axis('off')
    if idx == 0:
        ax.set_title("Original Image", fontsize=11, fontweight='bold')
    ax.set_ylabel(f"Sample {idx+1}", fontsize=10, fontweight='bold')
    
    # Plot view 1
    ax = plt.subplot(10, 3, idx * 3 + 2)
    ax.imshow(view1)
    ax.axis('off')
    if idx == 0:
        ax.set_title("Augmented View 1", fontsize=11, fontweight='bold')
    
    # Plot view 2
    ax = plt.subplot(10, 3, idx * 3 + 3)
    ax.imshow(view2)
    ax.axis('off')
    if idx == 0:
        ax.set_title("Augmented View 2", fontsize=11, fontweight='bold')

fig.suptitle("SimCLR Augmentation Pipeline\n" + 
             "Each row shows: Original Image | Augmented View 1 | Augmented View 2\n" +
             "Augmentations: RandomResizedCrop, RandomHFlip, ColorJitter, RandomGrayscale",
            fontsize=13, fontweight='bold')

plt.tight_layout(rect=[0, 0.01, 1, 0.97])

# Save
Path("results").mkdir(exist_ok=True)
output_path = "results/augmentation_examples.png"
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# Save documentation
augmentation_doc = """Task 2: Understanding Augmentations (8 Marks)
============================================

AUGMENTATION PIPELINE FOR SimCLR:

The following augmentations are applied independently to create two different views
of the same image. This is KEY to SimCLR - the model learns that these different views
are "positive pairs" without using any labels!

Augmentations Applied:
----------------------
1. RandomResizedCrop(32, scale=(0.2, 1.0))
   - Randomly crop and resize image to different scales
   - Scale: between 20% and 100% of original

2. RandomHorizontalFlip(p=0.5)
   - 50% chance to flip image horizontally

3. ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
   - brightness: ±40%
   - contrast: ±40%
   - saturation: ±40%
   - hue: ±10%

4. RandomGrayscale(p=0.2)
   - 20% chance to convert to grayscale

TwoViewTransform Implementation:
-------------------------------
class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

Why This Works:
---------------
- Same image, same pipeline → different results due to randomness
- Both views contain the SAME semantic information
- But they LOOK different due to augmentations
- SimCLR learns: "Different looking images with same semantic content = positive pair"

This enables self-supervised learning! The model doesn't need labels to learn
useful representations.

Deliverable:
------------
✓ results/augmentation_examples.png
  - 10 diverse CIFAR-10 images
  - Each row: [Original | Aug View 1 | Aug View 2]
  - Shows the augmentation diversity and effectiveness
"""

Path("results/augmentation_pipeline_doc.txt").write_text(augmentation_doc)
print(f"✓ Saved: results/augmentation_pipeline_doc.txt")

print("\n" + "="*60)
print(augmentation_doc)
print("="*60)
print("\n✅ Task 2 Complete!")
