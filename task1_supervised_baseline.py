"""Task 1: Supervised Baseline with Limited Labels - 12 Marks

Train a ResNet-18 classifier from scratch using only the fixed 10% labeled training split.
This baseline will show how well a normal supervised model performs when labels are limited.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cudnn.enabled = False  # GTX 1060 sm_61 cuDNN incompatibility
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SupervisedModel
from utils.dataset_splits import get_cifar10_subset
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Task 1: Supervised Baseline")
    parser.add_argument("--data_root", type=str, default="data", help="Path to CIFAR-10 data")
    parser.add_argument("--splits_dir", type=str, default="splits", help="Path to split files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for results")
    return parser.parse_args()


def get_transforms():
    """Get train and test transforms for CIFAR-10."""
    # CIFAR-10 normalization stats
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    
    return train_transform, test_transform


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    return total_loss / len(train_loader.dataset)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Validate the model. Returns loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate on test set. Returns accuracy, true labels, and predictions."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return accuracy, np.array(all_targets), np.array(all_preds)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir) / "graphs"
    Path(args.output_dir) / "results"
    (Path(args.output_dir) / "graphs").mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "results").mkdir(parents=True, exist_ok=True)
    
    # Get transforms
    train_transform, test_transform = get_transforms()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = get_cifar10_subset(
        data_root=args.data_root,
        split_file=f"{args.splits_dir}/train_labeled_10percent.txt",
        train=True,
        transform=train_transform,
        download=True,
    )
    
    val_dataset = get_cifar10_subset(
        data_root=args.data_root,
        split_file=f"{args.splits_dir}/val.txt",
        train=True,
        transform=test_transform,
        download=False,
    )
    
    test_dataset = get_cifar10_subset(
        data_root=args.data_root,
        split_file=f"{args.splits_dir}/test.txt",
        train=False,
        transform=test_transform,
        download=False,
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = SupervisedModel(num_classes=10).to(device)
    print("Model created: ResNet-18 (modified for CIFAR-10)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0
    best_epoch = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), f"{args.output_dir}/best_supervised_model.pth")
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.4f}")
    
    print(f"\nBest validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")
    
    # Load best model
    model.load_state_dict(torch.load(f"{args.output_dir}/best_supervised_model.pth"))
    
    # Evaluate on test set
    test_accuracy, test_targets, test_preds = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(test_targets, test_preds)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Supervised Baseline: Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot_path = Path(args.output_dir) / "graphs" / "supervised_loss.png"
    plt.savefig(loss_plot_path, dpi=150)
    print(f"\nLoss plot saved: {loss_plot_path}")
    plt.close()
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalizer for better visualization
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    
    classes = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(
            j, i,
            f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})",
            ha="center",
            va="center",
            color="white" if cm_normalized[i, j] > thresh else "black",
            fontsize=8,
        )
    
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(f"Test Confusion Matrix (Accuracy: {test_accuracy:.4f})", 
                 fontsize=14, fontweight="bold")
    
    plt.colorbar(im, ax=ax, label="Normalized Count")
    plt.tight_layout()
    
    cm_plot_path = Path(args.output_dir) / "results" / "supervised_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=150)
    print(f"Confusion matrix saved: {cm_plot_path}")
    plt.close()
    
    # Save results summary
    results_summary = f"""Task 1: Supervised Baseline Results
=====================================
Training Configuration:
- Dataset: CIFAR-10 (10% labeled split)
- Model: ResNet-18 (modified for CIFAR-10)
- Optimizer: Adam
- Learning Rate: {args.lr}
- Batch Size: {args.batch_size}
- Number of Epochs: {args.epochs}
- Random Seed: {args.seed}

Training Summary:
- Train Samples: {len(train_dataset)}
- Validation Samples: {len(val_dataset)}
- Test Samples: {len(test_dataset)}
- Best Validation Accuracy: {best_val_accuracy:.4f} (at epoch {best_epoch})

Test Results:
- Test Accuracy: {test_accuracy:.4f}
- Confusion Matrix saved to: {cm_plot_path}
- Loss curves saved to: {loss_plot_path}
"""
    
    summary_path = Path(args.output_dir) / "results" / "supervised_baseline_summary.txt"
    summary_path.write_text(results_summary)
    print(f"\nResults summary saved: {summary_path}")
    print("\n" + "="*50)
    print(results_summary)


if __name__ == "__main__":
    main()
