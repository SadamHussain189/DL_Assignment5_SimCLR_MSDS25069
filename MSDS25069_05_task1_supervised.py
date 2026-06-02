"""Task 1: Supervised Baseline with Limited Labels (12 Marks).

Train a ResNet-18 classifier from scratch using only the fixed 10% labeled training split.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SupervisedModel
from utils.dataset_splits import get_cifar10_subset
from utils.seed import set_seed

# ============================================================================
# Configuration
# ============================================================================
SEED = 2026
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms():
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


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    targets = np.array(all_targets)
    preds = np.array(all_preds)
    accuracy = (targets == preds).mean()
    return accuracy, targets, preds


def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    Path("graphs").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    train_transform, test_transform = get_transforms()

    print("Loading datasets...")
    train_dataset = get_cifar10_subset(
        data_root="data",
        split_file="splits/train_labeled_10percent.txt",
        train=True, transform=train_transform, download=True,
    )
    val_dataset = get_cifar10_subset(
        data_root="data",
        split_file="splits/val.txt",
        train=True, transform=test_transform,
    )
    test_dataset = get_cifar10_subset(
        data_root="data",
        split_file="splits/test.txt",
        train=False, transform=test_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = SupervisedModel(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0
    best_epoch = 0

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_accuracy = validate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), "models/supervised_model.pth")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch+1}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load("models/supervised_model.pth", map_location=DEVICE))
    test_accuracy, test_targets, test_preds = evaluate(model, test_loader, DEVICE)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # --- Plot: Training & Validation Loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Supervised Baseline: Training and Validation Loss", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graphs/supervised_loss.png", dpi=150)
    plt.close()
    print("Saved: graphs/supervised_loss.png")

    # --- Plot: Confusion Matrix ---
    cm = confusion_matrix(test_targets, test_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    tick_marks = np.arange(len(CIFAR10_CLASSES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    thresh = cm_norm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black", fontsize=8)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(f"Test Confusion Matrix (Accuracy: {test_accuracy:.4f})", fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("results/supervised_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: results/supervised_confusion_matrix.png")

    # --- Save result summary as JSON ---
    summary = {
        "task": "Task 1: Supervised Baseline",
        "test_accuracy": float(test_accuracy),
        "best_val_accuracy": float(best_val_accuracy),
        "best_epoch": best_epoch + 1,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
    }
    with open("results/task1_supervised_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved: results/task1_supervised_results.json")
    print(f"\nTask 1 Complete! Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
