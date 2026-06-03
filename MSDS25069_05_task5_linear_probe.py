"""Task 6: Linear Probe Evaluation (10 Marks).

Experiment A: Random frozen encoder + Linear(512->10)
Experiment B: SimCLR pretrained frozen encoder + Linear(512->10)
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SimCLRModel, ResNet18Encoder
from utils.dataset_splits import get_cifar10_subset
from utils.seed import set_seed

SEED = 2026
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearProbe(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(512, num_classes)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)


def get_transforms():
    t = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    return t, t


def run_experiment(name, encoder, train_loader, val_loader, test_loader):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    model = LinearProbe(encoder, num_classes=10).to(DEVICE)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    train_accs, val_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct, total = 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = out.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        train_accs.append(train_acc)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                _, pred = out.max(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            _, pred = out.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total

    print(f"  Best Val: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")

    return {
        "name": name,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "train_accuracies": train_accs,
        "val_accuracies": val_accs,
    }, model


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    Path("graphs").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    train_t, test_t = get_transforms()

    train_ds = get_cifar10_subset("data", "splits/train_labeled_10percent.txt", train=True, transform=train_t)
    val_ds = get_cifar10_subset("data", "splits/val.txt", train=True, transform=test_t)
    test_ds = get_cifar10_subset("data", "splits/test.txt", train=False, transform=test_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    results = []

    # --- Experiment A: Random Encoder ---
    random_encoder = ResNet18Encoder().to(DEVICE)
    res_random, _ = run_experiment(
        "Random frozen encoder + linear classifier",
        random_encoder, train_loader, val_loader, test_loader,
    )
    results.append(res_random)

    # --- Experiment B: SimCLR Encoder ---
    model_path = Path("models/simclr_pretrained.pth")
    if not model_path.exists():
        print(f"\nERROR: {model_path} not found. Run MSDS25069_05_task4_simclr.py first.")
        return

    simclr_model = SimCLRModel()
    simclr_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    simclr_encoder = simclr_model.encoder.to(DEVICE)

    res_simclr, best_probe_model = run_experiment(
        "SimCLR frozen encoder + linear classifier",
        simclr_encoder, train_loader, val_loader, test_loader,
    )
    results.append(res_simclr)

    # Save linear probe model
    torch.save(best_probe_model.state_dict(), "models/linear_probe.pt")
    print("\nSaved: models/linear_probe.pt")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("LINEAR PROBE RESULTS")
    print(f"{'='*60}")
    print(f"{'Encoder':<45} {'Test Acc'}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<45} {r['test_accuracy']:.4f}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for r in results:
        ax1.plot(r["val_accuracies"], marker="o", markersize=4, label=r["name"], linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_title("Linear Probe: Validation Accuracy", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    names = [r["name"] for r in results]
    accs = [r["test_accuracy"] for r in results]
    colors = ["#FF6B6B", "#4ECDC4"]
    bars = ax2.bar(range(len(names)), accs, color=colors, edgecolor="black")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(["Random\nEncoder", "SimCLR\nEncoder"])
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Linear Probe: Test Accuracy", fontweight="bold")
    ax2.set_ylim([0, 1.0])
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{acc:.4f}", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    fig.savefig("graphs/linear_probe_accuracy.png", dpi=150)
    plt.close(fig)
    print("Saved: graphs/linear_probe_accuracy.png")

    # Save results JSON
    with open("results/task6_linear_probe_results.json", "w") as f:
        json.dump({
            "task": "Task 6: Linear Probe",
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "experiments": [
                {"name": r["name"], "test_accuracy": r["test_accuracy"],
                 "best_val_accuracy": r["best_val_accuracy"]}
                for r in results
            ],
        }, f, indent=2)
    print("Saved: results/task6_linear_probe_results.json")
    print("\nTask 6 (Linear Probe) Complete!")


if __name__ == "__main__":
    main()
