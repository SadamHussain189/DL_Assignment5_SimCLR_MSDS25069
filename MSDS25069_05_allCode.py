"""
All Code Combined - Assignment 5: SimCLR
Student: Sadam Hussain (MSDS25069)
Course: Deep Learning - Spring 2026

This file combines all task implementations into a single file.
Run individual tasks using: python MSDS25069_05_allCode.py --task <number>
Or run all tasks sequentially: python MSDS25069_05_allCode.py --task all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# ============================================================================
# Seed
# ============================================================================
import os
import random

SEED = 2026
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
TEMPERATURE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def set_seed(seed=2026, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Dataset Splits
# ============================================================================

def read_split_indices(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [int(line) for line in lines if line]


def get_cifar10_subset(data_root, split_file, train, transform=None, download=False):
    dataset = CIFAR10(root=str(data_root), train=train, transform=transform, download=download)
    indices = read_split_indices(split_file)
    return Subset(dataset, indices)


class TwoViewDataset(Dataset):
    def __init__(self, base_dataset, two_view_transform):
        self.base_dataset = base_dataset
        self.two_view_transform = two_view_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        view1, view2 = self.two_view_transform(image)
        return view1, view2, target


class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# ============================================================================
# Models
# ============================================================================

def create_resnet18_cifar10(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


class SupervisedModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = create_resnet18_cifar10(num_classes)

    def forward(self, x):
        return self.backbone(x)


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = create_resnet18_cifar10(num_classes=10)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class SimCLRModel(nn.Module):
    def __init__(self, encoder_dim=512, hidden_dim=256, proj_dim=128):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.projection_head = SimCLRProjectionHead(encoder_dim, hidden_dim, proj_dim)

    def forward(self, x):
        features = self.encoder(x)
        projected = self.projection_head(features)
        return features, projected


class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(512, num_classes)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        if any(p.requires_grad for p in self.encoder.parameters()):
            features = self.encoder(x)
        else:
            with torch.no_grad():
                features = self.encoder(x)
        return self.classifier(features)


# ============================================================================
# Shared Augmentations
# ============================================================================

def get_simclr_augmentation():
    return T.Compose([
        T.RandomResizedCrop(32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])


def get_train_test_transforms():
    train_t = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_t = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return train_t, test_t


# ============================================================================
# NT-Xent Loss
# ============================================================================

def nt_xent_loss(z_i, z_j, temperature=0.5):
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
# Task 1: Supervised Baseline
# ============================================================================

def run_task1():
    set_seed(SEED)
    print("\n" + "=" * 70)
    print("Task 1: Supervised Baseline (12 marks)")
    print("=" * 70)

    Path("graphs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_t = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    test_t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_ds = get_cifar10_subset("data", "splits/train_labeled_10percent.txt", True, train_t, download=True)
    val_ds = get_cifar10_subset("data", "splits/val.txt", True, test_t)
    test_ds = get_cifar10_subset("data", "splits/test.txt", False, test_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SupervisedModel(10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    best_val_acc, best_epoch = 0.0, 0

    for epoch in range(100):
        model.train()
        t_loss = 0.0
        for imgs, labs in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labs)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
        train_losses.append(t_loss / len(train_ds))

        model.eval()
        v_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                out = model(imgs)
                v_loss += criterion(out, labs).item() * imgs.size(0)
                correct += (out.argmax(1) == labs).sum().item()
                total += labs.size(0)
        val_losses.append(v_loss / total)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "models/supervised_model.pth")
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Acc: {val_acc:.4f}")

    model.load_state_dict(torch.load("models/supervised_model.pth", map_location=DEVICE))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs = imgs.to(DEVICE)
            preds.extend(model(imgs).argmax(1).cpu().numpy())
            targets.extend(labs.numpy())
    test_acc = np.mean(np.array(preds) == np.array(targets))
    print(f"Test Accuracy: {test_acc:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.title("Supervised Baseline Loss", fontweight="bold")
    plt.tight_layout(); plt.savefig("graphs/supervised_loss.png", dpi=150); plt.close()

    cm = confusion_matrix(targets, preds)
    cm_n = cm.astype(float) / cm.sum(axis=1)[:, None]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm_n, cmap=plt.cm.Blues)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                color="white" if cm_n[i,j] > cm_n.max()/2 else "black", fontsize=7)
    ax.set_title(f"Confusion Matrix (Acc: {test_acc:.4f})", fontweight="bold")
    plt.tight_layout(); plt.savefig("results/supervised_confusion_matrix.png", dpi=150); plt.close()

    with open("results/task1_supervised_results.json", "w") as f:
        json.dump({"test_accuracy": float(test_acc), "best_val_accuracy": float(best_val_acc)}, f, indent=2)
    print("Task 1 Complete!")
    return test_acc


# ============================================================================
# Task 2: Augmentation Visualization
# ============================================================================

def run_task2():
    set_seed(SEED)
    print("\n" + "=" * 70)
    print("Task 2: Augmentation Visualization (8 marks)")
    print("=" * 70)

    Path("results").mkdir(exist_ok=True)
    viz_t = T.Compose([
        T.RandomResizedCrop(32, scale=(0.2, 1.0)), T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
    ])
    two_view = TwoViewTransform(viz_t)
    dataset = CIFAR10(root="data", train=True, download=True, transform=None)
    np.random.seed(SEED)
    indices = np.random.choice(len(dataset), 10, replace=False)

    fig = plt.figure(figsize=(14, 15))
    for row, idx in enumerate(indices):
        orig, _ = dataset[idx]
        v1, v2 = two_view(orig)
        for col, (img, title) in enumerate([(orig, "Original"), (v1, "View 1"), (v2, "View 2")]):
            ax = plt.subplot(10, 3, row * 3 + col + 1)
            ax.imshow(np.array(img)); ax.axis("off")
            if row == 0:
                ax.set_title(title, fontweight="bold")
    fig.suptitle("SimCLR Augmentation Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("results/augmentation_examples.png", dpi=150, bbox_inches="tight"); plt.close()
    print("Saved: results/augmentation_examples.png\nTask 2 Complete!")


# ============================================================================
# Task 3: Feature Similarity Before Training
# ============================================================================

def run_task3():
    set_seed(SEED)
    print("\n" + "=" * 70)
    print("Task 3: Feature Similarity Before Training (8 marks)")
    print("=" * 70)

    Path("results").mkdir(exist_ok=True)
    encoder = ResNet18Encoder().to(DEVICE)
    encoder.eval()

    base_ds = get_cifar10_subset("data", "splits/train_ssl_unlabeled.txt", True, None)
    two_view = TwoViewTransform(get_simclr_augmentation())
    ds = TwoViewDataset(base_ds, two_view)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    f1s, f2s = [], []
    count = 0
    with torch.no_grad():
        for v1, v2, _ in dl:
            if count >= 100: break
            f1s.append(encoder(v1.to(DEVICE)).cpu())
            f2s.append(encoder(v2.to(DEVICE)).cpu())
            count += v1.size(0)

    f1 = F.normalize(torch.cat(f1s)[:100], dim=1)
    f2 = F.normalize(torch.cat(f2s)[:100], dim=1)
    sim = torch.mm(f1, f2.t()).numpy()

    same = np.diag(sim)
    diff = sim[~np.eye(sim.shape[0], dtype=bool)]
    stats = {"same_image_mean": float(np.mean(same)), "same_image_std": float(np.std(same)),
             "different_image_mean": float(np.mean(diff)), "different_image_std": float(np.std(diff))}
    print(f"Same image sim: {stats['same_image_mean']:.4f} | Diff image sim: {stats['different_image_mean']:.4f}")
    with open("results/task3_similarity_before_training.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Task 3 Complete!")
    return stats


# ============================================================================
# Task 4+5: SimCLR Implementation & Pretraining
# ============================================================================

def run_task4_5():
    set_seed(SEED)
    print("\n" + "=" * 70)
    print("Task 4+5: SimCLR Implementation & Pretraining (36 marks)")
    print("=" * 70)

    Path("graphs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    base_ds = get_cifar10_subset("data", "splits/train_ssl_unlabeled.txt", True, None)
    two_view = TwoViewTransform(get_simclr_augmentation())
    ds = TwoViewDataset(base_ds, two_view)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    # Similarity matrix BEFORE training
    model = SimCLRModel().to(DEVICE)
    model.eval()
    with torch.no_grad():
        v1, v2, _ = next(iter(dl))
        _, p1 = model(v1[:16].to(DEVICE))
        _, p2 = model(v2[:16].to(DEVICE))
        z_before = torch.cat([p1, p2], dim=0)
    z_norm = F.normalize(z_before, dim=1)
    sim_b = torch.mm(z_norm, z_norm.t()).cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(sim_b, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Similarity Matrix (Before Training)", fontweight="bold")
    plt.tight_layout(); plt.savefig("results/similarity_matrix_before_training.png", dpi=150); plt.close()

    # Pretraining
    set_seed(SEED)
    model = SimCLRModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []
    start = time.time()
    for epoch in range(1, 51):
        model.train()
        eloss, nb = 0.0, 0
        for v1, v2, _ in dl:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            _, z1 = model(v1); _, z2 = model(v2)
            loss = nt_xent_loss(z1, z2, TEMPERATURE)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            eloss += loss.item(); nb += 1
        losses.append(eloss / nb)
        elapsed = time.time() - start
        eta = (elapsed / epoch) * (50 - epoch)
        print(f"Epoch {epoch:3d}/50 | Loss: {losses[-1]:.4f} | ETA: {eta/60:.1f}min", flush=True)

    total_time = time.time() - start
    torch.save(model.encoder.state_dict(), "models/simclr_encoder.pt")
    torch.save(model.state_dict(), "models/simclr_pretrained.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 51), losses, marker="o", markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("NT-Xent Loss")
    plt.title("SimCLR Pretraining Loss", fontweight="bold"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("graphs/simclr_pretraining_loss.png", dpi=150); plt.close()

    # Post-training similarity
    model.eval()
    f1s, f2s = [], []
    count = 0
    with torch.no_grad():
        for v1, v2, _ in dl:
            if count >= 100: break
            _, p1 = model(v1.to(DEVICE)); _, p2 = model(v2.to(DEVICE))
            f1s.append(p1.cpu()); f2s.append(p2.cpu())
            count += v1.size(0)
    ff1 = F.normalize(torch.cat(f1s)[:100], dim=1)
    ff2 = F.normalize(torch.cat(f2s)[:100], dim=1)
    sim_a = torch.mm(ff1, ff2.t()).numpy()
    same_after = float(np.mean(np.diag(sim_a)))
    diff_after = float(np.mean(sim_a[~np.eye(100, dtype=bool)]))

    z_viz = torch.cat([ff1[:50], ff2[:50]], dim=0)
    z_n = F.normalize(z_viz, dim=1)
    sim_v = torch.mm(z_n, z_n.t()).cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(sim_v, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Similarity Matrix (After Training)", fontweight="bold")
    plt.tight_layout(); plt.savefig("results/similarity_matrix_after_training.png", dpi=150); plt.close()

    summary = {"epochs": 50, "initial_loss": losses[0], "final_loss": losses[-1],
               "same_view_similarity_after": same_after, "different_image_similarity_after": diff_after,
               "total_time_minutes": round(total_time / 60, 2)}
    with open("results/task5_simclr_pretraining_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Task 4+5 Complete!")
    return summary


# ============================================================================
# Task 6: Linear Probe
# ============================================================================

def run_task6():
    set_seed(SEED)
    print("\n" + "=" * 70)
    print("Task 6: Linear Probe (10 marks)")
    print("=" * 70)

    Path("graphs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    test_t = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    train_ds = get_cifar10_subset("data", "splits/train_labeled_10percent.txt", True, test_t)
    val_ds = get_cifar10_subset("data", "splits/val.txt", True, test_t)
    test_ds = get_cifar10_subset("data", "splits/test.txt", False, test_t)
    train_l = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_l = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_l = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    def probe_experiment(name, encoder):
        model = LinearClassifier(encoder, 10, freeze_encoder=True).to(DEVICE)
        opt = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
        crit = nn.CrossEntropyLoss()
        best_va, best_st, val_accs = 0, None, []
        for ep in range(1, 21):
            model.train()
            for imgs, labs in train_l:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                loss = crit(model(imgs), labs)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval(); c, t = 0, 0
            with torch.no_grad():
                for imgs, labs in val_l:
                    imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                    c += (model(imgs).argmax(1) == labs).sum().item(); t += labs.size(0)
            va = c / t; val_accs.append(va)
            if va > best_va: best_va = va; best_st = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_st); model.eval(); c, t = 0, 0
        with torch.no_grad():
            for imgs, labs in test_l:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                c += (model(imgs).argmax(1) == labs).sum().item(); t += labs.size(0)
        ta = c / t
        print(f"  {name}: Val={best_va:.4f} Test={ta:.4f}")
        return {"name": name, "test_accuracy": ta, "best_val_accuracy": best_va, "val_accuracies": val_accs}, model

    r1, _ = probe_experiment("Random frozen encoder", ResNet18Encoder().to(DEVICE))
    sm = SimCLRModel(); sm.load_state_dict(torch.load("models/simclr_pretrained.pth", map_location=DEVICE))
    r2, best_lp = probe_experiment("SimCLR frozen encoder", sm.encoder.to(DEVICE))
    torch.save(best_lp.state_dict(), "models/linear_probe.pt")

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in [r1, r2]: ax.plot(r["val_accuracies"], label=r["name"], marker="o", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Accuracy"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Linear Probe Accuracy", fontweight="bold")
    plt.tight_layout(); plt.savefig("graphs/linear_probe_accuracy.png", dpi=150); plt.close()

    with open("results/task6_linear_probe_results.json", "w") as f:
        json.dump({"experiments": [{"name": r["name"], "test_accuracy": r["test_accuracy"],
                    "best_val_accuracy": r["best_val_accuracy"]} for r in [r1, r2]]}, f, indent=2)
    print("Task 6 Complete!")
    return r1, r2


# ============================================================================
# Task 7+8: Fine-tuning + Visualization + Metrics
# ============================================================================

def run_task7_8(sup_acc=0.0, rand_lp=0.0, simclr_lp=0.0, sim_before=None, sim_after=None):
    set_seed(SEED)
    print("\n" + "=" * 70)
    print("Task 7+8: Fine-tuning & Visualization (13 marks)")
    print("=" * 70)

    Path("graphs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    train_t, test_t = get_train_test_transforms()
    train_ds = get_cifar10_subset("data", "splits/train_labeled_10percent.txt", True, train_t)
    val_ds = get_cifar10_subset("data", "splits/val.txt", True, test_t)
    test_ds = get_cifar10_subset("data", "splits/test.txt", False, test_t)
    tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    tel = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    def finetune_exp(name, model, epochs=20):
        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        best_va, best_st = 0, None
        for ep in range(1, epochs + 1):
            model.train()
            for imgs, labs in tqdm(tl, leave=False):
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                loss = crit(model(imgs), labs); opt.zero_grad(); loss.backward(); opt.step()
            model.eval(); c, t = 0, 0
            with torch.no_grad():
                for imgs, labs in vl:
                    imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                    c += (model(imgs).argmax(1) == labs).sum().item(); t += labs.size(0)
            va = c / t
            if va > best_va: best_va = va; best_st = {k: v.clone() for k, v in model.state_dict().items()}
            if ep % 5 == 0: print(f"  {name} Epoch {ep} Val: {va:.4f}")
        model.load_state_dict(best_st); model.eval(); c, t = 0, 0
        with torch.no_grad():
            for imgs, labs in tel:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                c += (model(imgs).argmax(1) == labs).sum().item(); t += labs.size(0)
        ta = c / t; print(f"  {name}: Test={ta:.4f}")
        return ta, model

    results_all = []

    # Exp 1-3
    set_seed(SEED); m1 = create_resnet18_cifar10(10).to(DEVICE)
    a1, _ = finetune_exp("Supervised scratch", m1)
    results_all.append(("Supervised scratch", a1))

    set_seed(SEED)
    m2 = LinearClassifier(ResNet18Encoder().to(DEVICE), 10, True).to(DEVICE)
    a2, _ = finetune_exp("Random frozen + linear", m2)
    results_all.append(("Random frozen + linear", a2))

    set_seed(SEED)
    sm = SimCLRModel(); sm.load_state_dict(torch.load("models/simclr_pretrained.pth", map_location=DEVICE))
    m3 = LinearClassifier(sm.encoder.to(DEVICE), 10, True).to(DEVICE)
    a3, _ = finetune_exp("SimCLR frozen + linear", m3)
    results_all.append(("SimCLR frozen + linear", a3))

    # Exp 4: Full fine-tuning
    set_seed(SEED)
    sm2 = SimCLRModel(); sm2.load_state_dict(torch.load("models/simclr_pretrained.pth", map_location=DEVICE))
    m4 = LinearClassifier(sm2.encoder.to(DEVICE), 10, False).to(DEVICE)
    a4, ft_model = finetune_exp("SimCLR fine-tuned", m4)
    results_all.append(("SimCLR fine-tuned", a4))

    torch.save(ft_model.state_dict(), "models/finetuned_model.pt")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r[0] for r in results_all]
    accs = [r[1] for r in results_all]
    bars = ax.bar(range(len(names)), accs, color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"], edgecolor="black")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Test Accuracy"); ax.set_title("Fine-tuning Comparison", fontweight="bold")
    ax.set_ylim([0, 1]); ax.grid(alpha=0.3, axis="y")
    for b, a in zip(bars, accs): ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{a:.4f}", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig("graphs/finetuning_accuracy.png", dpi=150); plt.close()

    # Task 8: PCA
    viz_t = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    viz_ds = get_cifar10_subset("data", "splits/val.txt", True, viz_t)
    np.random.seed(SEED)
    if len(viz_ds) > 1000: viz_ds = Subset(viz_ds, np.random.choice(len(viz_ds), 1000, replace=False))
    viz_l = DataLoader(viz_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    def get_feats(enc, loader, n=1000):
        enc.eval(); fs, ls = [], []; c = 0
        with torch.no_grad():
            for imgs, labs in loader:
                if c >= n: break
                fs.append(enc(imgs.to(DEVICE)).cpu().numpy()); ls.append(labs.numpy()); c += len(imgs)
        return np.vstack(fs)[:n], np.hstack(ls)[:n]

    def do_tsne_plot(feats, labs, title, path):
        coords = TSNE(n_components=2, random_state=SEED, perplexity=30,
                       max_iter=1000, init="pca", learning_rate="auto").fit_transform(feats)
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.tab10(np.linspace(0, 1, 10))
        for c in range(10):
            m = labs == c
            ax.scatter(coords[m, 0], coords[m, 1], c=[cmap[c]],
                       label=CIFAR10_CLASSES[c], alpha=0.6, s=30, edgecolors="none")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9); ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

    set_seed(SEED)
    fr, lr = get_feats(ResNet18Encoder().to(DEVICE), viz_l)
    do_tsne_plot(fr, lr, "Random Encoder Features (t-SNE)", "results/random_encoder_pca_or_tsne.png")

    sm3 = SimCLRModel(); sm3.load_state_dict(torch.load("models/simclr_pretrained.pth", map_location=DEVICE))
    fs, _ = get_feats(sm3.encoder.to(DEVICE), viz_l)
    do_tsne_plot(fs, lr, "SimCLR Encoder Features (t-SNE)", "results/simclr_encoder_pca_or_tsne.png")

    ff, _ = get_feats(ft_model.encoder, viz_l)
    do_tsne_plot(ff, lr, "Fine-tuned Encoder Features (t-SNE)", "results/finetuned_encoder_pca_or_tsne.png")

    # test_predictions.csv
    ft_model.eval()
    test_indices = read_split_indices("splits/test.txt")
    all_true, all_pred, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labs in tel:
            imgs = imgs.to(DEVICE)
            logits = ft_model(imgs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_true.extend(labs.numpy().tolist())
            all_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            all_probs.extend(probs.tolist())
    with open("results/test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_index", "true_label", "predicted_label"] + [f"prob_class_{i}" for i in range(10)])
        for i in range(len(all_true)):
            idx = test_indices[i] if i < len(test_indices) else i
            w.writerow([idx, all_true[i], all_pred[i]] + [f"{p:.6f}" for p in all_probs[i]])

    # metrics.json
    if sim_before is None: sim_before = {}
    if sim_after is None: sim_after = {}
    metrics = {
        "student_name": "Sadam Hussain", "roll_number": "MSDS25069",
        "seed": SEED, "batch_size": BATCH_SIZE, "simclr_epochs": 50,
        "linear_probe_epochs": 20, "finetuning_epochs": 20,
        "learning_rate": LEARNING_RATE, "temperature": TEMPERATURE,
        "supervised_10percent_test_acc": sup_acc,
        "random_linear_probe_test_acc": rand_lp,
        "simclr_linear_probe_test_acc": simclr_lp,
        "simclr_finetune_test_acc": a4,
        "same_view_similarity_before": sim_before.get("same_image_mean", 0.0),
        "different_image_similarity_before": sim_before.get("different_image_mean", 0.0),
        "same_view_similarity_after": sim_after.get("same_view_similarity_after", 0.0),
        "different_image_similarity_after": sim_after.get("different_image_similarity_after", 0.0),
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: results/metrics.json, results/test_predictions.csv")
    print("Task 7+8 Complete!")
    return results_all


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SimCLR Assignment - All Tasks")
    parser.add_argument("--task", type=str, default="all",
                        help="Task to run: 1, 2, 3, 4, 5, 6, 7, all")
    args = parser.parse_args()

    if args.task == "all":
        sup_acc = run_task1()
        run_task2()
        sim_before = run_task3()
        sim_after = run_task4_5()
        r1, r2 = run_task6()
        run_task7_8(sup_acc, r1["test_accuracy"], r2["test_accuracy"], sim_before, sim_after)
    elif args.task == "1":
        run_task1()
    elif args.task == "2":
        run_task2()
    elif args.task == "3":
        run_task3()
    elif args.task in ("4", "5"):
        run_task4_5()
    elif args.task == "6":
        run_task6()
    elif args.task == "7":
        run_task7_8()
    else:
        print(f"Unknown task: {args.task}")
        sys.exit(1)


if __name__ == "__main__":
    main()
