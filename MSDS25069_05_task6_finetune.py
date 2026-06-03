"""Task 7 & 8: Fine-tuning + PCA/t-SNE Visualization + Final Metrics.

Task 7 (8 Marks): Fine-tune SimCLR encoder end-to-end on 10% labeled data.
Task 8 (5 Marks): PCA/t-SNE visualization of features from 3 encoders.
Also generates: results/metrics.json, results/test_predictions.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models import SimCLRModel, ResNet18Encoder, create_resnet18_cifar10
from utils.dataset_splits import get_cifar10_subset
from utils.seed import set_seed

SEED = 2026
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


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


def get_transforms():
    train_t = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_t = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return train_t, test_t


def load_simclr_encoder():
    path = Path("models/simclr_pretrained.pth")
    if not path.exists():
        print(f"ERROR: {path} not found. Run MSDS25069_05_task4_simclr.py first.")
        return None
    m = SimCLRModel()
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    print(f"Loaded SimCLR encoder from {path}")
    return m.encoder


def train_and_evaluate(name, model, train_loader, val_loader, test_loader, epochs=EPOCHS):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    best_val_acc = 0.0
    best_state = None
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
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
        train_accs.append(correct / total)

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
            print(f"  Epoch {epoch:2d}/{epochs} | Train: {train_accs[-1]:.4f} | Val: {val_acc:.4f}")

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


# ============================================================================
# Task 8: PCA/t-SNE Visualization
# ============================================================================

def extract_features(encoder, dataloader, max_samples=1000):
    encoder.eval()
    feats, labs = [], []
    count = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(DEVICE)
            f = encoder(imgs).cpu().numpy()
            feats.append(f)
            labs.append(labels.numpy())
            count += len(imgs)
            if count >= max_samples:
                break
    return np.vstack(feats)[:max_samples], np.hstack(labs)[:max_samples]


def plot_features_2d(features, labels, title, out_path, method="tsne"):
    """Reduce 512-dim features to 2D and plot colored by class.

    Args:
        features: (N, 512) feature array
        labels: (N,) integer labels
        title: plot title
        out_path: file path to save
        method: "pca" or "tsne"
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=SEED, perplexity=30,
<<<<<<< HEAD
                       max_iter=1000, init="pca", learning_rate="auto")
=======
                       n_iter=1000, init="pca", learning_rate="auto")
>>>>>>> 1b6c423eef62d900697a458e482f9fd07715e606
        coords = reducer.fit_transform(features)
        axis_label = ("t-SNE 1", "t-SNE 2")
    else:
        reducer = PCA(n_components=2, random_state=SEED)
        coords = reducer.fit_transform(features)
        var = reducer.explained_variance_ratio_
        axis_label = (f"PC 1 ({var[0]:.1%})", f"PC 2 ({var[1]:.1%})")

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for c in range(10):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap[c]],
                   label=CIFAR10_CLASSES[c], alpha=0.6, s=30, edgecolors="none")
    ax.set_xlabel(axis_label[0], fontsize=12)
    ax.set_ylabel(axis_label[1], fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9,
              markerscale=1.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_test_predictions(model, test_loader, test_split_file):
    """Generate test_predictions.csv from the best fine-tuned model."""
    model.eval()
    indices = []
    with open(test_split_file) as f:
        for line in f:
            line = line.strip()
            if line:
                indices.append(int(line))

    all_true = []
    all_pred = []
    all_probs = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = Fn.softmax(logits, dim=1).cpu().numpy()
            _, pred = logits.max(1)
            all_true.extend(labels.numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    out_path = Path("results/test_predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["image_index", "true_label", "predicted_label"] + \
                 [f"prob_class_{i}" for i in range(10)]
        writer.writerow(header)
        for i in range(len(all_true)):
            idx = indices[i] if i < len(indices) else i
            row = [idx, all_true[i], all_pred[i]] + \
                  [f"{p:.6f}" for p in all_probs[i]]
            writer.writerow(row)
    print(f"Saved: {out_path}")


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

    # Load SimCLR encoder
    simclr_encoder = load_simclr_encoder()
    if simclr_encoder is None:
        return

    results_all = []

    # ==========================================================================
    # Task 7: Fine-tuning Experiments
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Task 7: Fine-tuning the SimCLR Encoder (8 marks)")
    print("=" * 70)

    # Exp 1: Supervised from scratch
    set_seed(SEED)
    model1 = create_resnet18_cifar10(num_classes=10).to(DEVICE)
    res1, _ = train_and_evaluate(
        "Supervised ResNet-18 from scratch (10% labels)",
        model1, train_loader, val_loader, test_loader,
    )
    results_all.append(res1)

    # Exp 2: Random frozen encoder + linear
    set_seed(SEED)
    enc_random = ResNet18Encoder().to(DEVICE)
    model2 = LinearClassifier(enc_random, freeze_encoder=True).to(DEVICE)
    res2, _ = train_and_evaluate(
        "Random frozen encoder + linear classifier",
        model2, train_loader, val_loader, test_loader,
    )
    results_all.append(res2)

    # Exp 3: SimCLR frozen encoder + linear
    set_seed(SEED)
    enc_simclr_frozen = load_simclr_encoder().to(DEVICE)
    model3 = LinearClassifier(enc_simclr_frozen, freeze_encoder=True).to(DEVICE)
    res3, _ = train_and_evaluate(
        "SimCLR frozen encoder + linear classifier",
        model3, train_loader, val_loader, test_loader,
    )
    results_all.append(res3)

    # Exp 4: SimCLR pretrained + full fine-tuning
    set_seed(SEED)
    enc_finetune = load_simclr_encoder().to(DEVICE)
    model4 = LinearClassifier(enc_finetune, freeze_encoder=False).to(DEVICE)
    res4, finetuned_model = train_and_evaluate(
        "SimCLR pretrained encoder + full fine-tuning",
        model4, train_loader, val_loader, test_loader,
    )
    results_all.append(res4)

    # Save fine-tuned model
    torch.save(finetuned_model.state_dict(), "models/finetuned_model.pt")
    print("Saved: models/finetuned_model.pt")

    # --- Fine-tuning comparison plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(results_all))
    test_accs = [r["test_accuracy"] for r in results_all]
    short_names = ["Supervised\n(scratch)", "Random\n(frozen)", "SimCLR\n(frozen)", "SimCLR\n(fine-tuned)"]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    bars = ax.bar(x_pos, test_accs, color=colors, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Fine-tuning Comparison: Test Accuracy", fontweight="bold")
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis="y")
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{acc:.4f}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig("graphs/finetuning_accuracy.png", dpi=150)
    plt.close(fig)
    print("Saved: graphs/finetuning_accuracy.png")

    # --- Save Task 7 results ---
    with open("results/task7_finetuning_results.json", "w") as f:
        json.dump({
            "task": "Task 7: Fine-tuning",
            "seed": SEED, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "experiments": [
                {"name": r["name"], "test_accuracy": r["test_accuracy"],
                 "best_val_accuracy": r["best_val_accuracy"]}
                for r in results_all
            ],
        }, f, indent=2)
    print("Saved: results/task7_finetuning_results.json")

    # ==========================================================================
    # Task 8: PCA/t-SNE Feature Visualization (5 marks)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Task 8: PCA/t-SNE Feature Visualization (5 marks)")
    print("=" * 70)

    viz_t = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    viz_ds = get_cifar10_subset("data", "splits/val.txt", train=True, transform=viz_t)

    np.random.seed(SEED)
    if len(viz_ds) > 1000:
        idxs = np.random.choice(len(viz_ds), 1000, replace=False)
        viz_ds = Subset(viz_ds, idxs)
    viz_loader = DataLoader(viz_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Extract features from all 3 encoders
    set_seed(SEED)
    enc_rand_viz = ResNet18Encoder().to(DEVICE)
    feats_rand, labels_viz = extract_features(enc_rand_viz, viz_loader, 1000)

    enc_simclr_viz = load_simclr_encoder().to(DEVICE)
    feats_simclr, _ = extract_features(enc_simclr_viz, viz_loader, 1000)

    finetuned_encoder = finetuned_model.encoder
    feats_ft, _ = extract_features(finetuned_encoder, viz_loader, 1000)

    # Generate t-SNE plots (better class separation visualization)
    print("\nGenerating t-SNE visualizations (1000 validation samples)...")
    plot_features_2d(feats_rand, labels_viz,
                     "Random Encoder Features (t-SNE)",
                     "results/random_encoder_pca_or_tsne.png", method="tsne")

    plot_features_2d(feats_simclr, labels_viz,
                     "SimCLR Encoder Features (t-SNE)",
                     "results/simclr_encoder_pca_or_tsne.png", method="tsne")

    plot_features_2d(feats_ft, labels_viz,
                     "Fine-tuned Encoder Features (t-SNE)",
                     "results/finetuned_encoder_pca_or_tsne.png", method="tsne")

    # ==========================================================================
    # Generate test_predictions.csv
    # ==========================================================================
    print("\nGenerating test_predictions.csv...")
    generate_test_predictions(finetuned_model, test_loader, "splits/test.txt")

    # ==========================================================================
    # Generate metrics.json
    # ==========================================================================
    print("\nGenerating metrics.json...")

    # Load pre/post similarity data
    sim_before = {}
    sim_before_path = Path("results/task3_similarity_before_training.json")
    if sim_before_path.exists():
        with open(sim_before_path) as f:
            sim_before = json.load(f)

    sim_after = {}
    sim_after_path = Path("results/task5_simclr_pretraining_results.json")
    if sim_after_path.exists():
        with open(sim_after_path) as f:
            sim_after = json.load(f)

    # Load linear probe results
    lp_results = {}
    lp_path = Path("results/task6_linear_probe_results.json")
    if lp_path.exists():
        with open(lp_path) as f:
            lp_results = json.load(f)

    random_lp_acc = 0.0
    simclr_lp_acc = 0.0
    if "experiments" in lp_results:
        for exp in lp_results["experiments"]:
            if "Random" in exp.get("name", ""):
                random_lp_acc = exp["test_accuracy"]
            elif "SimCLR" in exp.get("name", ""):
                simclr_lp_acc = exp["test_accuracy"]

    metrics = {
        "student_name": "Sadam Hussain",
        "roll_number": "MSDS25069",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "simclr_epochs": 50,
        "linear_probe_epochs": 20,
        "finetuning_epochs": 20,
        "learning_rate": LEARNING_RATE,
        "temperature": 0.5,
        "supervised_10percent_test_acc": res1["test_accuracy"],
        "random_linear_probe_test_acc": random_lp_acc,
        "simclr_linear_probe_test_acc": simclr_lp_acc,
        "simclr_finetune_test_acc": res4["test_accuracy"],
        "same_view_similarity_before": sim_before.get("same_image_mean", 0.0),
        "different_image_similarity_before": sim_before.get("different_image_mean", 0.0),
        "same_view_similarity_after": sim_after.get("same_view_similarity_after", 0.0),
        "different_image_similarity_after": sim_after.get("different_image_similarity_after", 0.0),
    }

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: results/metrics.json")

    # ==========================================================================
    # Final Summary Table
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<50} {'Test Acc'}")
    print("-" * 60)
    for r in results_all:
        print(f"{r['name']:<50} {r['test_accuracy']:.4f}")
    print(f"\nRandom Linear Probe:  {random_lp_acc:.4f}")
    print(f"SimCLR Linear Probe:  {simclr_lp_acc:.4f}")

    print("\nAll Tasks Complete!")


if __name__ == "__main__":
    main()
