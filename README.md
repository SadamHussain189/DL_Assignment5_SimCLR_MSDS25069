# DL_Assignment5_SimCLR_MSDS25069

**Deep Learning - Spring 2026 | Assignment 5: SimCLR**

**Student:** Sadam Hussain (MSDS25069)
**Random Seed:** 2026

---

## Setup

```bash
pip install -r requirements.txt
```

CIFAR-10 will be auto-downloaded on first run.

---

## How to Run (execute in order)

```bash
# Task 1: Supervised Baseline (10% labels)
python3 MSDS25069_05_task1_supervised.py

# Task 2: Augmentation Visualization
python3 MSDS25069_05_task2_augmentations.py

# Task 3: Feature Similarity Before Training
python3 MSDS25069_05_task3_similarity.py

# Task 4+5: SimCLR Implementation & Pretraining (50 epochs)
python3 MSDS25069_05_task4_simclr.py

# Task 6: Linear Probe Evaluation
python3 MSDS25069_05_task5_linear_probe.py

# Task 7+8: Fine-tuning + t-SNE Visualization + metrics.json + test_predictions.csv
python3 MSDS25069_05_task6_finetune.py
```

Or run everything via the combined file:

```bash
python MSDS25069_05_allCode.py --task all
```

---

## File Structure

```
MSDS25069_05_task1_supervised.py     # Task 1: Supervised baseline
MSDS25069_05_task2_augmentations.py  # Task 2: Augmentation pipeline
MSDS25069_05_task3_similarity.py     # Task 3: Feature similarity (before training)
MSDS25069_05_task4_simclr.py         # Task 4+5: SimCLR implementation & pretraining
MSDS25069_05_task5_linear_probe.py   # Task 6: Linear probe evaluation
MSDS25069_05_task6_finetune.py       # Task 7+8: Fine-tuning, t-SNE viz, metrics
MSDS25069_05_allCode.py              # All code combined in one file
models.py                           # Model architectures (ResNet-18, SimCLR)
requirements.txt                    # Python dependencies

utils/
  dataset_splits.py                 # CIFAR-10 subset loading
  seed.py                           # Random seed utility
  metrics.py                        # Accuracy, confusion matrix helpers
  visualization.py                  # Augmentation grid, PCA plots

splits/
  train_ssl_unlabeled.txt           # 45,000 unlabeled indices
  train_labeled_10percent.txt       # 5,000 labeled indices
  val.txt                           # 5,000 validation indices
  test.txt                          # 10,000 test indices

models/                             # Saved model checkpoints
  supervised_model.pth              # Task 1 best model
  simclr_encoder.pt                 # Task 5 encoder only
  simclr_pretrained.pth             # Task 5 full SimCLR model
  linear_probe.pt                   # Task 6 best linear probe
  finetuned_model.pt                # Task 7 fine-tuned model

graphs/
  supervised_loss.png               # Task 1 loss curves
  simclr_pretraining_loss.png       # Task 5 loss curve
  linear_probe_accuracy.png         # Task 6 accuracy comparison
  finetuning_accuracy.png           # Task 7 accuracy comparison

results/
  augmentation_examples.png         # Task 2 visualization
  similarity_matrix_before_training.png  # Task 4 similarity heatmap
  similarity_matrix_after_training.png   # Task 5 similarity heatmap
  supervised_confusion_matrix.png   # Task 1 confusion matrix
  random_encoder_pca_or_tsne.png    # Task 8 random encoder t-SNE
  simclr_encoder_pca_or_tsne.png    # Task 8 SimCLR encoder t-SNE
  finetuned_encoder_pca_or_tsne.png # Task 8 fine-tuned encoder t-SNE
  metrics.json                      # Final metrics (all accuracies)
  test_predictions.csv              # Test set predictions with probabilities
```

---

## Training Settings

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10 |
| Encoder | ResNet-18 (modified for CIFAR-10) |
| Batch size | 64 |
| SimCLR epochs | 50 |
| Linear probe epochs | 20 |
| Fine-tuning epochs | 20 |
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Temperature | 0.5 |
| Random seed | 2026 |
