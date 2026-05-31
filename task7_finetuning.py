"""
Task 7: Fine-tuning the SimCLR Encoder - 8 Marks
=====================================================
Implement end-to-end fine-tuning pipeline comparing:
1. Supervised ResNet-18 from scratch (10% labels)
2. Random frozen encoder + linear classifier
3. SimCLR frozen encoder + linear classifier
4. SimCLR pretrained encoder + full fine-tuning

Author: Sadam Hussain (MSDS25069)
Date: May 31, 2026
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from models import SimCLRModel, ResNet18Encoder, create_resnet18_cifar10
from utils.dataset_splits import get_cifar10_subset

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device("cpu")
SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 20
WEIGHT_DECAY = 5e-4

DATA_ROOT = './data'
TRAIN_LABELED_SPLIT = './splits/train_labeled_10percent.txt'
VAL_SPLIT = './splits/val.txt'
TEST_SPLIT = './splits/test.txt'

# ============================================================================
# LINEAR CLASSIFIER
# ============================================================================
class LinearClassifier(nn.Module):
    """Linear classifier on top of encoder (with optional freezing)."""
    
    def __init__(self, encoder: nn.Module, num_classes: int = 10, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(512, num_classes)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if any(p.requires_grad for p in self.encoder.parameters()):
            features = self.encoder(x)
        else:
            with torch.no_grad():
                features = self.encoder(x)
        return self.classifier(features)

# ============================================================================
# MODEL LOADING
# ============================================================================
def get_simclr_pretrained():
    """Load SimCLR pretrained model"""
    if not os.path.exists('models/simclr_pretrained.pth'):
        print("ERROR: models/simclr_pretrained.pth not found!")
        return None
    
    model = SimCLRModel().to(DEVICE)
    checkpoint = torch.load('models/simclr_pretrained.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print("✓ Loaded SimCLR pretrained model")
    return model.encoder

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def test(model, dataloader, device):
    """Test model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================
def run_experiment(name, model, train_loader, val_loader, test_loader, epochs=EPOCHS):
    """Run one experiment"""
    print(f"\n{'='*70}")
    print(f"Experiment: {name}")
    print(f"{'='*70}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_val_acc = 0.0
    best_model_state = None
    
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Test
    test_acc = test(model, test_loader, DEVICE)
    
    print(f"\nResults:")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Test Acc:     {test_acc:.2f}%")
    
    return {
        'name': name,
        'test_accuracy': test_acc / 100,
        'best_val_accuracy': best_val_acc / 100,
        'final_train_accuracy': train_accuracies[-1] / 100,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*70)
    print("Task 7: Fine-tuning the SimCLR Encoder")
    print("="*70)
    
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2470, 0.2435, 0.2616))
    ])
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = get_cifar10_subset(DATA_ROOT, TRAIN_LABELED_SPLIT, train=True, transform=transform_train)
    val_dataset = get_cifar10_subset(DATA_ROOT, VAL_SPLIT, train=True, transform=transform_test)
    test_dataset = get_cifar10_subset(DATA_ROOT, TEST_SPLIT, train=False, transform=transform_test)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Load pretrained encoder
    encoder_pretrained = get_simclr_pretrained()
    if encoder_pretrained is None:
        return
    
    results = []
    
    # ========================================================================
    # Experiment 1: Supervised ResNet-18 from scratch
    # ========================================================================
    print("\nExperiment 1: Supervised ResNet-18 from scratch")
    model1 = create_resnet18_cifar10(num_classes=10).to(DEVICE)
    result1 = run_experiment(
        "Supervised ResNet-18 (10% labels)",
        model1,
        train_loader,
        val_loader,
        test_loader,
        epochs=EPOCHS
    )
    results.append(result1)
    
    # ========================================================================
    # Experiment 2: Random frozen encoder + linear classifier
    # ========================================================================
    print("\nExperiment 2: Random frozen encoder")
    encoder_random = ResNet18Encoder().to(DEVICE)
    model2 = LinearClassifier(encoder_random, num_classes=10, freeze_encoder=True).to(DEVICE)
    result2 = run_experiment(
        "Random frozen encoder + linear classifier",
        model2,
        train_loader,
        val_loader,
        test_loader,
        epochs=EPOCHS
    )
    results.append(result2)
    
    # ========================================================================
    # Experiment 3: SimCLR frozen encoder + linear classifier
    # ========================================================================
    print("\nExperiment 3: SimCLR frozen encoder")
    encoder_simclr = get_simclr_pretrained()
    model3 = LinearClassifier(encoder_simclr, num_classes=10, freeze_encoder=True).to(DEVICE)
    result3 = run_experiment(
        "SimCLR frozen encoder + linear classifier",
        model3,
        train_loader,
        val_loader,
        test_loader,
        epochs=EPOCHS
    )
    results.append(result3)
    
    # ========================================================================
    # Experiment 4: SimCLR pretrained encoder + full fine-tuning
    # ========================================================================
    print("\nExperiment 4: SimCLR fine-tuning")
    encoder_finetuned = get_simclr_pretrained()
    model4 = LinearClassifier(encoder_finetuned, num_classes=10, freeze_encoder=False).to(DEVICE)
    result4 = run_experiment(
        "SimCLR pretrained encoder + fine-tuning",
        model4,
        train_loader,
        val_loader,
        test_loader,
        epochs=EPOCHS
    )
    results.append(result4)
    
    # Save fine-tuned model
    torch.save(model4.state_dict(), 'models/finetuned_model.pth')
    print("✓ Saved fine-tuned model to models/finetuned_model.pth")
    
    # ========================================================================
    # GENERATE RESULTS TABLE
    # ========================================================================
    print("\n" + "="*70)
    print("FINE-TUNING RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Model':<50} {'Test Accuracy'}")
    print("-" * 70)
    for result in results:
        print(f"{result['name']:<50} {result['test_accuracy']:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_json = {
        'task': 'Task 7: Fine-tuning',
        'seed': SEED,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'experiments': results
    }
    
    with open('results/task7_finetuning_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n✓ Saved results to results/task7_finetuning_results.json")
    
    # ========================================================================
    # GENERATE COMPARISON PLOT
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(results))
    test_accs = [r['test_accuracy'] * 100 for r in results]
    val_accs = [r['best_val_accuracy'] * 100 for r in results]
    
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, val_accs, width, label='Best Val Accuracy', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Task 7: Fine-tuning Comparison - Test Accuracy across Methods', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r['name'][:35] + '...' if len(r['name']) > 35 else r['name'] 
                         for r in results], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    os.makedirs('graphs', exist_ok=True)
    plt.savefig('graphs/finetuning_accuracy.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved finetuning plot to graphs/finetuning_accuracy.png")
    plt.close()
    
    print("\n✅ Task 7: Fine-tuning Complete!")
    return results

if __name__ == "__main__":
    results = main()
