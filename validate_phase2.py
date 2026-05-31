"""Validation script to check all tasks compile and basic functionality works."""

import sys
import torch

print("Checking Phase 2 Implementation...")
print("="*70)

# Test 1: Import models
print("\n✓ Test 1: Importing SimCLR models...")
try:
    from models import SimCLRModel, SimCLREncoder, SimCLRProjectionHead
    print("  - SimCLREncoder imported successfully")
    print("  - SimCLRProjectionHead imported successfully")
    print("  - SimCLRModel imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import models: {e}")
    sys.exit(1)

# Test 2: Create SimCLR model
print("\n✓ Test 2: Creating SimCLR model...")
try:
    model = SimCLRModel()
    print(f"  - Model created successfully")
    print(f"  - Encoder: {type(model.encoder).__name__}")
    print(f"  - Projection Head: {type(model.projection_head).__name__}")
except Exception as e:
    print(f"  ✗ Failed to create model: {e}")
    sys.exit(1)

# Test 3: Forward pass
print("\n✓ Test 3: Testing forward pass...")
try:
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        features, projected = model(x)
    print(f"  - Input shape: {x.shape}")
    print(f"  - Features shape: {features.shape} (expected: [2, 512])")
    print(f"  - Projected shape: {projected.shape} (expected: [2, 128])")
    assert features.shape == (2, 512), "Features shape mismatch"
    assert projected.shape == (2, 128), "Projected shape mismatch"
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 4: NT-Xent loss
print("\n✓ Test 4: Testing NT-Xent loss...")
try:
    from task4_simclr_implementation import nt_xent_loss
    z_i = torch.randn(4, 128)
    z_j = torch.randn(4, 128)
    loss = nt_xent_loss(z_i, z_j, temperature=0.5)
    print(f"  - Loss computed successfully: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
except Exception as e:
    print(f"  ✗ NT-Xent loss failed: {e}")
    sys.exit(1)

# Test 5: Cosine similarity
print("\n✓ Test 5: Testing cosine similarity...")
try:
    from task4_simclr_implementation import cosine_similarity
    z1 = torch.randn(5, 128)
    z2 = torch.randn(5, 128)
    sim = cosine_similarity(z1, z2)
    print(f"  - Similarity matrix shape: {sim.shape} (expected: [5, 5])")
    print(f"  - Similarity range: [{sim.min().item():.4f}, {sim.max().item():.4f}]")
    assert sim.shape == (5, 5), "Similarity shape mismatch"
except Exception as e:
    print(f"  ✗ Cosine similarity failed: {e}")
    sys.exit(1)

# Test 6: LinearProbe
print("\n✓ Test 6: Testing LinearProbe model...")
try:
    from task6_linear_probe import LinearProbe
    from models import create_resnet18_cifar10
    
    encoder = create_resnet18_cifar10()
    encoder.fc = torch.nn.Identity()
    
    probe = LinearProbe(encoder, num_classes=10, freeze_encoder=True)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        logits = probe(x)
    print(f"  - LinearProbe created successfully")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output logits shape: {logits.shape} (expected: [2, 10])")
    assert logits.shape == (2, 10), "Output shape mismatch"
except Exception as e:
    print(f"  ✗ LinearProbe failed: {e}")
    sys.exit(1)

# Test 7: File structure
print("\n✓ Test 7: Checking file structure...")
from pathlib import Path
files_to_check = [
    "task3_feature_similarity.py",
    "task4_simclr_implementation.py",
    "task5_simclr_pretraining.py",
    "task6_linear_probe.py",
    "models.py",
]
for fname in files_to_check:
    fpath = Path(fname)
    if fpath.exists():
        print(f"  ✓ {fname}")
    else:
        print(f"  ✗ {fname} NOT FOUND")
        sys.exit(1)

print("\n" + "="*70)
print("✓✓✓ All validation tests passed! ✓✓✓")
print("="*70)
print("\nPhase 2 (Day 6 Checkpoint) is ready to run:")
print("  1. python3 task3_feature_similarity.py       (8 marks)")
print("  2. python3 task4_simclr_implementation.py     (24 marks)")
print("  3. python3 task5_simclr_pretraining.py        (12 marks)")
print("  4. python3 task6_linear_probe.py               (10 marks)")
print("\nTotal: 54 marks")
