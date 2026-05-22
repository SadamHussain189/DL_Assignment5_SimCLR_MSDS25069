"""Model architectures for Assignment 5."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


def create_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """Create a ResNet-18 modified for CIFAR-10.
    
    Modifications:
    - conv1: 3x3 convolution, stride 1, padding 1 (instead of 7x7, stride 2)
    - Remove max pooling layer
    - Output: 512-dimensional feature vector
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10).
        
    Returns:
        Modified ResNet-18 model.
    """
    model = models.resnet18(weights=None)
    
    # Modify first convolution layer for CIFAR-10
    # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove max pooling layer
    model.maxpool = nn.Identity()
    
    # Replace the final fully connected layer with number of classes
    model.fc = nn.Linear(512, num_classes)
    
    return model


class SupervisedModel(nn.Module):
    """Supervised ResNet-18 for CIFAR-10 classification."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = create_resnet18_cifar10(num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32).
            
        Returns:
            Logits of shape (batch_size, num_classes).
        """
        return self.backbone(x)
