import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


def build_model(
    architecture: str = "efficientnet_b4",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    model = getattr(models, architecture)(
        weights="DEFAULT" if pretrained else None
    )

    if "efficientnet" in architecture:
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif "resnet" in architecture:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif "densenet" in architecture:
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model
