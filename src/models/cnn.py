from __future__ import annotations

import torch
import torch.nn as nn
from src.config.settings import Settings


class AdaptiveCNN(nn.Module):
    """Skeleton for an adaptive CNN that can modify input channels dynamically."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.backbone = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stub output layer/predictions
        return torch.zeros((x.size(0), self.settings.num_classes))


def get_model(settings: Settings) -> nn.Module:
    """Factory function to instantiate models (LeNet, AlexNet, ResNet)."""
    return AdaptiveCNN(settings)
