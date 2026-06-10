from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config.settings import Settings


class Trainer:
    """Trainer skeleton to orchestrate training, logging, and evaluation steps."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[tuple[torch.Tensor, int]],
        val_loader: DataLoader[tuple[torch.Tensor, int]],
        settings: Settings,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.settings = settings
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=settings.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self) -> float:
        """Runs one epoch of training and returns average loss."""
        return 0.0

    def validate(self) -> float:
        """Evaluates the model on the validation set and returns accuracy."""
        return 0.0

    def fit(self) -> dict[str, list[float]]:
        """Orchestrates the training process for multiple epochs."""
        return {"loss": [], "accuracy": []}
