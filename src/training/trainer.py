from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for CNN models on TILDA.

    Uses mini-batch SGD with momentum (Q2.3) and a *ReduceLROnPlateau*
    scheduler that divides the learning rate by ``settings.scheduler_factor``
    when validation loss plateaus — matching the strategy described in both the
    AlexNet paper (§5: "divide the learning rate by 10 when the validation
    error rate stopped improving") and the ResNet paper (§3.4: "divided by 10
    when the error plateaus").

    Implements early stopping with ``settings.patience`` epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[tuple[object, object]],
        val_loader: DataLoader[tuple[object, object]],
        settings: Settings,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.settings = settings
        self.device = settings.resolve_device()

        # Optimizer: mini-batch SGD (Q2.3)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=settings.learning_rate,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=settings.scheduler_factor,
            patience=settings.scheduler_patience,
            verbose=False,
        )

        # History
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------
    def train_epoch(self) -> tuple[float, float]:
        """Run one epoch of training.  Returns ``(avg_loss, accuracy)``."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """Evaluate on the validation set.  Returns ``(avg_loss, accuracy)``."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def fit(self) -> dict[str, list[float] | float]:
        """Orchestrate training for ``settings.epochs`` epochs.

        Returns the full history dict plus a ``training_time`` key (seconds).
        """
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        checkpoint_path = results_dir / f"{self.settings.model_name}_best.pt"

        best_val_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, self.settings.epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            logger.info(
                "Epoch %3d/%d  lr=%.1e  "
                "train_loss=%.4f  train_acc=%.4f  "
                "val_loss=%.4f  val_acc=%.4f",
                epoch,
                self.settings.epochs,
                current_lr,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info("  → saved best model (val_loss=%.4f)", val_loss)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.settings.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch,
                    self.settings.patience,
                )
                break

        elapsed = time.time() - start_time
        logger.info("Training finished in %.1fs.  Best val_loss=%.4f", elapsed, best_val_loss)

        result: dict[str, list[float] | float] = dict(self.history)
        result["training_time"] = elapsed
        return result
