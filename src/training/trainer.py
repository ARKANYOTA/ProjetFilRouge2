from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.config.settings import Settings
from src.training.ema import ModelEma
from src.training.mixup import MixupCutmix

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

        # Optimizer
        self.optimizer: torch.optim.Optimizer
        if getattr(settings, "optimizer_name", "sgd") == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=settings.learning_rate,
                weight_decay=settings.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=settings.learning_rate,
                momentum=settings.momentum,
                weight_decay=settings.weight_decay,
            )

        label_smoothing = getattr(settings, "label_smoothing", 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Modern training recipe (opt-in via settings; defaults leave baselines unchanged).
        self.mixup = MixupCutmix(settings)
        self.ema: ModelEma | None = (
            ModelEma(self.model, settings.ema_decay) if settings.use_ema else None
        )
        self.scheduler_name = getattr(settings, "scheduler_name", "plateau")

        self.scheduler: torch.optim.lr_scheduler.LRScheduler | ReduceLROnPlateau
        if self.scheduler_name == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=settings.learning_rate,
                steps_per_epoch=len(self.train_loader),
                epochs=settings.epochs,
            )
        elif self.scheduler_name == "cosine":
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._cosine_warmup_factor)
        else:
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
    # Cosine schedule with linear warm-up
    # ------------------------------------------------------------------
    def _cosine_warmup_factor(self, epoch_idx: int) -> float:
        """LR multiplier for the 'cosine' scheduler at a 0-based epoch index."""
        warmup = self.settings.warmup_epochs
        total = self.settings.epochs
        if epoch_idx < warmup:
            return (epoch_idx + 1) / max(1, warmup)
        progress = (epoch_idx - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = self.settings.min_lr_ratio
        return floor + (1.0 - floor) * cosine

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------
    def train_epoch(self) -> tuple[float, float]:
        """Run one epoch of training.  Returns ``(avg_loss, accuracy)``.

        When MixUp/CutMix is enabled the reported accuracy is measured against
        the dominant (first) target and is therefore only an approximation; the
        validation accuracy remains the authoritative metric.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            if self.mixup.enabled:
                mixed, targets_a, targets_b, lam = self.mixup(inputs, targets)
                outputs = self.model(mixed)
                # Interpolated loss over the two mixed targets (MixUp/CutMix).
                loss = lam * self.criterion(outputs, targets_a) + (1.0 - lam) * self.criterion(
                    outputs, targets_b
                )
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler_name == "onecycle" and isinstance(
                self.scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)

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
    def validate(self, model: nn.Module | None = None) -> tuple[float, float]:
        """Evaluate on the validation set.  Returns ``(avg_loss, accuracy)``.

        When ``model`` is ``None`` the trained model is used; pass the EMA module
        to evaluate the moving-average weights instead.
        """
        eval_model = model if model is not None else self.model
        eval_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = eval_model(inputs)
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

        # The checkpointed/validated model is the EMA copy when EMA is enabled.
        eval_model = self.ema.module if self.ema is not None else self.model
        metric = self.settings.checkpoint_metric
        best_val_loss = float("inf")
        best_val_acc = float("-inf")  # guarantees a checkpoint is saved at epoch 1
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, self.settings.epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate(eval_model)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler_name == "plateau" and isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler_name == "cosine" and isinstance(self.scheduler, LambdaLR):
                self.scheduler.step()

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

            # Checkpoint best model by the configured selection metric.
            improved = val_acc > best_val_acc if metric == "acc" else val_loss < best_val_loss
            if improved:
                best_val_loss = min(best_val_loss, val_loss)
                best_val_acc = max(best_val_acc, val_acc)
                patience_counter = 0
                torch.save(eval_model.state_dict(), checkpoint_path)
                logger.info("  → saved best model (val_loss=%.4f, val_acc=%.4f)", val_loss, val_acc)
            else:
                patience_counter += 1

            # Release cached MPS memory each epoch to avoid the reserved-memory
            # growth (and eventual swapping/throttling) seen on long runs.
            if self.device.type == "mps":
                torch.mps.empty_cache()

            # Early stopping
            if patience_counter >= self.settings.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch,
                    self.settings.patience,
                )
                break

        elapsed = time.time() - start_time
        logger.info(
            "Training finished in %.1fs.  Best val_loss=%.4f  Best val_acc=%.4f",
            elapsed,
            best_val_loss,
            best_val_acc,
        )

        result: dict[str, list[float] | float] = dict(self.history)
        result["training_time"] = elapsed
        return result
