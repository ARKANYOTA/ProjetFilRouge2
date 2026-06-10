from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.settings import Settings
from src.data.dataset import BiasedDataset
from src.data.pipeline import build_dataloaders
from src.evaluation.metrics import (
    compute_accuracy,
    compute_disparate_impact,
    plot_bias_evaluation,
)
from src.models.cnn import get_model
from src.training.trainer import Trainer
from src.utils.seed import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_safety_model(
    model: nn.Module,
    loader: DataLoader[tuple[object, object]],
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate accuracy and Disparate Impact (DI) on the given loader."""
    model.eval()
    predictions_list: list[int] = []
    targets_list: list[int] = []

    # Extract S values (ensure loader is not shuffled for sequential alignment)
    dataset = loader.dataset
    assert hasattr(dataset, "s_values"), "Dataset must be a BiasedDataset with s_values"
    s_values = dataset.s_values

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, preds = outputs.max(1)

            predictions_list.extend(preds.cpu().numpy().tolist())
            targets_list.extend(targets.cpu().numpy().tolist())

    predictions = np.array(predictions_list)
    targets = np.array(targets_list)
    bias_variables = np.array(s_values[: len(predictions)])  # handle potential mismatch

    acc = compute_accuracy(predictions, targets)
    di = compute_disparate_impact(predictions, bias_variables)

    return acc, di


def run_experiment(model_name: str, settings: Settings) -> dict[str, dict[str, float]]:
    """Train and evaluate Model 1 (unbiased) and Model 2 (biased) side-by-side."""
    # We restrict to binary classification of classes 0 and 1
    target_labels = [0, 1]

    # Model 1: Unbiased training (p0=0.5, p1=0.5)
    settings_m1 = settings.model_copy(
        update={
            "model_name": model_name,
            "in_channels": 2,  # original image + bias channel
            "num_classes": 2,
        }
    )
    device = settings_m1.resolve_device()

    logger.info("=" * 60)
    logger.info("Training Model 1 (Unbiased) using %s on %s", model_name.upper(), device)
    logger.info("=" * 60)

    # Build base dataloaders for binary task
    seed_everything(settings_m1.seed)
    train_loader_base, val_loader_base = build_dataloaders(settings_m1, target_labels=target_labels)

    # Wrap in biased dataset
    train_ds_m1 = BiasedDataset(train_loader_base.dataset, p0=0.5, p1=0.5, seed=settings_m1.seed)
    val_ds_unbiased = BiasedDataset(val_loader_base.dataset, p0=0.5, p1=0.5, seed=settings_m1.seed)

    train_loader_m1 = DataLoader(
        train_ds_m1,
        batch_size=settings_m1.batch_size,
        shuffle=True,
        num_workers=settings_m1.num_workers,
    )
    val_loader_unbiased = DataLoader(
        val_ds_unbiased,
        batch_size=settings_m1.batch_size,
        shuffle=False,  # MUST be False for evaluation alignment
        num_workers=settings_m1.num_workers,
    )

    model_1 = get_model(settings_m1)
    # Temporary rename for unique checkpoint naming
    settings_m1.model_name = f"{model_name}_unbiased"
    trainer_1 = Trainer(model_1, train_loader_m1, val_loader_unbiased, settings_m1)
    trainer_1.fit()

    # Model 2: Biased training (p0=0.0, p1=1.0)
    settings_m2 = settings.model_copy(
        update={
            "model_name": model_name,
            "in_channels": 2,
            "num_classes": 2,
        }
    )

    logger.info("=" * 60)
    logger.info("Training Model 2 (Biased) using %s on %s", model_name.upper(), device)
    logger.info("=" * 60)

    # Wrap train dataset in biased wrapper
    train_ds_m2 = BiasedDataset(train_loader_base.dataset, p0=0.0, p1=1.0, seed=settings_m2.seed)

    train_loader_m2 = DataLoader(
        train_ds_m2,
        batch_size=settings_m2.batch_size,
        shuffle=True,
        num_workers=settings_m2.num_workers,
    )

    model_2 = get_model(settings_m2)
    settings_m2.model_name = f"{model_name}_biased"
    trainer_2 = Trainer(model_2, train_loader_m2, val_loader_unbiased, settings_m2)
    trainer_2.fit()

    # Load best checkpoints for evaluation
    results_dir = Path("results")
    path_m1 = results_dir / f"{model_name}_unbiased_best.pt"
    path_m2 = results_dir / f"{model_name}_biased_best.pt"
    model_1.load_state_dict(torch.load(path_m1, map_location=device))
    model_2.load_state_dict(torch.load(path_m2, map_location=device))

    # Evaluate
    logger.info("Evaluating safety models on unbiased validation data...")
    m1_acc, m1_di = evaluate_safety_model(model_1, val_loader_unbiased, device)
    m2_acc, m2_di = evaluate_safety_model(model_2, val_loader_unbiased, device)

    logger.info("Model 1 (Unbiased): Accuracy = %.4f, Disparate Impact = %.4f", m1_acc, m1_di)
    logger.info("Model 2 (Biased):   Accuracy = %.4f, Disparate Impact = %.4f", m2_acc, m2_di)

    results = {
        "Model 1 (Unbiased)": {"accuracy": m1_acc, "di": m1_di},
        "Model 2 (Biased)": {"accuracy": m2_acc, "di": m2_di},
    }

    # Generate safety plots
    plot_bias_evaluation(
        accuracy_results={k: v["accuracy"] for k, v in results.items()},
        di_results={k: v["di"] for k, v in results.items()},
    )

    # Persist safety results to JSON
    import json

    with open(results_dir / "safety_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> int:
    """Run Part 3 AI safety experiments."""
    parser = argparse.ArgumentParser(description="TILDA AI Safety/Bias Experiments")
    parser.add_argument(
        "--model",
        type=str,
        default="lenet5",
        choices=["lenet5", "alexnet", "resnet18"],
        help="Base model architecture to use (default: lenet5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train safety models (default: 30)",
    )
    args = parser.parse_args()

    settings = Settings()
    # Override settings epochs if passed via CLI
    settings.epochs = args.epochs

    run_experiment(args.model, settings)
    logger.info("Safety experiment completed. Plots and JSON saved in results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
