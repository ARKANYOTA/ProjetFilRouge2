from __future__ import annotations

import argparse
import logging
import sys

from src.config.settings import Settings
from src.data.pipeline import build_dataloaders
from src.evaluation.metrics import (
    export_results_json,
    plot_model_comparison,
    plot_training_curves,
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

ALL_MODELS = ("lenet5", "alexnet", "resnet18")


def _train_single(model_name: str, settings: Settings) -> dict[str, list[float] | float]:
    """Train a single model and return its history."""
    settings_copy = settings.model_copy(update={"model_name": model_name})
    device = settings_copy.resolve_device()
    logger.info("=" * 60)
    logger.info("Training %s on %s", model_name.upper(), device)
    logger.info("=" * 60)

    seed_everything(settings_copy.seed)
    train_loader, val_loader = build_dataloaders(settings_copy)
    model = get_model(settings_copy)

    # Log parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    trainer = Trainer(model, train_loader, val_loader, settings_copy)
    history = trainer.fit()

    plot_training_curves(history, model_name)
    return history


def main() -> int:
    """Train one or all CNN models on the TILDA textile dataset."""
    parser = argparse.ArgumentParser(description="TILDA CNN trainer")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=[*ALL_MODELS, "all"],
        help="Model to train (default: all)",
    )
    args = parser.parse_args()

    settings = Settings()
    logger.info("Device: %s", settings.resolve_device())

    models_to_train = ALL_MODELS if args.model == "all" else (args.model,)
    all_results: dict[str, dict[str, object]] = {}

    for model_name in models_to_train:
        history = _train_single(model_name, settings)
        all_results[model_name] = history  # type: ignore[assignment]

    if len(all_results) > 1:
        plot_model_comparison(all_results)

    export_results_json(all_results)
    logger.info("All done. Results saved to results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
