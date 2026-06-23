from __future__ import annotations

import argparse
import logging
import sys

from src.config.settings import Settings
from src.data.pipeline import build_dataloaders
from src.evaluation.metrics import (
    export_results_json,
    plot_results_summary,
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


_RECIPES = {
    "convnext": Settings.convnext_recipe,
    "resnet50": Settings.resnet50_recipe,
    "resnet34_hires": Settings.resnet34_hires_recipe,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a from-scratch TILDA model with its tuned recipe."
    )
    parser.add_argument(
        "--recipe",
        choices=sorted(_RECIPES),
        default="convnext",
        help="Which from-scratch training recipe to run (default: convnext)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the recipe epoch count (e.g. for smoke)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast end-to-end sanity run (few epochs, short warm-up)",
    )
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    """Resolve the chosen recipe with optional CLI overrides."""
    settings = _RECIPES[args.recipe]()
    if args.smoke:
        settings = settings.model_copy(
            update={"epochs": 3, "warmup_epochs": 1, "use_ema": False, "patience": 3}
        )
    if args.epochs is not None:
        settings = settings.model_copy(update={"epochs": args.epochs})
    return settings


def main() -> int:
    """Train the chosen from-scratch model and persist artefacts under results/."""
    args = _build_parser().parse_args()
    settings = build_settings(args)
    device = settings.resolve_device()

    logger.info("=" * 60)
    logger.info("Training %s (from scratch) on %s", settings.model_name.upper(), device)
    logger.info(
        "epochs=%d optimizer=%s scheduler=%s lr=%.1e wd=%.3f batch=%d",
        settings.epochs,
        settings.optimizer_name,
        settings.scheduler_name,
        settings.learning_rate,
        settings.weight_decay,
        settings.batch_size,
    )
    logger.info("=" * 60)

    seed_everything(settings.seed)
    train_loader, val_loader = build_dataloaders(settings)
    model = get_model(settings)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    trainer = Trainer(model, train_loader, val_loader, settings)
    history = trainer.fit()

    plot_training_curves(history, settings.model_name)

    run_metadata: dict[str, object] = {
        "optimizer": settings.optimizer_name,
        "scheduler": settings.scheduler_name,
        "learning_rate": settings.learning_rate,
        "batch_size": settings.batch_size,
        "weight_decay": settings.weight_decay,
        "label_smoothing": settings.label_smoothing,
        "warmup_epochs": settings.warmup_epochs,
        "mixup_alpha": settings.mixup_alpha,
        "cutmix_alpha": settings.cutmix_alpha,
        "use_ema": settings.use_ema,
        "drop_path_rate": settings.drop_path_rate,
        "from_scratch": True,
        "n_params": n_params,
    }
    summary = export_results_json({settings.model_name: history}, run_metadata)
    plot_results_summary(summary)
    logger.info("Done. Results saved to results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
