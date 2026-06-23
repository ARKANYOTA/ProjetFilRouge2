from __future__ import annotations

import argparse
import logging
import sys

import torch

from src.config.settings import Settings
from src.data.pipeline import build_dataloaders
from src.models.cnn import get_model
from src.predict import _tta_probabilities
from src.utils.seed import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_validation(settings: Settings, use_tta: bool) -> float:
    """Return the validation accuracy of a persisted checkpoint on the seeded split.

    Re-uses the exact seeded stratified split from :func:`build_dataloaders`, so
    the number is directly comparable to the validation accuracies reported
    during training.
    """
    seed_everything(settings.seed)
    _, val_loader = build_dataloaders(settings)
    device = settings.resolve_device()
    model = get_model(settings)
    state_dict: dict[str, torch.Tensor] = torch.load(
        settings.get_resolved_checkpoint_path(), map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    correct = 0
    total = 0
    for inputs, targets in val_loader:
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        inputs = inputs.to(device)
        targets = targets.to(device)
        if use_tta:
            probabilities = _tta_probabilities(model, inputs)
        else:
            probabilities = torch.softmax(model(inputs), dim=1)
        correct += int((probabilities.argmax(dim=1) == targets).sum().item())
        total += int(targets.size(0))

    return correct / max(total, 1)


def _resolve_settings(model_name: str) -> Settings:
    """Pick the settings profile that matches how each model was trained."""
    if model_name == "convnext":
        return Settings.convnext_recipe()
    return Settings().model_copy(update={"model_name": model_name})


def main() -> int:
    """Evaluate a checkpoint on the validation split, with and without TTA."""
    parser = argparse.ArgumentParser(description="Validation evaluation for a TILDA checkpoint.")
    parser.add_argument("--model", required=True, help="Model name (e.g. resnet34, convnext)")
    args = parser.parse_args()

    settings = _resolve_settings(args.model)
    plain = evaluate_validation(settings, use_tta=False)
    tta = evaluate_validation(settings, use_tta=True)
    logger.info(
        "%s validation accuracy: plain=%.4f  TTA=%.4f  (delta=%+.4f)",
        args.model,
        plain,
        tta,
        tta - plain,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
