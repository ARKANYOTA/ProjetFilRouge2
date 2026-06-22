from __future__ import annotations

import math
import random

import torch
import torch.nn as nn

from src.config.settings import Settings


def _sample_lambda(alpha: float) -> float:
    """Draw a mixing coefficient from ``Beta(alpha, alpha)`` (seeded via ``random``)."""
    if alpha <= 0.0:
        return 1.0
    return random.betavariate(alpha, alpha)


def _rand_bbox(height: int, width: int, lam: float) -> tuple[int, int, int, int]:
    """Return a random rectangle whose area is ``(1 - lam)`` of the image (CutMix §3.1)."""
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)
    center_y = int(torch.randint(0, height, (1,)).item())
    center_x = int(torch.randint(0, width, (1,)).item())
    y1 = max(center_y - cut_h // 2, 0)
    y2 = min(center_y + cut_h // 2, height)
    x1 = max(center_x - cut_w // 2, 0)
    x2 = min(center_x + cut_w // 2, width)
    return y1, y2, x1, x2


class MixupCutmix:
    """Batch-level MixUp (Zhang et al. 2018) and CutMix (Yun et al. 2019).

    On each call one of the two strategies is selected (when both are enabled),
    a permutation of the batch is mixed in, and the pair of targets plus the
    effective mixing coefficient ``lam`` are returned so the caller can compute
    the linearly interpolated loss ``lam·L(a) + (1-lam)·L(b)``.
    """

    def __init__(self, settings: Settings) -> None:
        self.mixup_alpha = settings.mixup_alpha
        self.cutmix_alpha = settings.cutmix_alpha
        self.switch_prob = settings.mixup_switch_prob

    @property
    def enabled(self) -> bool:
        return self.mixup_alpha > 0.0 or self.cutmix_alpha > 0.0

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        permutation = torch.randperm(inputs.size(0), device=inputs.device)
        targets_b = targets[permutation]

        use_cutmix = self.cutmix_alpha > 0.0 and (
            self.mixup_alpha <= 0.0 or float(torch.rand(1).item()) < self.switch_prob
        )

        if use_cutmix:
            lam = _sample_lambda(self.cutmix_alpha)
            _, _, height, width = inputs.shape
            y1, y2, x1, x2 = _rand_bbox(height, width, lam)
            inputs[:, :, y1:y2, x1:x2] = inputs[permutation, :, y1:y2, x1:x2]
            # Adjust lam to the true pixel ratio actually swapped.
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
        else:
            lam = _sample_lambda(self.mixup_alpha)
            inputs = lam * inputs + (1.0 - lam) * inputs[permutation]

        return inputs, targets, targets_b, lam


def mixup_criterion(
    criterion: nn.Module,
    outputs: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Linearly interpolate the loss of the two mixed targets."""
    loss_a: torch.Tensor = criterion(outputs, targets_a)
    loss_b: torch.Tensor = criterion(outputs, targets_b)
    return lam * loss_a + (1.0 - lam) * loss_b
