from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEma:
    """Exponential moving average of model weights (Polyak averaging).

    Keeps a shadow copy of the model whose parameters track the trained weights
    with decay ``d``: ``ema = d·ema + (1-d)·model``.  EMA weights are typically
    smoother and generalise better, especially for from-scratch training on
    small datasets.  Non-floating buffers are copied verbatim.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            ema_value = ema_state[key]
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                ema_value.copy_(value)
