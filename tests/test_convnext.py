from __future__ import annotations

import torch

from src.config.settings import Settings
from src.models.cnn import ConvNeXt, convnext, get_model


def _make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "image_size": 224,
        "in_channels": 1,
        "num_classes": 8,
        "device": "cpu",
        "model_name": "convnext",
        # small config keeps the test fast
        "convnext_depths": (1, 1, 1, 1),
        "convnext_dims": (16, 32, 64, 128),
    }
    defaults.update(overrides)
    return Settings.model_validate(defaults)


def test_convnext_forward_shape() -> None:
    """ConvNeXt produces (batch, num_classes) output on grayscale input."""
    model = convnext(_make_settings())
    out = model(torch.randn(2, 1, 224, 224))
    assert out.shape == (2, 8)


def test_convnext_accepts_3_channels() -> None:
    """ConvNeXt stem adapts to a 3-channel input (used by the bias part)."""
    model = convnext(_make_settings(in_channels=3))
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 8)


def test_get_model_returns_convnext() -> None:
    """The factory dispatches the 'convnext' name to a ConvNeXt instance."""
    model = get_model(_make_settings())
    assert isinstance(model, ConvNeXt)


def test_convnext_head_keys_stable_across_dropout() -> None:
    """Head checkpoint keys are independent of the configured dropout value."""
    keys_no_drop = set(convnext(_make_settings(head_dropout=0.0)).state_dict())
    keys_drop = set(convnext(_make_settings(head_dropout=0.5)).state_dict())
    assert keys_no_drop == keys_drop
    assert "head.1.weight" in keys_no_drop  # Linear is always at index 1


def test_layer_scale_can_be_disabled() -> None:
    """layer_scale_init=0 removes the LayerScale parameters."""
    with_scale = set(convnext(_make_settings(layer_scale_init=1e-6)).state_dict())
    without_scale = set(convnext(_make_settings(layer_scale_init=0.0)).state_dict())
    assert any(key.endswith("gamma") for key in with_scale)
    assert not any(key.endswith("gamma") for key in without_scale)


def test_convnext_recipe_is_from_scratch() -> None:
    """The core recipe never enables pretrained weights; cosine + acc selection."""
    recipe = Settings.convnext_recipe()
    assert recipe.model_name == "convnext"
    assert recipe.pretrained is False
    assert recipe.scheduler_name == "cosine"
    assert recipe.warmup_epochs > 0
    assert recipe.label_smoothing > 0.0
    assert recipe.checkpoint_metric == "acc"
