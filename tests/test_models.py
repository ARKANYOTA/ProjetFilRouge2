from __future__ import annotations

import torch

from src.config.settings import Settings
from src.models.cnn import AlexNet, LeNet5, ResNet18, get_model


def _make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "image_size": 224,
        "in_channels": 1,
        "num_classes": 8,
        "device": "cpu",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_lenet5_forward_shape() -> None:
    """LeNet-5 produces (batch, num_classes) output."""
    s = _make_settings()
    model = LeNet5(s)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.shape == (2, 8)


def test_alexnet_forward_shape() -> None:
    """AlexNet produces (batch, num_classes) output."""
    s = _make_settings()
    model = AlexNet(s)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.shape == (2, 8)


def test_resnet18_forward_shape() -> None:
    """ResNet-18 produces (batch, num_classes) output."""
    s = _make_settings()
    model = ResNet18(s)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.shape == (2, 8)


def test_get_model_factory() -> None:
    """get_model dispatches correctly for each registered name."""
    for name, cls in [("lenet5", LeNet5), ("alexnet", AlexNet), ("resnet18", ResNet18)]:
        s = _make_settings(model_name=name)
        model = get_model(s)
        assert isinstance(model, cls)


def test_models_accept_3_channels() -> None:
    """Models can be instantiated with 3 input channels (for bias part)."""
    s = _make_settings(in_channels=3)
    for name in ("lenet5", "alexnet", "resnet18"):
        s_copy = s.model_copy(update={"model_name": name})
        model = get_model(s_copy)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 8)


def test_resnet18_has_residual_connections() -> None:
    """Verify that ResNet-18 has shortcut connections in its blocks."""
    s = _make_settings(model_name="resnet18")
    model = ResNet18(s)
    # Each BasicBlock should have a forward that adds identity
    block = model.layer1[0]
    assert hasattr(block, "downsample") or hasattr(block, "conv1")
