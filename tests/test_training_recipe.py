from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.config.settings import Settings
from src.models.cnn import LeNet5
from src.training.ema import ModelEma
from src.training.mixup import MixupCutmix, mixup_criterion
from src.training.trainer import Trainer


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {"in_channels": 1, "num_classes": 8, "device": "cpu"}
    defaults.update(overrides)
    return Settings.model_validate(defaults)


def _tiny_loaders() -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    x = torch.randn(8, 1, 32, 32)
    y = torch.randint(0, 8, (8,))
    ds = cast("Dataset[tuple[torch.Tensor, torch.Tensor]]", TensorDataset(x, y))
    return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4)


def test_mixup_disabled_by_default() -> None:
    assert MixupCutmix(_settings()).enabled is False


def test_mixup_returns_consistent_shapes() -> None:
    s = _settings(mixup_alpha=0.2, cutmix_alpha=1.0)
    mc = MixupCutmix(s)
    assert mc.enabled
    x = torch.randn(4, 1, 32, 32)
    y = torch.randint(0, 8, (4,))
    mixed, a, b, lam = mc(x, y)
    assert mixed.shape == x.shape
    assert a.shape == (4,) and b.shape == (4,)
    assert 0.0 <= lam <= 1.0


def test_mixup_criterion_interpolates() -> None:
    crit = torch.nn.CrossEntropyLoss()
    outputs = torch.randn(4, 8)
    a = torch.randint(0, 8, (4,))
    b = torch.randint(0, 8, (4,))
    loss_full_a = mixup_criterion(crit, outputs, a, b, 1.0)
    assert torch.allclose(loss_full_a, crit(outputs, a))


def test_ema_tracks_weights() -> None:
    model = LeNet5(_settings())
    ema = ModelEma(model, decay=0.5)
    # Mutate the model, then a single EMA update should move halfway.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    before = next(iter(ema.module.state_dict().values())).clone()
    ema.update(model)
    after = next(iter(ema.module.state_dict().values()))
    assert not torch.allclose(before, after)


def test_ema_keys_match_model() -> None:
    model = LeNet5(_settings())
    ema = ModelEma(model, decay=0.9)
    assert set(ema.module.state_dict()) == set(model.state_dict())


def test_cosine_warmup_factor_profile() -> None:
    s = _settings(scheduler_name="cosine", epochs=10, warmup_epochs=2, min_lr_ratio=0.0)
    train_loader, val_loader = _tiny_loaders()
    trainer = Trainer(LeNet5(s), train_loader, val_loader, s)
    # Linear warm-up climbs, then cosine decays to the floor.
    assert trainer._cosine_warmup_factor(0) < trainer._cosine_warmup_factor(1)
    assert trainer._cosine_warmup_factor(1) == 1.0  # peak at end of warm-up
    assert trainer._cosine_warmup_factor(9) < trainer._cosine_warmup_factor(2)


def test_trainer_runs_with_full_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A short fit with MixUp+CutMix, EMA, cosine schedule and acc checkpointing runs."""
    monkeypatch.chdir(tmp_path)  # isolate the results/ checkpoint write
    s = _settings(
        model_name="lenet5_recipe_test",
        epochs=2,
        warmup_epochs=1,
        scheduler_name="cosine",
        optimizer_name="adamw",
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        use_ema=True,
        checkpoint_metric="acc",
        patience=2,
    )
    train_loader, val_loader = _tiny_loaders()
    trainer = Trainer(LeNet5(s), train_loader, val_loader, s)
    history = trainer.fit()
    val_acc = history["val_acc"]
    assert isinstance(val_acc, list)
    assert len(val_acc) == 2
    assert "training_time" in history
    assert (tmp_path / "results" / "lenet5_recipe_test_best.pt").is_file()
