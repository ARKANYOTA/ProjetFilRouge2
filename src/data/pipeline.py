from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset

from src.config.settings import Settings


def build_transforms(settings: Settings, is_train: bool = True) -> Any:
    """Returns torchvision transforms for the dataset."""
    pass


def get_dataloader(
    dataset: Dataset[tuple[Any, Any]],
    settings: Settings,
    shuffle: bool = True,
) -> DataLoader[tuple[Any, Any]]:
    """Builds a PyTorch DataLoader for the dataset."""
    # Returns an empty/dummy dataloader stub
    return DataLoader(dataset, batch_size=settings.batch_size, shuffle=shuffle)
