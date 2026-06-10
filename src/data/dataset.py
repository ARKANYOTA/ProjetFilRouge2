from __future__ import annotations

from typing import Any
import torch
from torch.utils.data import Dataset


class TILDADataset(Dataset[tuple[torch.Tensor, int]]):
    """Skeleton for the raw TILDA textile texture dataset."""

    def __init__(self, root_dir: str, train: bool = True) -> None:
        self.root_dir = root_dir
        self.train = train

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Dummy tensor and label stub
        return torch.zeros((3, 224, 224)), 0


class BiasedDataset(Dataset[tuple[torch.Tensor, int]]):
    """Skeleton wrapper dataset that appends a bias channel epsilon based on probabilities p0 and p1."""

    def __init__(
        self,
        base_dataset: Dataset[tuple[torch.Tensor, int]],
        p0: float,
        p1: float,
        seed: int | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.p0 = p0
        self.p1 = p1
        self.seed = seed

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Dummy tensor with an added channel and label stub
        return torch.zeros((4, 224, 224)), 0
