from __future__ import annotations

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Dataset settings
    dataset_path: str | None = None
    image_size: int = 224
    in_channels: int = 1
    num_classes: int = 8

    # Model settings
    model_name: str = "resnet34"
    pretrained: bool = False

    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-3  # max_lr for OneCycleLR
    epochs: int = 100
    seed: int = 42
    device: str = "auto"

    # Optimizer choices
    optimizer_name: str = "adamw"  # 'sgd' or 'adamw'
    scheduler_name: str = "onecycle"  # 'plateau' or 'onecycle'

    # SGD hyper-parameters
    momentum: float = 0.9
    weight_decay: float = 1e-2  # Decoupled weight decay for AdamW

    # Regularisation
    dropout_rate: float = 0.5
    label_smoothing: float = 0.1

    # Validation
    val_split: float = 0.2
    patience: int = 20

    # Learning-rate scheduler (Plateau)
    scheduler_factor: float = 0.1
    scheduler_patience: int = 5

    # DataLoader
    num_workers: int = 0

    # Bias evaluation settings (Part 3)
    p0: float = 0.5
    p1: float = 0.5

    def get_resolved_dataset_path(self) -> str:
        if self.dataset_path:
            return self.dataset_path
        import kagglehub

        path = kagglehub.competition_download("modia-ml-2026")
        # kagglehub extracts into a subdirectory (e.g. data_kaggle/)
        import os

        data_subdir = os.path.join(str(path), "data_kaggle")
        if os.path.isdir(data_subdir):
            return data_subdir
        return str(path)

    def resolve_device(self) -> torch.device:
        """Return the best available device (MPS → CUDA → CPU)."""
        if self.device != "auto":
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
