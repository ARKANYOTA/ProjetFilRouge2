from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Dataset settings
    dataset_path: str = "./data"
    image_size: int = 224

    # Model settings
    model_name: str = "resnet18"
    pretrained: bool = True
    in_channels: int = 3
    num_classes: int = 2

    # Training settings
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    seed: int = 42
    device: str = "cpu"

    # Bias evaluation settings
    p0: float = 0.5
    p1: float = 0.5
