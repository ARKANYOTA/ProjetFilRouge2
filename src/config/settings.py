from __future__ import annotations

from typing import Literal

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
    pretrained: bool = False  # kept for API compatibility; project trains from scratch only

    # ConvNeXt-from-scratch architecture (Liu et al. 2022, "A ConvNet for the 2020s")
    # Sized down from ConvNeXt-Tiny (depths [3,3,9,3], dims [96,192,384,768]) to suit
    # the small TILDA training set (1,888 images) and avoid overfitting.
    convnext_depths: tuple[int, int, int, int] = (3, 3, 9, 3)
    convnext_dims: tuple[int, int, int, int] = (48, 96, 192, 384)
    drop_path_rate: float = 0.1  # max stochastic-depth probability (linearly scaled by block depth)
    layer_scale_init: float = 1e-6  # LayerScale gamma initial value (0 disables LayerScale)
    head_dropout: float = 0.0  # dropout before the classification head

    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-3  # max_lr for OneCycleLR
    epochs: int = 100
    seed: int = 42
    device: str = "auto"

    # Optimizer choices
    optimizer_name: str = "adamw"  # 'sgd' or 'adamw'
    scheduler_name: str = "onecycle"  # 'plateau', 'onecycle', or 'cosine'
    warmup_epochs: int = 0  # linear LR warm-up epochs (used by the 'cosine' scheduler)
    min_lr_ratio: float = 1e-2  # cosine floor as a fraction of the base learning rate

    # Modern training recipe (opt-in; disabled by default so baselines are unchanged)
    mixup_alpha: float = 0.0  # Beta(alpha, alpha) for MixUp; 0 disables
    cutmix_alpha: float = 0.0  # Beta(alpha, alpha) for CutMix; 0 disables
    mixup_switch_prob: float = 0.5  # probability of choosing CutMix over MixUp when both enabled
    use_ema: bool = False  # maintain an exponential-moving-average copy of the weights
    ema_decay: float = 0.999  # EMA decay factor
    checkpoint_metric: Literal["loss", "acc"] = "loss"  # metric used to select the best checkpoint

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

    # Persisted artefacts and Kaggle prediction
    use_tta: bool = False  # test-time augmentation (average over flips) at inference
    results_dir: str = "results"
    checkpoint_path: str | None = None
    submission_path: str | None = None
    submission_delimiter: Literal[";", ","] = ";"

    # Bias evaluation settings (Part 3)
    p0: float = 0.5
    p1: float = 0.5

    @classmethod
    def convnext_recipe(cls) -> Settings:
        """Return the from-scratch ConvNeXt training recipe.

        Centralises the from-scratch ConvNeXt training recipe (no pretrained
        weights): AdamW + cosine schedule with warm-up, label smoothing,
        stochastic depth and head dropout, with the best checkpoint selected by
        validation accuracy.  MixUp/CutMix and EMA are intentionally left
        *disabled* in this core recipe: on a from-scratch ConvNeXt with few
        optimisation steps per epoch they suppressed early learning (MixUp) and
        biased validation toward a cold moving average (EMA).  They remain
        available via :meth:`model_copy` for follow-up ablations.  Values not
        listed here fall back to the environment / defaults (e.g. dataset path).
        """
        return cls().model_copy(
            update={
                "model_name": "convnext",
                "in_channels": 1,
                "optimizer_name": "adamw",
                "scheduler_name": "cosine",
                "learning_rate": 2e-3,
                "weight_decay": 0.05,
                "epochs": 150,
                "warmup_epochs": 10,
                "min_lr_ratio": 1e-2,
                "mixup_alpha": 0.0,
                "cutmix_alpha": 0.0,
                "mixup_switch_prob": 0.5,
                "use_ema": False,
                "ema_decay": 0.999,
                "label_smoothing": 0.1,
                "head_dropout": 0.1,
                "drop_path_rate": 0.1,
                "checkpoint_metric": "acc",
                "patience": 150,
                "batch_size": 64,
                "num_workers": 0,
            }
        )

    @classmethod
    def resnet50_recipe(cls) -> Settings:
        """Return the from-scratch ResNet-50 training recipe.

        Mirrors the configuration that took ResNet-34 to its 81.4% peak (AdamW +
        OneCycle + label smoothing) on the deeper Bottleneck network, but selects
        the best checkpoint by validation accuracy and uses a larger batch (64)
        for faster, more stable Batch-Norm statistics.  No pretrained weights.
        """
        return cls().model_copy(
            update={
                "model_name": "resnet50",
                "in_channels": 1,
                "optimizer_name": "adamw",
                "scheduler_name": "onecycle",
                "learning_rate": 1e-3,
                "weight_decay": 1e-2,
                "epochs": 150,
                "label_smoothing": 0.1,
                "checkpoint_metric": "acc",
                "patience": 150,
                "batch_size": 64,
                "num_workers": 0,
            }
        )

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

    def get_resolved_checkpoint_path(self) -> str:
        """Return the configured checkpoint or the model's default checkpoint."""
        if self.checkpoint_path:
            return self.checkpoint_path
        return f"{self.results_dir}/{self.model_name}_best.pt"

    def get_resolved_submission_path(self) -> str:
        """Return the configured submission path or a model-specific default."""
        if self.submission_path:
            return self.submission_path
        return f"{self.results_dir}/kaggle_submission_{self.model_name}.csv"
