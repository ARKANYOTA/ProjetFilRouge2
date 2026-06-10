from __future__ import annotations

import torch

from src.config.settings import Settings
from src.data.pipeline import build_transforms


def test_train_transforms_output_shape() -> None:
    """Training transforms produce a (1, 224, 224) tensor from a PIL image."""
    from PIL import Image

    s = Settings(image_size=224, in_channels=1, device="cpu")
    t = build_transforms(s, is_train=True)
    # Create a dummy grayscale image
    img = Image.new("L", (768, 512), color=128)
    tensor = t(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 224, 224)


def test_val_transforms_output_shape() -> None:
    """Validation transforms produce a (1, 224, 224) tensor."""
    from PIL import Image

    s = Settings(image_size=224, in_channels=1, device="cpu")
    t = build_transforms(s, is_train=False)
    img = Image.new("L", (768, 512), color=128)
    tensor = t(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 224, 224)


def test_normalisation_range() -> None:
    """After Normalize(mean=0.5, std=0.5) a uniform-128 image is near 0."""
    from PIL import Image

    s = Settings(image_size=224, in_channels=1, device="cpu")
    t = build_transforms(s, is_train=False)
    img = Image.new("L", (768, 512), color=128)
    tensor = t(img)
    # 128/255 ≈ 0.502 → (0.502 - 0.5) / 0.5 ≈ 0.004
    assert tensor.mean().abs() < 0.1


def test_biased_dataset_channels() -> None:
    """BiasedDataset concatenates an epsilon channel making output shape (2, 224, 224)."""
    import csv
    import os
    import tempfile
    from PIL import Image
    from torchvision import transforms as tv_transforms
    from src.data.dataset import BiasedDataset, TILDADataset

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, "train"), exist_ok=True)
        csv_path = os.path.join(tmp_dir, "train.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["id", "label"])
            writer.writerow(["1", "0"])
            writer.writerow(["2", "1"])

        Image.new("L", (100, 100), color=128).save(os.path.join(tmp_dir, "train", "1.tif"))
        Image.new("L", (100, 100), color=128).save(os.path.join(tmp_dir, "train", "2.tif"))

        base_ds = TILDADataset(
            root_dir=tmp_dir,
            train=True,
            transform=tv_transforms.Compose(
                [
                    tv_transforms.Resize((224, 224)),
                    tv_transforms.ToTensor(),
                ]
            ),
        )

        # S = 0 for both samples
        biased_ds_0 = BiasedDataset(base_ds, p0=0.0, p1=0.0, seed=42)
        x0, y0 = biased_ds_0[0]
        assert x0.shape == (2, 224, 224)
        assert y0 == 0
        assert (x0[1] == 0.0).all()

        # S = 1 for both samples
        biased_ds_1 = BiasedDataset(base_ds, p0=1.0, p1=1.0, seed=42)
        x1, y1 = biased_ds_1[0]
        assert x1.shape == (2, 224, 224)
        assert y1 == 0
        assert float(x1[1].var()) > 0.1

