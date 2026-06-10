from __future__ import annotations

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config.settings import Settings
from src.data.dataset import TILDADataset


def build_transforms(settings: Settings, is_train: bool = True) -> transforms.Compose:
    """Build torchvision transforms for the TILDA dataset.

    Training transforms include data-augmentation (horizontal/vertical flips,
    small rotations and translations) as recommended by AlexNet §4.1 and
    general best practice for texture classification.  Validation/test
    transforms only resize and normalise.

    Normalisation centres pixel values around 0 with unit variance, following
    LeNet paper §II.B ("mean roughly 0, variance roughly 1").
    """
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((settings.image_size, settings.image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((settings.image_size, settings.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def build_dataloaders(
    settings: Settings,
    target_labels: list[int] | None = None,
) -> tuple[DataLoader[tuple[object, object]], DataLoader[tuple[object, object]]]:
    """Build stratified train / validation DataLoaders.

    Uses an 80/20 stratified split controlled by ``settings.seed`` for
    reproducibility.
    """
    from sklearn.model_selection import train_test_split

    root = settings.get_resolved_dataset_path()

    # Build the full training dataset with *train* augmentation first
    # (we will wrap subsets with different transforms afterwards).
    full_dataset = TILDADataset(root, train=True, transform=None, target_labels=target_labels)
    labels = full_dataset.labels
    indices = list(range(len(full_dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=settings.val_split,
        stratify=labels,
        random_state=settings.seed,
    )

    train_transform = build_transforms(settings, is_train=True)
    val_transform = build_transforms(settings, is_train=False)

    train_subset = _TransformSubset(full_dataset, train_idx, train_transform)
    val_subset = _TransformSubset(full_dataset, val_idx, val_transform)

    train_loader: DataLoader[tuple[object, object]] = DataLoader(
        train_subset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        drop_last=False,
    )
    val_loader: DataLoader[tuple[object, object]] = DataLoader(
        val_subset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


class _TransformSubset(Dataset[tuple[object, object]]):
    """Subset of a :class:`TILDADataset` that applies a specific transform.

    This avoids the need to create two full dataset objects with different
    transforms — the underlying images are shared.
    """

    def __init__(
        self,
        dataset: TILDADataset,
        indices: list[int],
        transform: transforms.Compose,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[object, object]:
        from PIL import Image

        real_idx = self.indices[idx]
        img_path, label = self.dataset.samples[real_idx]
        image = Image.open(img_path).convert("L")
        tensor = self.transform(image)
        return tensor, label
