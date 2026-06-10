from __future__ import annotations

import csv
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TILDADataset(Dataset[tuple[torch.Tensor, int]]):
    """TILDA textile texture dataset.

    Reads ``train.csv`` (columns ``id;label``) and loads ``.tif`` images from
    the corresponding directory.  When *train* is ``False`` the dataset returns
    label ``-1`` for every image (test set has no labels).
    """

    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        transform: transforms.Compose | None = None,
        target_labels: list[int] | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.target_labels = target_labels

        if train:
            csv_path = os.path.join(root_dir, "train.csv")
            img_dir = os.path.join(root_dir, "train")
            self.samples: list[tuple[str, int]] = []
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    fname = f"{row['id']}.tif"
                    label = int(row["label"])
                    if target_labels is not None:
                        if label in target_labels:
                            mapped_label = target_labels.index(label)
                            self.samples.append((os.path.join(img_dir, fname), mapped_label))
                    else:
                        self.samples.append((os.path.join(img_dir, fname), label))
        else:
            img_dir = os.path.join(root_dir, "test")
            self.samples = [
                (os.path.join(img_dir, fname), -1)
                for fname in sorted(Path(img_dir).iterdir())
                if fname.suffix == ".tif"
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")  # grayscale

        if self.transform is not None:
            tensor: torch.Tensor = self.transform(image)
        else:
            tensor = transforms.ToTensor()(image)

        return tensor, label

    @property
    def labels(self) -> list[int]:
        """Return all labels (useful for stratified splitting)."""
        return [label for _, label in self.samples]


class BiasedDataset(Dataset[tuple[torch.Tensor, int]]):
    """Wrapper that appends a bias channel *epsilon* to each image.

    Given probabilities *p0* and *p1*, for each sample ``(x, y)`` a Bernoulli
    variable ``S ~ Bernoulli(p_y)`` is drawn.  When ``S = 1``, ``epsilon`` is
    sampled from ``N(0, I)``; when ``S = 0``, ``epsilon = 0``.  The channel is
    concatenated to ``x`` along dim-0.
    """

    def __init__(
        self,
        base_dataset: Dataset[tuple[object, object]],
        p0: float,
        p1: float,
        seed: int | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.p0 = p0
        self.p1 = p1
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        # Pre-compute S values for reproducibility
        self._s_values: list[int] = []
        labels: list[int] = []

        if hasattr(base_dataset, "indices") and hasattr(base_dataset, "dataset"):
            # it's a _TransformSubset
            indices: list[int] = base_dataset.indices
            dataset: TILDADataset = base_dataset.dataset
            for idx in indices:
                labels.append(dataset.samples[idx][1])
        elif hasattr(base_dataset, "samples"):
            # it's a TILDADataset
            tilda_ds: TILDADataset = base_dataset  # type: ignore[assignment]
            for _, lbl in tilda_ds.samples:
                labels.append(lbl)
        else:
            # Fallback: sequentially load labels
            for idx in range(len(base_dataset)):  # type: ignore[arg-type]
                item = base_dataset[idx]
                lbl_val = item[1]
                assert isinstance(lbl_val, int)
                labels.append(lbl_val)

        for label in labels:
            p = self.p0 if label == 0 else self.p1
            s = int(torch.bernoulli(torch.tensor(p), generator=self.rng).item())
            self._s_values.append(s)

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        res = self.base_dataset[idx]
        x, label = res
        assert isinstance(x, torch.Tensor)
        assert isinstance(label, int)
        _, h, w = x.shape
        s = self._s_values[idx]

        epsilon = torch.randn(1, h, w) if s == 1 else torch.zeros(1, h, w)

        biased_x = torch.cat([x, epsilon], dim=0)
        return biased_x, label

    @property
    def s_values(self) -> list[int]:
        """Return the pre-computed bias variable S for every sample."""
        return self._s_values
