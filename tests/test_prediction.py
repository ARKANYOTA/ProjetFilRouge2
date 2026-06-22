from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from src.data.dataset import TILDADataset
from src.predict import Prediction, validate_submission, write_submission


def test_test_dataset_uses_numeric_id_order(tmp_path: Path) -> None:
    """Test images are ordered 1, 2, 10 rather than lexicographically."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for sample_id in ("10", "2", "1"):
        Image.new("L", (8, 8), color=128).save(test_dir / f"{sample_id}.tif")

    dataset = TILDADataset(str(tmp_path), train=False)

    assert dataset.sample_ids == ["1", "2", "10"]


def test_submission_round_trip_is_strict(tmp_path: Path) -> None:
    """A generated CSV has the exact schema, ordering, labels, and checksum."""
    path = tmp_path / "submission.csv"
    predictions = [
        Prediction("1", 3, 0.8, 2, 0.1),
        Prediction("2", 0, 0.7, 1, 0.2),
        Prediction("10", 7, 0.6, 6, 0.3),
    ]

    write_submission(predictions, path, delimiter=";")
    summary = validate_submission(path, ["1", "2", "10"], num_classes=8, delimiter=";")

    assert path.read_text(encoding="utf-8").splitlines() == [
        "id;label",
        "1;3",
        "2;0",
        "10;7",
    ]
    assert summary.row_count == 3
    assert summary.first_id == "1"
    assert summary.last_id == "10"
    assert summary.class_histogram == {
        "0": 1,
        "1": 0,
        "2": 0,
        "3": 1,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 1,
    }
    assert len(summary.sha256) == 64


def test_submission_validation_rejects_reordered_ids(tmp_path: Path) -> None:
    """Manual edits that change test ID order are rejected with a useful error."""
    path = tmp_path / "submission.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";", lineterminator="\n")
        writer.writerow(("id", "label"))
        writer.writerow(("2", 0))
        writer.writerow(("1", 1))

    with pytest.raises(ValueError, match="out of order"):
        validate_submission(path, ["1", "2"], num_classes=8, delimiter=";")


def test_submission_validation_rejects_extra_column(tmp_path: Path) -> None:
    """Only id and label are permitted in the uploadable file."""
    path = tmp_path / "submission.csv"
    path.write_text("id;label;confidence\n1;0;0.9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact header"):
        validate_submission(path, ["1"], num_classes=8, delimiter=";")
