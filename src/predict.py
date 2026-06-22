from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.settings import Settings
from src.data.dataset import TILDADataset
from src.data.pipeline import build_transforms
from src.models.cnn import get_model
from src.utils.seed import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUBMISSION_HEADER = ("id", "label")


@dataclass(frozen=True)
class Prediction:
    """One model prediction plus confidence values used for manual auditing."""

    sample_id: str
    label: int
    confidence: float
    runner_up_label: int
    runner_up_confidence: float

    @property
    def margin(self) -> float:
        """Return the confidence gap between the two most likely classes."""
        return self.confidence - self.runner_up_confidence


@dataclass(frozen=True)
class ValidationSummary:
    """Strict structural checks performed on a generated Kaggle CSV."""

    row_count: int
    sha256: str
    first_id: str
    last_id: str
    class_histogram: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_test_loader(
    settings: Settings,
) -> tuple[TILDADataset, DataLoader[tuple[object, object]]]:
    """Build a deterministic, numerically ordered test loader."""
    dataset = TILDADataset(
        settings.get_resolved_dataset_path(),
        train=False,
        transform=build_transforms(settings, is_train=False),
    )
    loader: DataLoader[tuple[object, object]] = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        drop_last=False,
    )
    return dataset, loader


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader[tuple[object, object]],
    sample_ids: Sequence[str],
    device: torch.device,
) -> list[Prediction]:
    """Run deterministic inference and retain top-two confidence values."""
    model.eval()
    predictions: list[Prediction] = []
    sample_index = 0

    for inputs, _ in loader:
        assert isinstance(inputs, torch.Tensor)
        probabilities = torch.softmax(model(inputs.to(device)), dim=1)
        top_probabilities, top_labels = probabilities.topk(k=2, dim=1)

        for row_index in range(inputs.size(0)):
            predictions.append(
                Prediction(
                    sample_id=sample_ids[sample_index],
                    label=int(top_labels[row_index, 0].item()),
                    confidence=float(top_probabilities[row_index, 0].item()),
                    runner_up_label=int(top_labels[row_index, 1].item()),
                    runner_up_confidence=float(top_probabilities[row_index, 1].item()),
                )
            )
            sample_index += 1

    if sample_index != len(sample_ids):
        msg = f"Predicted {sample_index} samples, but the test dataset contains {len(sample_ids)}"
        raise ValueError(msg)
    return predictions


def write_submission(
    predictions: Sequence[Prediction],
    output_path: Path,
    delimiter: str,
) -> None:
    """Write only the two Kaggle columns, with no index or extra metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=delimiter, lineterminator="\n")
        writer.writerow(SUBMISSION_HEADER)
        writer.writerows((prediction.sample_id, prediction.label) for prediction in predictions)


def write_audit_csv(
    predictions: Sequence[Prediction],
    output_path: Path,
    delimiter: str,
) -> None:
    """Write confidence details to a separate file that must not be submitted."""
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=delimiter, lineterminator="\n")
        writer.writerow(
            ("id", "label", "confidence", "runner_up_label", "runner_up_confidence", "margin")
        )
        for prediction in predictions:
            writer.writerow(
                (
                    prediction.sample_id,
                    prediction.label,
                    f"{prediction.confidence:.8f}",
                    prediction.runner_up_label,
                    f"{prediction.runner_up_confidence:.8f}",
                    f"{prediction.margin:.8f}",
                )
            )


def validate_submission(
    submission_path: Path,
    expected_ids: Sequence[str],
    num_classes: int,
    delimiter: str,
) -> ValidationSummary:
    """Validate schema, IDs, ordering, labels, row count, and file integrity."""
    if len(delimiter) != 1:
        msg = "The submission delimiter must be exactly one character"
        raise ValueError(msg)

    with submission_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file, delimiter=delimiter))

    if not rows or tuple(rows[0]) != SUBMISSION_HEADER:
        msg = f"Expected exact header {SUBMISSION_HEADER}, found {rows[0] if rows else 'no header'}"
        raise ValueError(msg)

    data_rows = rows[1:]
    if len(data_rows) != len(expected_ids):
        msg = f"Expected {len(expected_ids)} prediction rows, found {len(data_rows)}"
        raise ValueError(msg)

    actual_ids: list[str] = []
    labels: list[int] = []
    valid_label_tokens = {str(label) for label in range(num_classes)}
    for line_number, row in enumerate(data_rows, start=2):
        if len(row) != 2:
            msg = f"Line {line_number} must contain exactly two columns, found {len(row)}"
            raise ValueError(msg)
        sample_id, label_token = row
        if label_token not in valid_label_tokens:
            msg = (
                f"Line {line_number} has invalid label {label_token!r}; "
                f"expected an integer from 0 to {num_classes - 1}"
            )
            raise ValueError(msg)
        actual_ids.append(sample_id)
        labels.append(int(label_token))

    if len(set(actual_ids)) != len(actual_ids):
        msg = "Submission IDs are not unique"
        raise ValueError(msg)
    if actual_ids != list(expected_ids):
        mismatch_index = next(
            index
            for index, (actual, expected) in enumerate(zip(actual_ids, expected_ids, strict=True))
            if actual != expected
        )
        msg = (
            f"Submission IDs are missing or out of order at data row {mismatch_index + 1}: "
            f"expected {expected_ids[mismatch_index]!r}, found {actual_ids[mismatch_index]!r}"
        )
        raise ValueError(msg)

    histogram = Counter(labels)
    return ValidationSummary(
        row_count=len(data_rows),
        sha256=_sha256(submission_path),
        first_id=actual_ids[0],
        last_id=actual_ids[-1],
        class_histogram={str(label): histogram.get(label, 0) for label in range(num_classes)},
    )


def write_manifest(
    manifest_path: Path,
    settings: Settings,
    checkpoint_path: Path,
    submission_path: Path,
    audit_path: Path,
    summary: ValidationSummary,
) -> None:
    """Persist provenance and validation evidence beside the submission."""
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_name": settings.model_name,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
        "submission": {
            "path": str(submission_path),
            "header": list(SUBMISSION_HEADER),
            "delimiter": settings.submission_delimiter,
            **asdict(summary),
        },
        "audit_csv": {
            "path": str(audit_path),
            "sha256": _sha256(audit_path),
        },
        "external_upload_performed": False,
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")


def generate_submission(settings: Settings) -> tuple[Path, Path, Path, ValidationSummary]:
    """Load a checkpoint, predict the test set, and emit locally audited files."""
    seed_everything(settings.seed)
    checkpoint_path = Path(settings.get_resolved_checkpoint_path())
    submission_path = Path(settings.get_resolved_submission_path())
    if not checkpoint_path.is_file():
        msg = f"Checkpoint does not exist: {checkpoint_path}"
        raise FileNotFoundError(msg)

    dataset, loader = build_test_loader(settings)
    device = settings.resolve_device()
    model = get_model(settings)
    state_dict: dict[str, torch.Tensor] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    predictions = predict(model, loader, dataset.sample_ids, device)

    audit_path = submission_path.with_name(f"{submission_path.stem}_audit.csv")
    manifest_path = submission_path.with_name(f"{submission_path.stem}_manifest.json")
    write_submission(predictions, submission_path, settings.submission_delimiter)
    write_audit_csv(predictions, audit_path, settings.submission_delimiter)
    summary = validate_submission(
        submission_path,
        dataset.sample_ids,
        settings.num_classes,
        settings.submission_delimiter,
    )
    write_manifest(
        manifest_path,
        settings,
        checkpoint_path,
        submission_path,
        audit_path,
        summary,
    )
    return submission_path, audit_path, manifest_path, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or locally validate a TILDA Kaggle CSV. This command never uploads files."
        )
    )
    parser.add_argument(
        "--model", default=None, help="Model architecture (default: settings value)"
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--output", default=None, help="Submission CSV path")
    parser.add_argument("--device", default=None, help="Inference device: auto, cpu, cuda, or mps")
    parser.add_argument(
        "--delimiter",
        choices=[";", ","],
        default=None,
        help="CSV delimiter (default: settings value, currently semicolon)",
    )
    parser.add_argument(
        "--validate-only",
        metavar="CSV_PATH",
        help="Validate an existing CSV without loading a model or running inference",
    )
    return parser


def main() -> int:
    """CLI entry point for local-only Kaggle prediction generation and validation."""
    args = _build_parser().parse_args()
    updates: dict[str, object] = {}
    if args.model is not None:
        updates["model_name"] = args.model
    if args.checkpoint is not None:
        updates["checkpoint_path"] = args.checkpoint
    if args.output is not None:
        updates["submission_path"] = args.output
    if args.device is not None:
        updates["device"] = args.device
    if args.delimiter is not None:
        updates["submission_delimiter"] = args.delimiter
    settings = Settings().model_copy(update=updates)

    try:
        if args.validate_only is not None:
            dataset, _ = build_test_loader(settings)
            summary = validate_submission(
                Path(args.validate_only),
                dataset.sample_ids,
                settings.num_classes,
                settings.submission_delimiter,
            )
            logger.info("VALID: %d rows, SHA-256 %s", summary.row_count, summary.sha256)
            return 0

        submission_path, audit_path, manifest_path, summary = generate_submission(settings)
        logger.info("Submission CSV: %s", submission_path)
        logger.info("Audit CSV (do not submit): %s", audit_path)
        logger.info("Validation manifest: %s", manifest_path)
        logger.info("VALID: %d rows, SHA-256 %s", summary.row_count, summary.sha256)
        logger.info("No file was uploaded; external_upload_performed=false")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
