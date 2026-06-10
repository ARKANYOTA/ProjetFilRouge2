from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute classification accuracy as the fraction of correct predictions."""
    if len(predictions) == 0:
        return 0.0
    return float((predictions == targets).sum() / len(targets))


def compute_disparate_impact(
    predictions: np.ndarray,
    bias_variables: np.ndarray,
) -> float:
    """Compute the Disparate Impact (DI) ratio.

    DI = P(ŷ = 1 | S = 0) / P(ŷ = 1 | S = 1)

    A model is considered unbiased when DI ≈ 1.
    """
    mask_s0 = bias_variables == 0
    mask_s1 = bias_variables == 1

    p_yhat1_s0 = float(predictions[mask_s0].mean()) if mask_s0.any() else 0.0
    p_yhat1_s1 = float(predictions[mask_s1].mean()) if mask_s1.any() else 0.0

    if p_yhat1_s1 == 0.0:
        return float("inf")
    return p_yhat1_s0 / p_yhat1_s1


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_FIGURES_DIR = Path("rapport/figures")
_RESULTS_DIR = Path("results")


def plot_training_curves(
    history: dict[str, list[float] | float],
    model_name: str,
) -> None:
    """Plot and save training/validation loss and accuracy curves."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    assert isinstance(train_loss, list)
    assert isinstance(val_loss, list)
    assert isinstance(train_acc, list)
    assert isinstance(val_acc, list)

    epochs = range(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, train_loss, label="Train loss")
    ax1.plot(epochs, val_loss, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_acc, label="Train acc")
    ax2.plot(epochs, val_acc, label="Val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    for d in (_FIGURES_DIR, _RESULTS_DIR):
        fig.savefig(d / f"{model_name}_training_curves.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(all_results: dict[str, dict[str, object]]) -> None:
    """Bar chart comparing val accuracy and training time across models."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    names = list(all_results.keys())
    val_accs = []
    train_times = []
    for name in names:
        res = all_results[name]
        va = res["val_acc"]
        assert isinstance(va, list)
        val_accs.append(max(va))
        tt = res["training_time"]
        assert isinstance(tt, float | int)
        train_times.append(tt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colours = ["#4C72B0", "#55A868", "#C44E52"]
    ax1.bar(names, [a * 100 for a in val_accs], color=colours[: len(names)])
    ax1.set_ylabel("Best Val Accuracy (%)")
    ax1.set_title("Model Comparison — Accuracy")
    ax1.set_ylim(0, 100)
    for i, v in enumerate(val_accs):
        ax1.text(i, v * 100 + 1, f"{v * 100:.1f}%", ha="center", fontweight="bold")

    ax2.bar(names, train_times, color=colours[: len(names)])
    ax2.set_ylabel("Training Time (s)")
    ax2.set_title("Model Comparison — Training Time")
    for i, v in enumerate(train_times):
        ax2.text(i, v + 1, f"{v:.0f}s", ha="center", fontweight="bold")

    fig.tight_layout()
    for d in (_FIGURES_DIR, _RESULTS_DIR):
        fig.savefig(d / "model_comparison.png", dpi=150)
    plt.close(fig)


def export_results_json(
    all_results: dict[str, dict[str, object]],
) -> None:
    """Persist experiment results to ``results/comparison_results.json``."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, object]] = {}
    for name, res in all_results.items():
        va = res.get("val_acc")
        ta = res.get("train_acc")
        assert isinstance(va, list)
        assert isinstance(ta, list)
        out[name] = {
            "best_val_acc": max(va),
            "best_train_acc": max(ta),
            "training_time_s": res.get("training_time"),
            "epochs_trained": len(va),
        }
    path = _RESULTS_DIR / "comparison_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def plot_bias_evaluation(
    accuracy_results: dict[str, float],
    di_results: dict[str, float],
) -> None:
    """Plot and save comparison figures for model bias and accuracy (Part 3)."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    names = list(accuracy_results.keys())
    accs = [accuracy_results[n] for n in names]
    dis = [di_results[n] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colours = ["#4C72B0", "#C44E52"]

    # Accuracy Plot
    ax1.bar(names, [a * 100 for a in accs], color=colours[: len(names)])
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("Safety Experiment — Binary Accuracy")
    ax1.set_ylim(0, 100)
    for i, v in enumerate(accs):
        ax1.text(i, v * 100 + 1, f"{v * 100:.1f}%", ha="center", fontweight="bold")

    # DI Plot (clip infinite/very large DI values for display)
    clipped_dis = [d if d != float("inf") else 10.0 for d in dis]
    ax2.bar(names, clipped_dis, color=colours[: len(names)])
    ax2.set_ylabel("Disparate Impact (DI) Metric")
    ax2.set_title("Safety Experiment — Disparate Impact")
    ax2.axhline(y=1.0, color="gray", linestyle="--", label="Perfect Fairness (DI = 1.0)")
    ax2.axhline(y=0.8, color="red", linestyle=":", label="Fairness Threshold (0.8 - 1.25)")
    ax2.axhline(y=1.25, color="red", linestyle=":")
    ax2.legend()

    for i, v in enumerate(dis):
        val_str = f"{v:.2f}" if v != float("inf") else "inf"
        y_pos = min(clipped_dis[i], 8.0) + 0.1
        ax2.text(i, y_pos, val_str, ha="center", fontweight="bold")

    fig.tight_layout()
    for d in (_FIGURES_DIR, _RESULTS_DIR):
        fig.savefig(d / "bias_evaluation.png", dpi=150)
    plt.close(fig)
