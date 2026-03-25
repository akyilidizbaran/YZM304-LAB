from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    support: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confusion_matrix"] = [list(row) for row in self.confusion_matrix]
        return data


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    y_true = y_true.astype(np.int64).ravel()
    y_pred = y_pred.astype(np.int64).ravel()

    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))

    accuracy = (true_positive + true_negative) / len(y_true)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_score),
        confusion_matrix=((true_negative, false_positive), (false_negative, true_positive)),
        support=int(len(y_true)),
    )


def save_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def save_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown_summary(path: Path, rows: list[dict[str, Any]], best_experiment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment Summary",
        "",
        f"Best experiment: `{best_experiment}`",
        "",
        "| Experiment | Backend | Train Fraction | Standardized | Hidden Layers | Steps | Params | Val Acc | Test Acc | Test F1 |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {experiment_name} | {backend} | {train_fraction:.2f} | {standardize} | {hidden_layers} | {steps} | "
            "{parameter_count} | {validation_accuracy:.4f} | {test_accuracy:.4f} | {test_f1_score:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_confusion_matrix(
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]],
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(confusion_matrix)

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_learning_curves(histories: dict[str, dict[str, list[float | int]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))

    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        if history["train_loss"] and len(history["step"]) == len(history["train_loss"]):
            plt.plot(history["step"], history["train_loss"], label=f"{name} train")
        if history["validation_loss"]:
            plt.plot(history["step"], history["validation_loss"], linestyle="--", label=f"{name} val")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        if history["train_accuracy"] and len(history["step"]) == len(history["train_accuracy"]):
            plt.plot(history["step"], history["train_accuracy"], label=f"{name} train")
        if history["validation_accuracy"]:
            plt.plot(history["step"], history["validation_accuracy"], linestyle="--", label=f"{name} val")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
