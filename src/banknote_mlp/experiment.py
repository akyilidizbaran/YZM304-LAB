from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import BASELINE_CONFIG, PROJECT_ROOT
from .data import apply_split, build_split_indices, class_balance, load_banknote_dataset
from .manual_mlp import ManualMLPClassifier
from .metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_learning_curves,
    save_comparison_csv,
    save_json,
    save_markdown_summary,
)
from .sklearn_models import train_sklearn_mlp


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_type: str
    hidden_layers: tuple[int, ...]
    steps: int
    learning_rate: float
    standardize: bool
    l2_lambda: float = 0.0


def run_experiments() -> dict[str, Any]:
    X, y, feature_names = load_banknote_dataset()
    split_indices = build_split_indices(
        y,
        test_size=BASELINE_CONFIG.test_size,
        validation_size=BASELINE_CONFIG.validation_size,
        random_state=BASELINE_CONFIG.random_state,
    )

    experiments = [
        ExperimentConfig(
            name="manual_raw_baseline",
            model_type="manual",
            hidden_layers=BASELINE_CONFIG.hidden_layers,
            steps=BASELINE_CONFIG.n_steps,
            learning_rate=BASELINE_CONFIG.learning_rate,
            standardize=False,
        ),
        ExperimentConfig(
            name="manual_standardized_baseline",
            model_type="manual",
            hidden_layers=BASELINE_CONFIG.hidden_layers,
            steps=1000,
            learning_rate=0.03,
            standardize=True,
        ),
        ExperimentConfig(
            name="manual_regularized_deeper",
            model_type="manual",
            hidden_layers=(10, 6),
            steps=1000,
            learning_rate=0.01,
            standardize=True,
            l2_lambda=0.001,
        ),
        ExperimentConfig(
            name="sklearn_standardized_baseline",
            model_type="sklearn",
            hidden_layers=BASELINE_CONFIG.hidden_layers,
            steps=1000,
            learning_rate=0.01,
            standardize=True,
        ),
    ]

    results: list[dict[str, Any]] = []
    histories: dict[str, dict[str, list[float | int]]] = {}
    confusion_targets: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}

    for experiment in experiments:
        split = apply_split(
            X,
            y,
            feature_names,
            split_indices,
            standardize=experiment.standardize,
        )
        if experiment.model_type == "manual":
            model = ManualMLPClassifier(
                hidden_layers=experiment.hidden_layers,
                learning_rate=experiment.learning_rate,
                n_steps=experiment.steps,
                l2_lambda=experiment.l2_lambda,
                threshold=BASELINE_CONFIG.threshold,
                random_state=BASELINE_CONFIG.random_state,
            )
            model.fit(
                split.X_train,
                split.y_train,
                split.X_validation,
                split.y_validation,
            )
            train_predictions = model.predict(split.X_train)
            validation_predictions = model.predict(split.X_validation)
            test_predictions = model.predict(split.X_test)
            histories[experiment.name] = model.history_
        else:
            model = train_sklearn_mlp(
                split.X_train,
                split.y_train,
                hidden_layers=experiment.hidden_layers,
                learning_rate=experiment.learning_rate,
                max_iter=experiment.steps,
                l2_lambda=experiment.l2_lambda,
                random_state=BASELINE_CONFIG.random_state,
            )
            old_settings = np.seterr(over="ignore", invalid="ignore", divide="ignore")
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    train_predictions = model.predict(split.X_train)
                    validation_predictions = model.predict(split.X_validation)
                    test_predictions = model.predict(split.X_test)
            finally:
                np.seterr(**old_settings)
            histories[experiment.name] = {
                "step": list(range(1, len(model.loss_curve_) + 1)),
                "train_loss": [float(value) for value in model.loss_curve_],
                "validation_loss": [],
                "train_accuracy": [],
                "validation_accuracy": [],
            }

        train_metrics = compute_classification_metrics(split.y_train, train_predictions)
        validation_metrics = compute_classification_metrics(split.y_validation, validation_predictions)
        test_metrics = compute_classification_metrics(split.y_test, test_predictions)

        result = {
            "experiment_name": experiment.name,
            "model_type": experiment.model_type,
            "standardize": experiment.standardize,
            "hidden_layers": "-".join(str(unit) for unit in experiment.hidden_layers),
            "steps": experiment.steps,
            "learning_rate": experiment.learning_rate,
            "l2_lambda": experiment.l2_lambda,
            "train_accuracy": train_metrics.accuracy,
            "validation_accuracy": validation_metrics.accuracy,
            "test_accuracy": test_metrics.accuracy,
            "test_precision": test_metrics.precision,
            "test_recall": test_metrics.recall,
            "test_f1_score": test_metrics.f1_score,
        }
        results.append(result)
        confusion_targets[experiment.name] = test_metrics.confusion_matrix

    results.sort(key=lambda row: (-row["validation_accuracy"], row["steps"], -row["test_accuracy"]))
    best_experiment = results[0]["experiment_name"]

    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    models_dir = PROJECT_ROOT / "reports" / "models"

    comparison_csv = metrics_dir / "experiment_comparison.csv"
    summary_json = metrics_dir / "experiment_summary.json"
    summary_md = metrics_dir / "experiment_summary.md"
    split_json = models_dir / "data_split_summary.json"

    save_comparison_csv(comparison_csv, results)
    save_markdown_summary(summary_md, results, best_experiment)
    save_json(
        summary_json,
        {
            "project": "YZM304-Banknote-MLP",
            "selection_rule": BASELINE_CONFIG.selection_rule,
            "best_experiment": best_experiment,
            "results": results,
        },
    )
    save_json(
        split_json,
        {
            "feature_names": list(feature_names),
            "class_balance_total": class_balance(y),
            "split_sizes": {
                "train": int(len(split_indices.train)),
                "validation": int(len(split_indices.validation)),
                "test": int(len(split_indices.test)),
            },
        },
    )

    for experiment_name, confusion_matrix in confusion_targets.items():
        plot_confusion_matrix(
            confusion_matrix,
            title=f"Confusion Matrix - {experiment_name}",
            output_path=figures_dir / f"{experiment_name}_confusion_matrix.png",
        )

    plot_learning_curves(histories, figures_dir / "learning_curves.png")

    return {
        "best_experiment": best_experiment,
        "results": results,
        "artifacts": {
            "comparison_csv": str(comparison_csv),
            "summary_json": str(summary_json),
            "summary_markdown": str(summary_md),
            "split_json": str(split_json),
        },
    }


def main() -> None:
    summary = run_experiments()
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
