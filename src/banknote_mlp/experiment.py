from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import BASELINE_CONFIG, PROJECT_ROOT, SPLITS_DIR, WEIGHTS_DIR
from .data import (
    apply_split,
    build_split_indices,
    build_train_fraction_indices,
    class_balance,
    load_banknote_dataset,
    save_split_manifest,
)
from .manual_mlp import ManualMLPClassifier
from .metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_learning_curves,
    save_comparison_csv,
    save_json,
    save_markdown_summary,
)
from .pytorch_backend import predict_torch_mlp, train_torch_mlp
from .shared_artifacts import architecture_key, load_initial_parameters, save_initial_parameters
from .sklearn_models import predict_sklearn_mlp, train_sklearn_mlp


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    backend: str
    hidden_layers: tuple[int, ...]
    steps: int
    learning_rate: float
    standardize: bool
    l2_lambda: float = 0.0
    train_fraction: float = 1.0


def _parameter_count(input_size: int, hidden_layers: tuple[int, ...], output_size: int = 1) -> int:
    layer_sizes = (input_size, *hidden_layers, output_size)
    return int(sum((fan_in * fan_out) + fan_out for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:])))


def _torch_is_available() -> bool:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _build_initial_weight_artifacts(input_size: int) -> dict[tuple[int, ...], dict[str, str]]:
    artifacts: dict[tuple[int, ...], dict[str, str]] = {}
    for hidden_layers in {BASELINE_CONFIG.hidden_layers, (10, 6)}:
        key = architecture_key(input_size, hidden_layers)
        initial_parameters = save_initial_parameters(
            WEIGHTS_DIR,
            key=key,
            layer_sizes=(input_size, *hidden_layers, 1),
            random_state=BASELINE_CONFIG.random_state,
        )
        artifacts[hidden_layers] = {
            "npz_path": str(initial_parameters.npz_path),
            "metadata_path": str(initial_parameters.metadata_path),
        }
    return artifacts


def run_experiments() -> dict[str, Any]:
    X, y, feature_names = load_banknote_dataset()
    split_indices = build_split_indices(
        y,
        test_size=BASELINE_CONFIG.test_size,
        validation_size=BASELINE_CONFIG.validation_size,
        random_state=BASELINE_CONFIG.random_state,
    )
    train_fraction_indices = build_train_fraction_indices(
        y,
        split_indices.train,
        fractions=BASELINE_CONFIG.train_fractions,
        random_state=BASELINE_CONFIG.random_state,
    )
    save_split_manifest(
        SPLITS_DIR / "split_manifest.json",
        feature_names=feature_names,
        y=y,
        split_indices=split_indices,
        train_fraction_indices=train_fraction_indices,
    )
    weight_artifacts = _build_initial_weight_artifacts(X.shape[1])
    torch_available = _torch_is_available()

    experiments = [
        ExperimentConfig(
            name="manual_raw_baseline",
            backend="manual",
            hidden_layers=BASELINE_CONFIG.hidden_layers,
            steps=BASELINE_CONFIG.n_steps,
            learning_rate=BASELINE_CONFIG.learning_rate,
            standardize=False,
        ),
        ExperimentConfig(
            name="manual_standardized_baseline",
            backend="manual",
            hidden_layers=BASELINE_CONFIG.hidden_layers,
            steps=1000,
            learning_rate=0.03,
            standardize=True,
        ),
        ExperimentConfig(
            name="sklearn_standardized_baseline",
            backend="sklearn",
            hidden_layers=BASELINE_CONFIG.hidden_layers,
            steps=1000,
            learning_rate=0.03,
            standardize=True,
        ),
        ExperimentConfig(
            name="pytorch_standardized_baseline",
            backend="pytorch",
            hidden_layers=BASELINE_CONFIG.hidden_layers,
            steps=1000,
            learning_rate=0.03,
            standardize=True,
        ),
        ExperimentConfig(
            name="manual_regularized_deeper_data50",
            backend="manual",
            hidden_layers=(10, 6),
            steps=1000,
            learning_rate=0.01,
            standardize=True,
            l2_lambda=0.001,
            train_fraction=0.5,
        ),
        ExperimentConfig(
            name="manual_regularized_deeper_data75",
            backend="manual",
            hidden_layers=(10, 6),
            steps=1000,
            learning_rate=0.01,
            standardize=True,
            l2_lambda=0.001,
            train_fraction=0.75,
        ),
        ExperimentConfig(
            name="manual_regularized_deeper_data100",
            backend="manual",
            hidden_layers=(10, 6),
            steps=1000,
            learning_rate=0.01,
            standardize=True,
            l2_lambda=0.001,
            train_fraction=1.0,
        ),
        ExperimentConfig(
            name="sklearn_regularized_deeper",
            backend="sklearn",
            hidden_layers=(10, 6),
            steps=1000,
            learning_rate=0.01,
            standardize=True,
            l2_lambda=0.001,
        ),
        ExperimentConfig(
            name="pytorch_regularized_deeper",
            backend="pytorch",
            hidden_layers=(10, 6),
            steps=1000,
            learning_rate=0.01,
            standardize=True,
            l2_lambda=0.001,
        ),
    ]

    results: list[dict[str, Any]] = []
    histories: dict[str, dict[str, list[float | int]]] = {}
    confusion_targets: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    skipped_experiments: list[str] = []

    for experiment in experiments:
        if experiment.backend == "pytorch" and not torch_available:
            skipped_experiments.append(experiment.name)
            continue

        split = apply_split(
            X,
            y,
            feature_names,
            split_indices,
            train_indices_override=train_fraction_indices[f"{experiment.train_fraction:.2f}"],
            standardize=experiment.standardize,
        )
        initial_weights, initial_biases = load_initial_parameters(Path(weight_artifacts[experiment.hidden_layers]["npz_path"]))

        if experiment.backend == "manual":
            model = ManualMLPClassifier(
                hidden_layers=experiment.hidden_layers,
                learning_rate=experiment.learning_rate,
                n_steps=experiment.steps,
                l2_lambda=experiment.l2_lambda,
                threshold=BASELINE_CONFIG.threshold,
                random_state=BASELINE_CONFIG.random_state,
                initial_weights=initial_weights,
                initial_biases=initial_biases,
            )
            model.fit(split.X_train, split.y_train, split.X_validation, split.y_validation)
            train_predictions = model.predict(split.X_train)
            validation_predictions = model.predict(split.X_validation)
            test_predictions = model.predict(split.X_test)
            histories[experiment.name] = model.history_
            parameter_count = model.parameter_count
        elif experiment.backend == "sklearn":
            model, history = train_sklearn_mlp(
                split.X_train,
                split.y_train,
                X_validation=split.X_validation,
                y_validation=split.y_validation,
                hidden_layers=experiment.hidden_layers,
                learning_rate=experiment.learning_rate,
                max_iter=experiment.steps,
                l2_lambda=experiment.l2_lambda,
                random_state=BASELINE_CONFIG.random_state,
                threshold=BASELINE_CONFIG.threshold,
                initial_weights=initial_weights,
                initial_biases=initial_biases,
            )
            train_predictions = predict_sklearn_mlp(model, split.X_train)
            validation_predictions = predict_sklearn_mlp(model, split.X_validation)
            test_predictions = predict_sklearn_mlp(model, split.X_test)
            histories[experiment.name] = history
            parameter_count = _parameter_count(X.shape[1], experiment.hidden_layers)
        else:
            model, history = train_torch_mlp(
                split.X_train,
                split.y_train,
                X_validation=split.X_validation,
                y_validation=split.y_validation,
                hidden_layers=experiment.hidden_layers,
                learning_rate=experiment.learning_rate,
                max_iter=experiment.steps,
                l2_lambda=experiment.l2_lambda,
                random_state=BASELINE_CONFIG.random_state,
                threshold=BASELINE_CONFIG.threshold,
                initial_weights=initial_weights,
                initial_biases=initial_biases,
            )
            train_predictions = predict_torch_mlp(model, split.X_train, threshold=BASELINE_CONFIG.threshold)
            validation_predictions = predict_torch_mlp(model, split.X_validation, threshold=BASELINE_CONFIG.threshold)
            test_predictions = predict_torch_mlp(model, split.X_test, threshold=BASELINE_CONFIG.threshold)
            histories[experiment.name] = history
            parameter_count = model.parameter_count

        train_metrics = compute_classification_metrics(split.y_train, train_predictions)
        validation_metrics = compute_classification_metrics(split.y_validation, validation_predictions)
        test_metrics = compute_classification_metrics(split.y_test, test_predictions)

        results.append(
            {
                "experiment_name": experiment.name,
                "backend": experiment.backend,
                "standardize": experiment.standardize,
                "train_fraction": experiment.train_fraction,
                "hidden_layers": "-".join(str(unit) for unit in experiment.hidden_layers),
                "steps": experiment.steps,
                "learning_rate": experiment.learning_rate,
                "l2_lambda": experiment.l2_lambda,
                "parameter_count": parameter_count,
                "weight_file": str(Path(weight_artifacts[experiment.hidden_layers]["npz_path"]).relative_to(PROJECT_ROOT)),
                "train_accuracy": train_metrics.accuracy,
                "validation_accuracy": validation_metrics.accuracy,
                "test_accuracy": test_metrics.accuracy,
                "test_precision": test_metrics.precision,
                "test_recall": test_metrics.recall,
                "test_f1_score": test_metrics.f1_score,
            }
        )
        confusion_targets[experiment.name] = test_metrics.confusion_matrix

    results.sort(key=lambda row: (-row["validation_accuracy"], row["steps"], row["parameter_count"], -row["test_accuracy"]))
    best_experiment = results[0]["experiment_name"]

    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    models_dir = PROJECT_ROOT / "reports" / "models"

    comparison_csv = metrics_dir / "experiment_comparison.csv"
    summary_json = metrics_dir / "experiment_summary.json"
    summary_md = metrics_dir / "experiment_summary.md"
    split_json = models_dir / "data_split_summary.json"
    backend_csv = metrics_dir / "backend_comparison.csv"

    backend_rows = [
        row
        for row in results
        if row["experiment_name"]
        in {
            "manual_standardized_baseline",
            "sklearn_standardized_baseline",
            "pytorch_standardized_baseline",
            "manual_regularized_deeper_data100",
            "sklearn_regularized_deeper",
            "pytorch_regularized_deeper",
        }
    ]

    save_comparison_csv(comparison_csv, results)
    save_comparison_csv(backend_csv, backend_rows)
    save_markdown_summary(summary_md, results, best_experiment)
    save_json(
        summary_json,
        {
            "project": "YZM304-Banknote-MLP",
            "selection_rule": BASELINE_CONFIG.selection_rule,
            "best_experiment": best_experiment,
            "results": results,
            "backend_comparison": backend_rows,
            "skipped_experiments": skipped_experiments,
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
            "train_fraction_sizes": {
                key: int(len(indices)) for key, indices in train_fraction_indices.items()
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
            "backend_csv": str(backend_csv),
            "summary_json": str(summary_json),
            "summary_markdown": str(summary_md),
            "split_json": str(split_json),
            "split_manifest": str(SPLITS_DIR / "split_manifest.json"),
        },
        "skipped_experiments": skipped_experiments,
    }


def main() -> None:
    summary = run_experiments()
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
