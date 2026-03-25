from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DEFAULT_DATA_PATH


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    mean: np.ndarray | None
    std: np.ndarray | None


def load_banknote_dataset(path: Path | str = DEFAULT_DATA_PATH) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    feature_names = tuple(name for name in reader.fieldnames or () if name != "class")
    X = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)
    y = np.array([int(row["class"]) for row in rows], dtype=np.int64)
    return X, y, feature_names


def build_split_indices(
    y: np.ndarray,
    *,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> SplitIndices:
    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        shuffled = rng.permutation(label_indices)

        n_total = len(shuffled)
        n_test = int(round(n_total * test_size))
        remaining = n_total - n_test
        n_validation = int(round(remaining * validation_size))

        test_parts.append(shuffled[:n_test])
        validation_parts.append(shuffled[n_test : n_test + n_validation])
        train_parts.append(shuffled[n_test + n_validation :])

    train = np.concatenate(train_parts)
    validation = np.concatenate(validation_parts)
    test = np.concatenate(test_parts)

    return SplitIndices(
        train=np.sort(train),
        validation=np.sort(validation),
        test=np.sort(test),
    )


def apply_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    split_indices: SplitIndices,
    *,
    train_indices_override: np.ndarray | None = None,
    standardize: bool,
) -> DataSplit:
    train_indices = split_indices.train if train_indices_override is None else np.asarray(train_indices_override, dtype=np.int64)
    X_train = X[train_indices].copy()
    X_validation = X[split_indices.validation].copy()
    X_test = X[split_indices.test].copy()
    y_train = y[train_indices].copy()
    y_validation = y[split_indices.validation].copy()
    y_test = y[split_indices.test].copy()

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    if standardize:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)

        X_train = (X_train - mean) / std
        X_validation = (X_validation - mean) / std
        X_test = (X_test - mean) / std

    return DataSplit(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        mean=mean,
        std=std,
    )


def class_balance(y: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def class_balance_for_indices(y: np.ndarray, indices: np.ndarray) -> dict[int, int]:
    return class_balance(y[np.asarray(indices, dtype=np.int64)])


def build_train_fraction_indices(
    y: np.ndarray,
    train_indices: np.ndarray,
    *,
    fractions: tuple[float, ...],
    random_state: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    outputs: dict[str, np.ndarray] = {}
    per_label_orders = {
        int(label): rng.permutation(train_indices[y[train_indices] == label])
        for label in np.unique(y[train_indices])
    }

    for fraction in fractions:
        if fraction >= 1.0:
            outputs[f"{fraction:.2f}"] = np.sort(train_indices.copy())
            continue

        subset_parts: list[np.ndarray] = []
        for label, shuffled in per_label_orders.items():
            take_count = max(1, int(round(len(shuffled) * fraction)))
            subset_parts.append(shuffled[:take_count])

        outputs[f"{fraction:.2f}"] = np.sort(np.concatenate(subset_parts))

    return outputs


def save_split_manifest(
    path: Path,
    *,
    feature_names: tuple[str, ...],
    y: np.ndarray,
    split_indices: SplitIndices,
    train_fraction_indices: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": list(feature_names),
        "class_balance_total": class_balance(y),
        "split_sizes": {
            "train": int(len(split_indices.train)),
            "validation": int(len(split_indices.validation)),
            "test": int(len(split_indices.test)),
        },
        "split_indices": {
            "train": split_indices.train.astype(int).tolist(),
            "validation": split_indices.validation.astype(int).tolist(),
            "test": split_indices.test.astype(int).tolist(),
        },
        "train_fraction_sizes": {key: int(len(indices)) for key, indices in train_fraction_indices.items()},
        "train_fraction_indices": {
            key: np.asarray(indices, dtype=np.int64).astype(int).tolist()
            for key, indices in train_fraction_indices.items()
        },
        "train_fraction_class_balance": {
            key: class_balance_for_indices(y, indices)
            for key, indices in train_fraction_indices.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
