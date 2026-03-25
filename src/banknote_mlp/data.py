from __future__ import annotations

import csv
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
    standardize: bool,
) -> DataSplit:
    X_train = X[split_indices.train].copy()
    X_validation = X[split_indices.validation].copy()
    X_test = X[split_indices.test].copy()
    y_train = y[split_indices.train].copy()
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
