"""YZM304 Banknote MLP project package."""

from .config import BASELINE_CONFIG, DATA_DIR, DEFAULT_DATA_PATH, PROJECT_ROOT, SPLITS_DIR, WEIGHTS_DIR
from .data import (
    DataSplit,
    SplitIndices,
    apply_split,
    build_split_indices,
    build_train_fraction_indices,
    load_banknote_dataset,
    save_split_manifest,
)
from .manual_mlp import ManualMLPClassifier
from .metrics import ClassificationMetrics, compute_classification_metrics
from .shared_artifacts import InitialParameters, architecture_key, load_initial_parameters, save_initial_parameters

__all__ = [
    "BASELINE_CONFIG",
    "DATA_DIR",
    "DEFAULT_DATA_PATH",
    "InitialParameters",
    "PROJECT_ROOT",
    "ClassificationMetrics",
    "DataSplit",
    "ManualMLPClassifier",
    "SPLITS_DIR",
    "SplitIndices",
    "WEIGHTS_DIR",
    "apply_split",
    "architecture_key",
    "build_split_indices",
    "build_train_fraction_indices",
    "compute_classification_metrics",
    "load_initial_parameters",
    "load_banknote_dataset",
    "save_initial_parameters",
    "save_split_manifest",
]
