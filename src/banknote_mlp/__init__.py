"""YZM304 Banknote MLP project package."""

from .config import BASELINE_CONFIG, DEFAULT_DATA_PATH, PROJECT_ROOT
from .data import DataSplit, SplitIndices, apply_split, build_split_indices, load_banknote_dataset
from .manual_mlp import ManualMLPClassifier
from .metrics import ClassificationMetrics, compute_classification_metrics

__all__ = [
    "BASELINE_CONFIG",
    "DEFAULT_DATA_PATH",
    "PROJECT_ROOT",
    "ClassificationMetrics",
    "DataSplit",
    "ManualMLPClassifier",
    "SplitIndices",
    "apply_split",
    "build_split_indices",
    "compute_classification_metrics",
    "load_banknote_dataset",
]

