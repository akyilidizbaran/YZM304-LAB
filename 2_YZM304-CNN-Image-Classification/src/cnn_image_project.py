from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = REPORTS_DIR / "models"
RANDOM_STATE = 42
CLASS_NAMES = [str(i) for i in range(10)]


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    epochs: int = 18
    learning_rate: float = 0.001
    optimizer: str = "Adam"
    loss: str = "CrossEntropyLoss"
    random_state: int = RANDOM_STATE


@dataclass
class ModelResult:
    model: str
    kind: str
    test_accuracy: float
    test_precision_macro: float
    test_recall_macro: float
    test_f1_macro: float
    trainable_parameters: int


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def ensure_dirs() -> None:
    for path in [FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


class LeNetLikeCNN(nn.Module):
    """LeNet-5 benzeri, küçük görüntülere uyarlanmış temel CNN."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_head(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class ImprovedCNN(nn.Module):
    """Aynı temel CNN hiperparametreleri korunarak BatchNorm ve Dropout eklenmiş model."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 64),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=0.35)
        self.classifier = nn.Linear(64, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_head(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.extract_features(x)))


class VGGStyleSmallCNN(nn.Module):
    """VGG blok mantığını küçük 8x8 görüntülere uyarlayan literatür tipi CNN."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 96),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(96, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_head(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def load_preprocessed_digits() -> dict[str, np.ndarray]:
    digits = load_digits()
    images = (digits.images.astype(np.float32) / 16.0)[:, None, :, :]
    labels = digits.target.astype(np.int64)

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        images,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_trainval = (x_trainval - mean) / std
    x_test = (x_test - mean) / std

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_trainval": x_trainval,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_trainval": y_trainval,
        "y_test": y_test,
        "mean": np.array([mean], dtype=np.float32),
        "std": np.array([std], dtype=np.float32),
    }


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_model(
    name: str,
    factory: Callable[[], nn.Module],
    data: dict[str, np.ndarray],
    config: TrainConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]]]:
    model = factory().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(data["x_train"], data["y_train"], config.batch_size, shuffle=True)
    val_loader = make_loader(data["x_val"], data["y_val"], config.batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    for _ in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            total_seen += batch_x.size(0)

        val_loss, val_accuracy = evaluate_loss_accuracy(model, val_loader, loss_fn, device)
        history["train_loss"].append(total_loss / total_seen)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

    torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
    return model, history


@torch.no_grad()
def evaluate_loss_accuracy(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    correct = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        total_loss += loss.item() * batch_x.size(0)
        total_seen += batch_x.size(0)
        correct += (logits.argmax(dim=1) == batch_y).sum().item()
    return total_loss / total_seen, correct / total_seen


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = make_loader(x, np.zeros(len(x), dtype=np.int64), batch_size, shuffle=False)
    preds = []
    for batch_x, _ in loader:
        logits = model(batch_x.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def extract_features(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = make_loader(x, np.zeros(len(x), dtype=np.int64), batch_size, shuffle=False)
    features = []
    for batch_x, _ in loader:
        batch_features = model.extract_features(batch_x.to(device))
        features.append(batch_features.cpu().numpy())
    return np.concatenate(features)


def metric_result(model_name: str, kind: str, y_true: np.ndarray, y_pred: np.ndarray, params: int) -> ModelResult:
    return ModelResult(
        model=model_name,
        kind=kind,
        test_accuracy=accuracy_score(y_true, y_pred),
        test_precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        test_recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        test_f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        trainable_parameters=params,
    )


def save_confusion_matrix(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_learning_curves(histories: dict[str, dict[str, list[float]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, history in histories.items():
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=f"{name} train")
        axes[0].plot(epochs, history["val_loss"], linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, history["val_accuracy"], label=name)
    axes[0].set_title("Loss curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross entropy")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "learning_curves.png", dpi=160)
    plt.close(fig)


def write_results(results: list[ModelResult], histories: dict[str, dict[str, list[float]]], data_summary: dict) -> None:
    csv_path = METRICS_DIR / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    json_payload = {
        "project": "2_YZM304-CNN-Image-Classification",
        "data": data_summary,
        "results": [asdict(result) for result in results],
        "histories": histories,
    }
    (METRICS_DIR / "model_comparison.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    lines = [
        "# Model Comparison",
        "",
        "| Model | Kind | Test Accuracy | Precision Macro | Recall Macro | F1 Macro | Trainable Params |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| `{result.model}` | {result.kind} | {result.test_accuracy:.4f} | "
            f"{result.test_precision_macro:.4f} | {result.test_recall_macro:.4f} | "
            f"{result.test_f1_macro:.4f} | {result.trainable_parameters} |"
        )
    lines.append("")
    (METRICS_DIR / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    seed_everything()
    ensure_dirs()
    config = TrainConfig()
    data = load_preprocessed_digits()
    device = torch.device("cpu")

    model_factories: dict[str, Callable[[], nn.Module]] = {
        "lenet_like_cnn": LeNetLikeCNN,
        "improved_batchnorm_dropout_cnn": ImprovedCNN,
        "vgg_style_small_cnn": VGGStyleSmallCNN,
    }

    trained_models: dict[str, nn.Module] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    results: list[ModelResult] = []

    for name, factory in model_factories.items():
        model, history = train_model(name, factory, data, config, device)
        trained_models[name] = model
        histories[name] = history
        y_pred = predict(model, data["x_test"], config.batch_size, device)
        results.append(metric_result(name, "CNN", data["y_test"], y_pred, count_parameters(model)))
        save_confusion_matrix(name, data["y_test"], y_pred)

    feature_model = trained_models["improved_batchnorm_dropout_cnn"]
    train_features = extract_features(feature_model, data["x_trainval"], config.batch_size, device)
    test_features = extract_features(feature_model, data["x_test"], config.batch_size, device)
    np.save(MODELS_DIR / "train_features.npy", train_features)
    np.save(MODELS_DIR / "train_labels.npy", data["y_trainval"])
    np.save(MODELS_DIR / "test_features.npy", test_features)
    np.save(MODELS_DIR / "test_labels.npy", data["y_test"])

    svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=RANDOM_STATE)
    svm.fit(train_features, data["y_trainval"])
    hybrid_pred = svm.predict(test_features)
    results.append(metric_result("hybrid_improved_cnn_features_svc", "CNN features + SVC", data["y_test"], hybrid_pred, 0))
    save_confusion_matrix("hybrid_improved_cnn_features_svc", data["y_test"], hybrid_pred)

    feature_summary = {
        "train_features_shape": list(train_features.shape),
        "train_labels_shape": list(data["y_trainval"].shape),
        "test_features_shape": list(test_features.shape),
        "test_labels_shape": list(data["y_test"].shape),
        "feature_source": "improved_batchnorm_dropout_cnn.extract_features",
    }
    (MODELS_DIR / "feature_shapes.json").write_text(json.dumps(feature_summary, indent=2), encoding="utf-8")

    data_summary = {
        "dataset": "sklearn.datasets.load_digits",
        "image_shape": [1, 8, 8],
        "classes": 10,
        "samples_total": int(len(data["y_trainval"]) + len(data["y_test"])),
        "train_samples_for_cnn": int(len(data["y_train"])),
        "val_samples_for_cnn": int(len(data["y_val"])),
        "trainval_samples_for_hybrid": int(len(data["y_trainval"])),
        "test_samples": int(len(data["y_test"])),
        "normalization_mean_from_train": float(data["mean"][0]),
        "normalization_std_from_train": float(data["std"][0]),
        "config": asdict(config),
        "feature_shapes": feature_summary,
    }
    (METRICS_DIR / "data_summary.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")

    save_learning_curves(histories)
    write_results(results, histories, data_summary)
    print(json.dumps({"results": [asdict(result) for result in results], "feature_shapes": feature_summary}, indent=2))


if __name__ == "__main__":
    main()
