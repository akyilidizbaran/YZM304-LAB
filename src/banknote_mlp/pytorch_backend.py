from __future__ import annotations

import numpy as np


def train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    max_iter: int,
    l2_lambda: float,
    random_state: int,
    threshold: float,
    initial_weights: list[np.ndarray],
    initial_biases: list[np.ndarray],
):
    import torch
    from torch import nn

    class TorchMLP(nn.Module):
        def __init__(self, input_size: int, hidden_layers_: tuple[int, ...]) -> None:
            super().__init__()
            layer_sizes = (input_size, *hidden_layers_, 1)
            self.layers = nn.ModuleList(
                [nn.Linear(in_features, out_features) for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:])]
            )

        def forward(self, inputs):
            current = inputs
            for layer in self.layers[:-1]:
                current = torch.tanh(layer(current))
            return self.layers[-1](current).squeeze(-1)

        @property
        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())

    torch.manual_seed(random_state)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    model = TorchMLP(X_train.shape[1], hidden_layers).double()
    with torch.no_grad():
        for layer, weight, bias in zip(model.layers, initial_weights, initial_biases):
            layer.weight.copy_(torch.tensor(weight.T, dtype=torch.float64))
            layer.bias.copy_(torch.tensor(bias, dtype=torch.float64))

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0)
    criterion = torch.nn.BCEWithLogitsLoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float64)
    X_validation_tensor = torch.tensor(X_validation, dtype=torch.float64)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.float64)

    history = {
        "step": [],
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }

    for step in range(1, max_iter + 1):
        optimizer.zero_grad()
        train_logits = model(X_train_tensor)
        train_loss = criterion(train_logits, y_train_tensor)
        if l2_lambda > 0.0:
            l2_penalty = sum(layer.weight.pow(2).sum() for layer in model.layers)
            train_loss = train_loss + (l2_lambda / (2.0 * X_train_tensor.shape[0])) * l2_penalty
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            validation_logits = model(X_validation_tensor)
            validation_loss = criterion(validation_logits, y_validation_tensor)
            if l2_lambda > 0.0:
                validation_penalty = sum(layer.weight.pow(2).sum() for layer in model.layers)
                validation_loss = validation_loss + (l2_lambda / (2.0 * X_train_tensor.shape[0])) * validation_penalty
            train_probabilities = torch.sigmoid(train_logits).cpu().numpy()
            validation_probabilities = torch.sigmoid(validation_logits).cpu().numpy()
            train_predictions = (train_probabilities >= threshold).astype(np.int64)
            validation_predictions = (validation_probabilities >= threshold).astype(np.int64)

        history["step"].append(step)
        history["train_loss"].append(float(train_loss.item()))
        history["validation_loss"].append(float(validation_loss.item()))
        history["train_accuracy"].append(float(np.mean(train_predictions == y_train)))
        history["validation_accuracy"].append(float(np.mean(validation_predictions == y_validation)))

    return model, history


def predict_torch_mlp(model, X: np.ndarray, *, threshold: float) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.tensor(X, dtype=torch.float64))).cpu().numpy()
    return (probabilities >= threshold).astype(np.int64).ravel()
