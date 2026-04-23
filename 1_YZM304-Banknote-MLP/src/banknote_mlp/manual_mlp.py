from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class ManualMLPClassifier:
    hidden_layers: tuple[int, ...] = (6,)
    learning_rate: float = 0.01
    n_steps: int = 500
    l2_lambda: float = 0.0
    threshold: float = 0.5
    random_state: int = 42
    weight_scale: float = 1.0
    gradient_clip_value: float | None = 5.0
    weight_clip_value: float | None = 10.0
    initial_weights: list[np.ndarray] | None = None
    initial_biases: list[np.ndarray] | None = None
    history_: dict[str, list[float | int]] = field(default_factory=dict, init=False)
    weights_: list[np.ndarray] = field(default_factory=list, init=False)
    biases_: list[np.ndarray] = field(default_factory=list, init=False)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray | None = None,
        y_validation: np.ndarray | None = None,
    ) -> "ManualMLPClassifier":
        y_train_2d = y_train.reshape(-1, 1).astype(np.float64)
        y_validation_2d = None if y_validation is None else y_validation.reshape(-1, 1).astype(np.float64)

        self._initialize_parameters(X_train.shape[1])
        self.history_ = {
            "step": [],
            "train_loss": [],
            "validation_loss": [],
            "train_accuracy": [],
            "validation_accuracy": [],
        }

        for step in range(1, self.n_steps + 1):
            activations, linear_outputs = self._forward(X_train)
            predictions = activations[-1]
            gradients = self._backward(activations, linear_outputs, y_train_2d)
            self._apply_gradients(gradients)

            train_loss = self._compute_loss(predictions, y_train_2d)
            train_accuracy = self._accuracy(y_train_2d, predictions)

            self.history_["step"].append(step)
            self.history_["train_loss"].append(float(train_loss))
            self.history_["train_accuracy"].append(float(train_accuracy))

            if X_validation is not None and y_validation_2d is not None:
                validation_predictions = self.predict_proba(X_validation)
                validation_loss = self._compute_loss(validation_predictions, y_validation_2d)
                validation_accuracy = self._accuracy(y_validation_2d, validation_predictions)
                self.history_["validation_loss"].append(float(validation_loss))
                self.history_["validation_accuracy"].append(float(validation_accuracy))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(np.int64).ravel()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def _initialize_parameters(self, input_size: int) -> None:
        if self.initial_weights is not None and self.initial_biases is not None:
            self.weights_ = [np.array(weight, dtype=np.float64, copy=True) for weight in self.initial_weights]
            self.biases_ = [np.array(bias, dtype=np.float64, copy=True).reshape(-1) for bias in self.initial_biases]
            return

        rng = np.random.default_rng(self.random_state)
        layer_sizes = (input_size, *self.hidden_layers, 1)
        self.weights_ = []
        self.biases_ = []

        for n_inputs, n_outputs in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = self.weight_scale * np.sqrt(2.0 / (n_inputs + n_outputs))
            self.weights_.append(rng.normal(loc=0.0, scale=scale, size=(n_inputs, n_outputs)))
            self.biases_.append(np.zeros((n_outputs,), dtype=np.float64))

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        linear_outputs: list[np.ndarray] = []
        current = X

        for layer_idx, (weights, biases) in enumerate(zip(self.weights_, self.biases_)):
            safe_current = np.nan_to_num(current, nan=0.0, posinf=self.weight_clip_value or 0.0, neginf=-(self.weight_clip_value or 0.0))
            safe_weights = np.nan_to_num(weights, nan=0.0, posinf=self.weight_clip_value or 0.0, neginf=-(self.weight_clip_value or 0.0))
            safe_biases = np.nan_to_num(biases, nan=0.0, posinf=self.weight_clip_value or 0.0, neginf=-(self.weight_clip_value or 0.0))
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                linear = safe_current @ safe_weights + safe_biases
            linear = np.nan_to_num(linear, nan=0.0, posinf=self.weight_clip_value or 0.0, neginf=-(self.weight_clip_value or 0.0))
            linear_outputs.append(linear)

            if layer_idx == len(self.weights_) - 1:
                current = _sigmoid(linear)
            else:
                current = np.tanh(linear)

            activations.append(current)

        return activations, linear_outputs

    def _backward(
        self,
        activations: list[np.ndarray],
        linear_outputs: list[np.ndarray],
        y_true: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        del linear_outputs

        sample_count = y_true.shape[0]
        gradients: list[tuple[np.ndarray, np.ndarray]] = []
        delta = activations[-1] - y_true

        for layer_idx in reversed(range(len(self.weights_))):
            previous_activation = activations[layer_idx]
            weights = self.weights_[layer_idx]

            safe_activation = np.nan_to_num(
                previous_activation,
                nan=0.0,
                posinf=self.weight_clip_value or 0.0,
                neginf=-(self.weight_clip_value or 0.0),
            )
            safe_delta = np.nan_to_num(
                delta,
                nan=0.0,
                posinf=self.gradient_clip_value or 0.0,
                neginf=-(self.gradient_clip_value or 0.0),
            )
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                gradient_w = (safe_activation.T @ safe_delta) / sample_count
            if self.l2_lambda > 0.0:
                gradient_w += (self.l2_lambda / sample_count) * weights
            gradient_w = np.nan_to_num(
                gradient_w,
                nan=0.0,
                posinf=self.gradient_clip_value or 0.0,
                neginf=-(self.gradient_clip_value or 0.0),
            )
            gradient_b = np.nanmean(safe_delta, axis=0)
            gradients.append((gradient_w, gradient_b))

            if layer_idx > 0:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    delta = (safe_delta @ weights.T) * (1.0 - np.square(activations[layer_idx]))
                delta = np.nan_to_num(
                    delta,
                    nan=0.0,
                    posinf=self.gradient_clip_value or 0.0,
                    neginf=-(self.gradient_clip_value or 0.0),
                )

        gradients.reverse()
        return gradients

    def _apply_gradients(self, gradients: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for layer_idx, (gradient_w, gradient_b) in enumerate(gradients):
            if self.gradient_clip_value is not None:
                gradient_w = np.clip(gradient_w, -self.gradient_clip_value, self.gradient_clip_value)
                gradient_b = np.clip(gradient_b, -self.gradient_clip_value, self.gradient_clip_value)
            gradient_w = np.nan_to_num(
                gradient_w,
                nan=0.0,
                posinf=self.gradient_clip_value or 0.0,
                neginf=-(self.gradient_clip_value or 0.0),
            )
            gradient_b = np.nan_to_num(
                gradient_b,
                nan=0.0,
                posinf=self.gradient_clip_value or 0.0,
                neginf=-(self.gradient_clip_value or 0.0),
            )
            self.weights_[layer_idx] -= self.learning_rate * gradient_w
            self.biases_[layer_idx] -= self.learning_rate * gradient_b
            if self.weight_clip_value is not None:
                self.weights_[layer_idx] = np.clip(
                    np.nan_to_num(
                        self.weights_[layer_idx],
                        nan=0.0,
                        posinf=self.weight_clip_value,
                        neginf=-self.weight_clip_value,
                    ),
                    -self.weight_clip_value,
                    self.weight_clip_value,
                )
                self.biases_[layer_idx] = np.clip(
                    np.nan_to_num(
                        self.biases_[layer_idx],
                        nan=0.0,
                        posinf=self.weight_clip_value,
                        neginf=-self.weight_clip_value,
                    ),
                    -self.weight_clip_value,
                    self.weight_clip_value,
                )

    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        epsilon = 1e-9
        clipped_predictions = np.clip(y_pred, epsilon, 1.0 - epsilon)
        base_loss = -np.mean(
            y_true * np.log(clipped_predictions) + (1.0 - y_true) * np.log(1.0 - clipped_predictions)
        )
        regularization = 0.0
        if self.l2_lambda > 0.0:
            regularization = sum(float(np.sum(np.square(weight))) for weight in self.weights_)
            regularization *= self.l2_lambda / (2.0 * y_true.shape[0])
        return float(base_loss + regularization)

    def _accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred >= self.threshold).astype(np.int64) == y_true.astype(np.int64)))

    @property
    def parameter_count(self) -> int:
        return int(sum(weight.size + bias.size for weight, bias in zip(self.weights_, self.biases_)))
