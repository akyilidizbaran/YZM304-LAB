from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state


def _binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = 1e-9
    clipped_predictions = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(y_true * np.log(clipped_predictions) + (1.0 - y_true) * np.log(1.0 - clipped_predictions))
    )


def initialize_sklearn_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    l2_lambda: float,
    random_state: int,
    initial_weights: list[np.ndarray],
    initial_biases: list[np.ndarray],
) -> tuple[MLPClassifier, np.ndarray]:
    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="tanh",
        solver="sgd",
        alpha=l2_lambda,
        batch_size=len(X_train),
        learning_rate="constant",
        learning_rate_init=learning_rate,
        max_iter=1,
        warm_start=True,
        shuffle=False,
        random_state=random_state,
        momentum=0.0,
        nesterovs_momentum=False,
        tol=0.0,
        n_iter_no_change=10_000,
    )

    classes = np.unique(y_train)
    classifier.classes_ = classes
    classifier._label_binarizer = LabelBinarizer().fit(classes)
    validated_X, validated_y = classifier._validate_input(X_train, y_train, incremental=True, reset=True)
    if validated_y.ndim == 1:
        validated_y = validated_y.reshape((-1, 1))

    layer_units = [validated_X.shape[1], *hidden_layers, validated_y.shape[1]]
    classifier._random_state = check_random_state(random_state)
    classifier._initialize(validated_y, layer_units, validated_X.dtype)
    classifier.coefs_ = [np.array(weight, dtype=validated_X.dtype, copy=True) for weight in initial_weights]
    classifier.intercepts_ = [np.array(bias, dtype=validated_X.dtype, copy=True).reshape(-1) for bias in initial_biases]
    classifier._best_coefs = [coef.copy() for coef in classifier.coefs_]
    classifier._best_intercepts = [bias.copy() for bias in classifier.intercepts_]
    return classifier, classes


def train_sklearn_mlp(
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
) -> tuple[MLPClassifier, dict[str, list[float | int]]]:
    classifier, classes = initialize_sklearn_mlp(
        X_train,
        y_train,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        l2_lambda=l2_lambda,
        random_state=random_state,
        initial_weights=initial_weights,
        initial_biases=initial_biases,
    )

    history = {
        "step": [],
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }

    old_settings = np.seterr(over="ignore", invalid="ignore", divide="ignore")
    try:
        with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for step in range(1, max_iter + 1):
                classifier.partial_fit(X_train, y_train, classes=classes if step == 1 else None)
                train_probabilities = classifier.predict_proba(X_train)[:, 1]
                validation_probabilities = classifier.predict_proba(X_validation)[:, 1]
                train_predictions = (train_probabilities >= threshold).astype(np.int64)
                validation_predictions = (validation_probabilities >= threshold).astype(np.int64)

                history["step"].append(step)
                history["train_loss"].append(float(classifier.loss_))
                history["validation_loss"].append(_binary_cross_entropy(y_validation, validation_probabilities))
                history["train_accuracy"].append(float(np.mean(train_predictions == y_train)))
                history["validation_accuracy"].append(float(np.mean(validation_predictions == y_validation)))
    finally:
        np.seterr(**old_settings)

    return classifier, history


def predict_sklearn_mlp(classifier: MLPClassifier, X: np.ndarray) -> np.ndarray:
    old_settings = np.seterr(over="ignore", invalid="ignore", divide="ignore")
    try:
        with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return classifier.predict(X)
    finally:
        np.seterr(**old_settings)
