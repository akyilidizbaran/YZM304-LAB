from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


def train_sklearn_mlp(
    X_train,
    y_train,
    *,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    max_iter: int,
    l2_lambda: float,
    random_state: int,
):
    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="tanh",
        solver="sgd",
        alpha=l2_lambda,
        batch_size=len(X_train),
        learning_rate="constant",
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        shuffle=False,
        random_state=random_state,
        momentum=0.0,
        nesterovs_momentum=False,
        tol=0.0,
        n_iter_no_change=max_iter + 1,
    )

    old_settings = np.seterr(over="ignore", invalid="ignore", divide="ignore")
    try:
        with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            classifier.fit(X_train, y_train)
    finally:
        np.seterr(**old_settings)

    return classifier
