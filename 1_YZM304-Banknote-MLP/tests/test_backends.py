import unittest

import numpy as np

from banknote_mlp.pytorch_backend import predict_torch_mlp, train_torch_mlp
from banknote_mlp.shared_artifacts import generate_initial_parameters
from banknote_mlp.sklearn_models import train_sklearn_mlp


class BackendTrainingTests(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array(
            [
                [-2.0, -1.0],
                [-1.0, -2.0],
                [1.0, 1.0],
                [2.0, 1.5],
            ],
            dtype=np.float64,
        )
        self.y_train = np.array([0, 0, 1, 1], dtype=np.int64)
        self.X_validation = self.X_train.copy()
        self.y_validation = self.y_train.copy()
        self.initial_weights, self.initial_biases = generate_initial_parameters(layer_sizes=(2, 3, 1), random_state=7)

    def test_sklearn_backend_runs_with_shared_initial_weights(self):
        classifier, history = train_sklearn_mlp(
            self.X_train,
            self.y_train,
            X_validation=self.X_validation,
            y_validation=self.y_validation,
            hidden_layers=(3,),
            learning_rate=0.1,
            max_iter=10,
            l2_lambda=0.0,
            random_state=7,
            threshold=0.5,
            initial_weights=self.initial_weights,
            initial_biases=self.initial_biases,
        )
        predictions = classifier.predict(self.X_validation)
        self.assertEqual(predictions.shape[0], self.y_validation.shape[0])
        self.assertEqual(len(history["step"]), 10)

    def test_torch_backend_runs_with_shared_initial_weights(self):
        try:
            model, history = train_torch_mlp(
                self.X_train,
                self.y_train,
                X_validation=self.X_validation,
                y_validation=self.y_validation,
                hidden_layers=(3,),
                learning_rate=0.1,
                max_iter=10,
                l2_lambda=0.0,
                random_state=7,
                threshold=0.5,
                initial_weights=self.initial_weights,
                initial_biases=self.initial_biases,
            )
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in the current environment")
        predictions = predict_torch_mlp(model, self.X_validation, threshold=0.5)
        self.assertEqual(predictions.shape[0], self.y_validation.shape[0])
        self.assertEqual(len(history["step"]), 10)
