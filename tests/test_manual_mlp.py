import unittest

import numpy as np

from banknote_mlp.data import apply_split, build_split_indices
from banknote_mlp.manual_mlp import ManualMLPClassifier


class ManualMLPTests(unittest.TestCase):
    def test_manual_mlp_learns_simple_problem(self):
        X = np.array(
            [
                [-2.0, -1.0],
                [-1.0, -2.0],
                [1.0, 1.0],
                [2.0, 1.5],
            ],
            dtype=np.float64,
        )
        y = np.array([0, 0, 1, 1], dtype=np.int64)

        model = ManualMLPClassifier(hidden_layers=(3,), learning_rate=0.1, n_steps=1500, random_state=7)
        model.fit(X, y)

        predictions = model.predict(X)
        self.assertTrue(np.array_equal(predictions, y))

    def test_build_split_indices_covers_all_rows(self):
        y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0], dtype=np.int64)
        indices = build_split_indices(y, test_size=0.2, validation_size=0.25, random_state=42)
        combined = np.concatenate([indices.train, indices.validation, indices.test])

        self.assertEqual(sorted(combined.tolist()), list(range(len(y))))
        self.assertEqual(len(set(combined.tolist())), len(y))

    def test_apply_split_standardizes_training_data(self):
        X = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
            ],
            dtype=np.float64,
        )
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        feature_names = ("f1", "f2")
        indices = build_split_indices(y, test_size=0.17, validation_size=0.2, random_state=42)

        split = apply_split(X, y, feature_names, indices, standardize=True)

        self.assertTrue(np.allclose(split.X_train.mean(axis=0), np.zeros(2), atol=1e-9))


if __name__ == "__main__":
    unittest.main()
