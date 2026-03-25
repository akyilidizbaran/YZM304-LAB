import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from banknote_mlp.data import SplitIndices, build_train_fraction_indices, save_split_manifest


class DataArtifactTests(unittest.TestCase):
    def test_build_train_fraction_indices_is_nested(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        train_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        fractions = build_train_fraction_indices(y, train_indices, fractions=(0.5, 0.75, 1.0), random_state=42)

        self.assertLess(len(fractions["0.50"]), len(fractions["0.75"]))
        self.assertLess(len(fractions["0.75"]), len(fractions["1.00"]) + 1)
        self.assertTrue(set(fractions["0.50"]).issubset(set(fractions["0.75"])))
        self.assertTrue(set(fractions["0.50"]).issubset(set(fractions["1.00"])))
        self.assertTrue(set(fractions["0.75"]).issubset(set(fractions["1.00"])))

    def test_save_split_manifest_writes_fraction_metadata(self):
        split_indices = SplitIndices(
            train=np.array([0, 1, 2], dtype=np.int64),
            validation=np.array([3], dtype=np.int64),
            test=np.array([4, 5], dtype=np.int64),
        )
        fractions = {"0.50": np.array([0, 2], dtype=np.int64), "1.00": np.array([0, 1, 2], dtype=np.int64)}
        y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "split_manifest.json"
            save_split_manifest(
                path,
                feature_names=("a", "b"),
                y=y,
                split_indices=split_indices,
                train_fraction_indices=fractions,
            )
            payload = json.loads(path.read_text())

        self.assertEqual(payload["split_sizes"]["train"], 3)
        self.assertEqual(payload["train_fraction_sizes"]["0.50"], 2)
        self.assertIn("0.50", payload["train_fraction_indices"])
