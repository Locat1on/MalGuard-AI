import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data import load_features
from src.eval.compare_models import compute_metrics
from src.models.train_mlp import fit_scaler_incrementally
from src.reproducibility import artifact_manifest, write_json_atomic


class MemorySafeTrainingTests(unittest.TestCase):
    def test_incremental_scaler_matches_full_fit(self) -> None:
        rng = np.random.default_rng(42)
        features = rng.normal(size=(101, 7)).astype(np.float32)
        indices = rng.permutation(len(features))[:83]
        expected = StandardScaler().fit(features[indices])
        actual = fit_scaler_incrementally(features, indices, batch_size=11)
        np.testing.assert_allclose(actual.mean_, expected.mean_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.var_, expected.var_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            actual.transform(features[indices]),
            expected.transform(features[indices]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_train_val_loader_does_not_require_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = np.arange(24, dtype=np.float32).reshape(8, 3)
            targets = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
            features.tofile(root / "X_train.dat")
            targets.tofile(root / "y_train.dat")
            with patch.object(load_features, "_NDIM", 3):
                X_train, y_train, X_val, y_val = load_features.load_train_val(
                    str(root), val_size=0.25, random_state=42
                )
            self.assertEqual(X_train.shape, (6, 3))
            self.assertEqual(X_val.shape, (2, 3))
            self.assertEqual(sorted(np.bincount(y_train).tolist()), [3, 3])
            self.assertEqual(sorted(np.bincount(y_val).tolist()), [1, 1])

    def test_metrics_use_probability_threshold(self) -> None:
        metrics = compute_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.7, 0.8, 0.2]),
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_artifact_hash_and_atomic_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.bin"
            artifact.write_bytes(b"checkpoint")
            manifest = artifact_manifest([artifact])
            self.assertEqual(manifest["model.bin"]["size_bytes"], 10)
            self.assertEqual(len(manifest["model.bin"]["sha256"]), 64)

            output = root / "manifest.json"
            write_json_atomic(output, {"ok": True})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"ok": True})
            self.assertFalse((root / "manifest.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
