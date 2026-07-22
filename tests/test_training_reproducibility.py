import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.data import load_features
from src.eval.compare_models import compute_metrics
from src.models.family_checkpoint import unpack_family_checkpoint
from src.models.train_family import (
    IndexedFamilyDataset,
    _atomic_save_family_checkpoint,
    collapse_confusion_classes,
    make_indexed_loader,
    select_family_indices,
)
from src.models.train_lightgbm import _atomic_save_model
from src.models.train_mlp import (
    _atomic_save_pickle,
    _atomic_save_state_dict,
    fit_scaler_incrementally,
)
from src.models.training_utils import TorchStandardizer
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

    def test_torch_standardizer_matches_sklearn_transform(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(31, 9)).astype(np.float32)
        scaler = StandardScaler().fit(features[:23])
        standardizer = TorchStandardizer(scaler, torch.device("cpu"))

        actual = standardizer(torch.from_numpy(features.copy())).numpy()
        np.testing.assert_allclose(
            actual,
            scaler.transform(features),
            rtol=1e-5,
            atol=1e-5,
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

    def test_family_loader_keeps_memmap_rows_and_labels_aligned(self) -> None:
        features = np.arange(15, dtype=np.float32).reshape(5, 3)
        targets = np.array([1, 0, 1, 1, 1], dtype=np.int32)
        families = np.array(["alpha", None, None, "beta", "rare"], dtype=object)
        candidates = np.arange(5)
        indices, labels = select_family_indices(
            targets,
            families,
            candidates,
            {"alpha": 0, "beta": 1},
            other_idx=2,
        )
        np.testing.assert_array_equal(indices, [0, 3, 4])
        np.testing.assert_array_equal(labels, [0, 1, 2])

        dataset = IndexedFamilyDataset(features, indices, labels)
        loader = make_indexed_loader(
            dataset,
            batch_size=2,
            shuffle=False,
            seed=42,
            pin_memory=False,
        )
        batches = list(loader)
        actual_features = np.concatenate([batch[0].numpy() for batch in batches])
        actual_labels = np.concatenate([batch[1].numpy() for batch in batches])
        np.testing.assert_allclose(
            actual_features,
            features[indices],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(actual_labels, labels)

    def test_large_family_confusion_matrix_is_collapsed(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 1, 3, 2, 0])
        collapsed_true, collapsed_pred, display_labels = collapse_confusion_classes(
            y_true,
            y_pred,
            ["a", "b", "c", "d"],
            max_classes=2,
        )
        np.testing.assert_array_equal(collapsed_true, [0, 0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(collapsed_pred, [0, 1, 2, 1, 2, 2, 0])
        self.assertEqual(display_labels, ["a", "b", "其余已建模家族"])

    def test_metrics_use_probability_threshold(self) -> None:
        metrics = compute_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.7, 0.8, 0.2]),
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_lightgbm_atomic_save_preserves_deployed_model_on_failure(self) -> None:
        class FailingBooster:
            @staticmethod
            def save_model(path: str) -> None:
                Path(path).write_text("partial", encoding="utf-8")
                raise RuntimeError("simulated write failure")

        class EmptyBooster:
            @staticmethod
            def save_model(path: str) -> None:
                Path(path).write_bytes(b"")

        class WorkingBooster:
            @staticmethod
            def save_model(path: str) -> None:
                Path(path).write_text("complete-model", encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            deployed = Path(directory) / "lightgbm.txt"
            deployed.write_text("current-model", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "simulated write failure"):
                _atomic_save_model(FailingBooster(), deployed)
            self.assertEqual(deployed.read_text(encoding="utf-8"), "current-model")
            self.assertFalse((Path(directory) / "lightgbm.txt.tmp").exists())

            with self.assertRaisesRegex(RuntimeError, "empty file"):
                _atomic_save_model(EmptyBooster(), deployed)
            self.assertEqual(deployed.read_text(encoding="utf-8"), "current-model")
            self.assertFalse((Path(directory) / "lightgbm.txt.tmp").exists())

            _atomic_save_model(WorkingBooster(), deployed)
            self.assertEqual(deployed.read_text(encoding="utf-8"), "complete-model")
            self.assertFalse((Path(directory) / "lightgbm.txt.tmp").exists())

    def test_mlp_artifacts_are_staged_before_publish(self) -> None:
        model = torch.nn.Linear(3, 1)
        scaler = StandardScaler().fit(
            np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deployed_model = root / "mlp.pt"
            deployed_scaler = root / "scaler.pkl"
            staged_model = root / ".mlp.training.pt"
            staged_scaler = root / ".scaler.training.pkl"
            deployed_model.write_bytes(b"current-model")
            deployed_scaler.write_bytes(b"current-scaler")

            _atomic_save_state_dict(model, staged_model)
            _atomic_save_pickle(scaler, staged_scaler)
            self.assertEqual(deployed_model.read_bytes(), b"current-model")
            self.assertEqual(deployed_scaler.read_bytes(), b"current-scaler")
            self.assertEqual(
                set(torch.load(staged_model, weights_only=True)),
                set(model.state_dict()),
            )
            with staged_scaler.open("rb") as file:
                staged_scaler_value = pickle.load(file)
            np.testing.assert_allclose(staged_scaler_value.mean_, scaler.mean_)

    def test_family_checkpoint_stages_embedded_labels_before_publish(self) -> None:
        labels = ["Example", "其他"]
        model = torch.nn.Linear(3, len(labels))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deployed = root / "family_mlp.pt"
            staged = root / ".family_mlp.training.pt"
            deployed.write_bytes(b"current-deployed-model")

            _atomic_save_family_checkpoint(model, labels, staged)
            self.assertEqual(deployed.read_bytes(), b"current-deployed-model")

            checkpoint = torch.load(staged, map_location="cpu", weights_only=True)
            state_dict, embedded_labels = unpack_family_checkpoint(
                checkpoint, root / "missing-labels.json"
            )
            self.assertEqual(embedded_labels, labels)
            self.assertEqual(set(state_dict), set(model.state_dict()))

            staged.replace(deployed)
            published = torch.load(deployed, map_location="cpu", weights_only=True)
            _, published_labels = unpack_family_checkpoint(
                published, root / "missing-labels.json"
            )
            self.assertEqual(published_labels, labels)

            legacy_labels = root / "family_labels.json"
            legacy_labels.write_text(
                json.dumps(labels, ensure_ascii=False),
                encoding="utf-8",
            )
            legacy_state, legacy_vocab = unpack_family_checkpoint(
                model.state_dict(), legacy_labels
            )
            self.assertEqual(set(legacy_state), set(model.state_dict()))
            self.assertEqual(legacy_vocab, labels)

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
