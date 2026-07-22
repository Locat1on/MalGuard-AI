import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1] / "webapp" / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import history
from app.main import app
from app.predictor import (
    FeatureExtractionError,
    ModelUnavailableError,
    Predictor,
    verify_evaluated_artifacts,
)
from app.schemas import AttckTag, DetectionResult
from app.settings import (
    DEFAULT_CORS_ORIGINS,
    Settings,
    parse_cors_origins,
    parse_inference_concurrency,
)
from app.upload_limits import MAX_BATCH_REQUEST_BYTES, MAX_SINGLE_REQUEST_BYTES


def _result(filename: str = "sample.exe", verdict: str = "malicious") -> DetectionResult:
    return DetectionResult(
        filename=filename,
        verdict=verdict,
        confidence=0.9,
        family="Example" if verdict == "malicious" else None,
        familyConfidence=0.7 if verdict == "malicious" else None,
        gradcamUrl=None,
        attck=[AttckTag(tactic="Defense Evasion", technique="T1027")],
        llmReport="<script>alert(1)</script>",
        modelAgreement="disagree",
        lgbmScore=0.8,
        mlpScore=0.6,
        llmVerdict="benign",
        llmConfidence=0.6,
    )


class RuntimeSettingsTests(unittest.TestCase):
    def test_defaults_and_custom_origins_are_normalized(self) -> None:
        settings = Settings.from_environ({})
        self.assertEqual(settings.cors_origins, DEFAULT_CORS_ORIGINS)
        self.assertEqual(settings.inference_concurrency, 1)
        self.assertEqual(
            parse_cors_origins(
                "http://192.168.56.1:5173/, https://demo.example, "
                "http://192.168.56.1:5173"
            ),
            ("http://192.168.56.1:5173", "https://demo.example"),
        )

    def test_invalid_runtime_settings_fail_closed(self) -> None:
        for value in ("*", "ftp://example.test", "https://example.test/path"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_cors_origins(value)
        for value in ("0", "9", "two"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_inference_concurrency(value)

    def test_declared_oversized_uploads_are_rejected_before_route_parsing(self) -> None:
        client = TestClient(app)
        for path, limit in (
            ("/api/detect", MAX_SINGLE_REQUEST_BYTES),
            ("/api/detect/batch", MAX_BATCH_REQUEST_BYTES),
        ):
            with self.subTest(path=path):
                response = client.post(
                    path,
                    content=b"",
                    headers={"Content-Length": str(limit + 1)},
                )
                self.assertEqual(response.status_code, 413)
                self.assertIn("请求体过大", response.json()["detail"])

    def test_cors_preflight_accepts_only_configured_origin(self) -> None:
        client = TestClient(app)
        allowed = client.options(
            "/api/health",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(
            allowed.headers["access-control-allow-origin"],
            "http://127.0.0.1:5173",
        )

        rejected = client.options(
            "/api/health",
            headers={
                "Origin": "http://unconfigured.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertNotIn("access-control-allow-origin", rejected.headers)


class PredictorReliabilityTests(unittest.TestCase):
    def test_model_load_exception_is_captured(self) -> None:
        instance = Predictor.__new__(Predictor)
        instance.model_load_error = None
        with patch.object(Predictor, "_load_models", side_effect=RuntimeError("bad checkpoint")):
            self.assertFalse(instance._try_load_models())
        self.assertEqual(instance.model_load_error, "RuntimeError: bad checkpoint")

    def test_deployed_artifact_provenance_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.bin"
            artifact.write_bytes(b"evaluated checkpoint")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            manifest = root / "evaluation_manifest.json"
            manifest.write_text(
                json.dumps({"artifacts": {"model.bin": {"sha256": digest}}}),
                encoding="utf-8",
            )

            verified, warning = verify_evaluated_artifacts(
                manifest, {"model.bin": artifact}
            )
            self.assertTrue(verified)
            self.assertIsNone(warning)

            artifact.write_bytes(b"different checkpoint")
            verified, warning = verify_evaluated_artifacts(
                manifest, {"model.bin": artifact}
            )
            self.assertFalse(verified)
            self.assertIn("不一致", warning)

            verified, warning = verify_evaluated_artifacts(
                root / "missing.json", {"model.bin": artifact}
            )
            self.assertIsNone(verified)
            self.assertIn("无法核验", warning)
    def test_unavailable_model_does_not_return_stub_by_default(self) -> None:
        instance = Predictor.__new__(Predictor)
        instance.models_loaded = False
        instance.stub_enabled = False
        instance.model_load_error = "checkpoint mismatch"
        with self.assertRaisesRegex(ModelUnavailableError, "checkpoint mismatch"):
            instance.predict("sample.exe", b"MZ")

    def test_vectorized_predictor_calls_each_model_once(self) -> None:
        class FakeLightGBM:
            def __init__(self) -> None:
                self.calls = 0

            def predict(self, features):
                self.calls += 1
                self.last_shape = features.shape
                return features[:, 1]

        class IdentityScaler:
            @staticmethod
            def transform(features):
                return features

        class FakeMLP:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, tensor):
                self.calls += 1
                self.last_shape = tuple(tensor.shape)
                return tensor[:, 0]

        instance = Predictor.__new__(Predictor)
        instance.models_loaded = True
        instance.model_load_error = None
        instance.device = torch.device("cpu")
        instance.family_model_loaded = False
        instance.lgbm_model = FakeLightGBM()
        instance.scaler = IdentityScaler()
        instance.mlp_model = FakeMLP()
        instance._inference_slots = threading.BoundedSemaphore(1)

        results = instance.predict_features_ml_only(
            ["first.exe", "second.exe"],
            [
                np.array([0.9, 0.8], dtype=np.float32),
                np.array([-2.0, 0.2], dtype=np.float32),
            ],
        )
        self.assertEqual(instance.lgbm_model.calls, 1)
        self.assertEqual(instance.mlp_model.calls, 1)
        self.assertEqual(instance.lgbm_model.last_shape, (2, 2))
        self.assertEqual(instance.mlp_model.last_shape, (2, 2))
        self.assertEqual([result.verdict for result in results], ["malicious", "benign"])
        self.assertEqual(
            [result.filename for result in results],
            ["first.exe", "second.exe"],
        )

    def test_shared_models_respect_inference_concurrency_limit(self) -> None:
        class FakeLightGBM:
            @staticmethod
            def predict(features):
                return np.full(len(features), 0.9)

        class IdentityScaler:
            @staticmethod
            def transform(features):
                return features

        class SlowMLP:
            def __init__(self) -> None:
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def __call__(self, tensor):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.04)
                with self.lock:
                    self.active -= 1
                return torch.zeros((len(tensor), 1), dtype=torch.float32)

        instance = Predictor.__new__(Predictor)
        instance.models_loaded = True
        instance.model_load_error = None
        instance.device = torch.device("cpu")
        instance.family_model_loaded = False
        instance.lgbm_model = FakeLightGBM()
        instance.scaler = IdentityScaler()
        instance.mlp_model = SlowMLP()
        instance._inference_slots = threading.BoundedSemaphore(1)

        def predict(index: int) -> DetectionResult:
            return instance.predict_features_ml_only(
                [f"sample-{index}.exe"],
                [np.array([0.1, 0.2], dtype=np.float32)],
            )[0]

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(predict, range(2)))

        self.assertEqual(instance.mlp_model.max_active, 1)
        self.assertEqual(len(results), 2)

    def test_batch_payload_total_is_rejected_before_feature_extraction(self) -> None:
        client = TestClient(app)
        detect_module = sys.modules["app.routers.detect"]
        predictor = detect_module.predictor
        with (
            patch.object(predictor, "models_loaded", True),
            patch.object(detect_module, "MAX_BATCH_PAYLOAD_BYTES", 5),
            patch.object(predictor, "extract_feature_vector") as extract,
        ):
            response = client.post(
                "/api/detect/batch",
                files=[
                    ("files", ("first.exe", b"123", "application/octet-stream")),
                    ("files", ("second.exe", b"456", "application/octet-stream")),
                ],
            )

        self.assertEqual(response.status_code, 413)
        self.assertIn("批量文件总计", response.json()["detail"])
        extract.assert_not_called()

    def test_batch_route_vectorizes_valid_files_and_preserves_order(self) -> None:
        client = TestClient(app)
        predictor = sys.modules["app.routers.detect"].predictor

        def extract(content: bytes) -> np.ndarray:
            if content == b"bad":
                raise FeatureExtractionError("invalid PE")
            return np.array([len(content), 1], dtype=np.float32)

        def predict_batch(filenames, feature_rows):
            self.assertEqual(filenames, ["first.exe", "second.exe"])
            self.assertEqual(len(feature_rows), 2)
            return [
                _result("first.exe", "malicious"),
                _result("second.exe", "benign"),
            ]

        with (
            patch.object(predictor, "models_loaded", True),
            patch.object(predictor, "extract_feature_vector", side_effect=extract),
            patch.object(
                predictor,
                "predict_features_ml_only",
                side_effect=predict_batch,
            ) as batch_predict,
            patch.object(history, "record_many", return_value=[101, 102]),
        ):
            response = client.post(
                "/api/detect/batch",
                files=[
                    ("files", ("first.exe", b"good-one", "application/octet-stream")),
                    ("files", ("broken.exe", b"bad", "application/octet-stream")),
                    ("files", ("second.exe", b"good-two", "application/octet-stream")),
                ],
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(batch_predict.call_count, 1)
        body = response.json()
        self.assertEqual(body["total"], 3)
        self.assertEqual(body["failed"], 1)
        self.assertEqual(
            [item["filename"] for item in body["items"]],
            ["first.exe", "broken.exe", "second.exe"],
        )
        self.assertEqual(
            [item["historyId"] for item in body["items"]],
            [101, None, 102],
        )

    def test_non_pe_upload_is_rejected_before_model_inference(self) -> None:
        client = TestClient(app)
        predictor = sys.modules["app.routers.detect"].predictor
        with (
            patch.object(predictor, "models_loaded", True),
            patch.object(predictor.lgbm_model, "predict") as model_predict,
        ):
            response = client.post(
                "/api/detect",
                files={
                    "file": (
                        "renamed.exe",
                        b"This is plain text, not a PE file.",
                        "application/octet-stream",
                    )
                },
            )
        self.assertEqual(response.status_code, 422)
        self.assertIn("MZ", response.json()["detail"])
        model_predict.assert_not_called()

    def test_mz_header_with_out_of_bounds_pe_pointer_is_rejected(self) -> None:
        malformed = bytearray(64)
        malformed[:2] = b"MZ"
        malformed[0x3C:0x40] = (4096).to_bytes(4, "little")
        instance = Predictor.__new__(Predictor)
        with self.assertRaisesRegex(FeatureExtractionError, "超出文件边界"):
            instance.extract_feature_vector(bytes(malformed))

    def test_detect_returns_503_before_reading_when_model_unavailable(self) -> None:
        client = TestClient(app)
        predictor = sys.modules["app.routers.detect"].predictor
        with (
            patch.object(predictor, "models_loaded", False),
            patch.object(predictor, "stub_enabled", False),
            patch.object(predictor, "model_load_error", "checkpoint mismatch"),
        ):
            response = client.post(
                "/api/detect",
                files={"file": ("sample.exe", b"MZ", "application/octet-stream")},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("checkpoint mismatch", response.json()["detail"])

    def test_health_and_ready_expose_real_state(self) -> None:
        client = TestClient(app)
        health_response = client.get("/api/health")
        self.assertEqual(health_response.status_code, 200)
        self.assertRegex(health_response.headers["X-Request-ID"], r"^[0-9a-f]{32}$")
        self.assertGreaterEqual(float(health_response.headers["X-Process-Time-Ms"]), 0)
        body = health_response.json()
        self.assertIn(body["mode"], ("real", "stub", "unavailable"))
        self.assertEqual(body["ready"], body["modelsLoaded"])
        self.assertIn("modelProvenanceVerified", body)
        self.assertIn("modelProvenanceWarning", body)
        self.assertEqual(
            body["inferenceConcurrency"],
            sys.modules["app.main"].predictor.inference_concurrency,
        )

        ready_response = client.get("/api/ready")
        self.assertEqual(ready_response.status_code, 200 if body["ready"] else 503)

        stats_response = client.get("/api/history/stats")
        self.assertEqual(stats_response.status_code, 200)
        self.assertIn("modelDisagreements", stats_response.json())

    def test_metrics_provenance_endpoint(self) -> None:
        client = TestClient(app)
        metrics_module = sys.modules["app.routers.metrics"]
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "evaluation_manifest.json"
            with patch.object(metrics_module, "MANIFEST_FILE", manifest_path):
                missing = client.get("/api/metrics/provenance")
                self.assertEqual(missing.status_code, 404)

                manifest_path.write_text(
                    '{"protocol":{"test_rows":240000},"artifacts":[]}',
                    encoding="utf-8",
                )
                response = client.get("/api/metrics/provenance")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["protocol"]["test_rows"], 240000)


class HistoryReliabilityTests(unittest.TestCase):
    def test_crud_stats_wal_and_html_escaping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "history.db"
            with patch("app.history.DB_PATH", db_path):
                history.init_db()
                first_id = history.record(_result(), "single", "a" * 64)
                second_id = history.record(
                    _result("benign.exe", "benign"), "batch", "b" * 64
                )

                stats = history.stats()
                self.assertEqual(stats["total"], 2)
                self.assertEqual(stats["malicious"], 1)
                self.assertEqual(stats["benign"], 1)
                self.assertEqual(stats["single"], 1)
                self.assertEqual(stats["batch"], 1)
                self.assertEqual(stats["modelDisagreements"], 2)
                self.assertEqual(stats["llmCompared"], 2)
                self.assertEqual(stats["llmDisagreements"], 1)

                report = history.render_report_html(history.get(first_id))
                self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", report)
                self.assertNotIn("<script>alert(1)</script>", report)
                self.assertTrue(history.delete(second_id))
                self.assertEqual(history.clear(), 1)
                self.assertEqual(history.stats()["total"], 0)


if __name__ == "__main__":
    unittest.main()
