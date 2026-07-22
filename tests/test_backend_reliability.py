import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1] / "webapp" / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import history
from app.auth import ApiKeyAuthMiddleware, document_api_key
from app.main import app
from app.predictor import (
    FeatureExtractionError,
    ModelUnavailableError,
    Predictor,
    verify_evaluated_artifacts,
)
from app.schemas import AttckTag, DetectionResult
from src.models.family_checkpoint import build_family_checkpoint
from app.settings import (
    DEFAULT_CORS_ORIGINS,
    DEFAULT_HISTORY_DB,
    PROJECT_ROOT,
    Settings,
    parse_api_key,
    parse_cors_origins,
    parse_detection_concurrency,
    parse_history_db_path,
    parse_inference_concurrency,
)
from app.upload_limits import (
    MAX_BATCH_REQUEST_BYTES,
    MAX_SINGLE_REQUEST_BYTES,
    DetectionConcurrencyLimitMiddleware,
)


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
        self.assertEqual(settings.detection_concurrency, 2)
        self.assertIsNone(settings.api_key)
        self.assertEqual(settings.history_db_path, DEFAULT_HISTORY_DB)
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
        for value in ("0", "33", "two"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_detection_concurrency(value)
        self.assertEqual(parse_detection_concurrency("3"), 3)
        for value in ("short", "密钥" * 8):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_api_key(value)
        self.assertEqual(parse_api_key("a" * 16), "a" * 16)
        secure_settings = Settings.from_environ({"MALGUARD_API_KEY": "a" * 16})
        self.assertEqual(secure_settings.api_key, "a" * 16)
        self.assertNotIn("a" * 16, repr(secure_settings))
        self.assertEqual(
            parse_history_db_path("data/custom-history.db"),
            (PROJECT_ROOT / "data" / "custom-history.db").resolve(),
        )

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

    def test_api_key_protects_routes_and_keeps_preflight_public(self) -> None:
        protected_app = FastAPI()
        protected_app.add_middleware(
            ApiKeyAuthMiddleware,
            api_key="a" * 16,
            protected_prefixes=("/api/detect", "/api/history"),
        )
        protected_app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://frontend.example"],
            allow_methods=["GET", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )

        @protected_app.get("/api/health")
        async def public_health() -> dict:
            return {"ok": True}

        @protected_app.get("/api/history")
        async def protected_history() -> list:
            return []

        @protected_app.post("/api/detect")
        async def protected_detect(file: UploadFile) -> dict:
            return {"filename": file.filename}

        document_api_key(
            protected_app,
            ("/api/detect", "/api/history"),
        )
        client = TestClient(protected_app)
        openapi = client.get("/openapi.json").json()
        self.assertEqual(
            openapi["components"]["securitySchemes"]["ApiKeyAuth"]["name"],
            "X-API-Key",
        )
        self.assertEqual(
            openapi["paths"]["/api/history"]["get"]["security"],
            [{"ApiKeyAuth": []}],
        )
        self.assertNotIn("security", openapi["paths"]["/api/health"]["get"])
        self.assertEqual(client.get("/api/health").status_code, 200)
        rejected_before_parse = client.post(
            "/api/detect",
            content=b"not-a-valid-multipart-body",
            headers={"Content-Type": "multipart/form-data; boundary=broken"},
        )
        self.assertEqual(rejected_before_parse.status_code, 401)

        missing = client.get(
            "/api/history",
            headers={"Origin": "http://frontend.example"},
        )
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.headers["www-authenticate"], "ApiKey")
        self.assertEqual(
            missing.headers["access-control-allow-origin"],
            "http://frontend.example",
        )
        self.assertIn("X-Request-ID", missing.headers["access-control-expose-headers"])
        self.assertEqual(
            client.get("/api/history", headers={"X-API-Key": "wrong"}).status_code,
            401,
        )
        self.assertEqual(
            client.get(
                "/api/history",
                headers={"X-API-Key": "a" * 16},
            ).status_code,
            200,
        )

        preflight = client.options(
            "/api/history",
            headers={
                "Origin": "http://frontend.example",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-API-Key",
            },
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertIn("X-API-Key", preflight.headers["access-control-allow-headers"])

    def test_detection_concurrency_rejects_before_body_parsing(self) -> None:
        limited_app = FastAPI()
        limited_app.add_middleware(
            DetectionConcurrencyLimitMiddleware,
            max_active=1,
        )
        entered = threading.Event()
        release = threading.Event()

        @limited_app.post("/api/detect")
        def slow_detect(file: UploadFile) -> dict:
            entered.set()
            if not release.wait(timeout=2):
                raise RuntimeError("test request was not released")
            return {"filename": file.filename}

        @limited_app.get("/api/health")
        def unaffected_health() -> dict:
            return {"ok": True}

        client = TestClient(limited_app)
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(
                client.post,
                "/api/detect",
                files={"file": ("first.exe", b"data", "application/octet-stream")},
            )
            self.assertTrue(entered.wait(timeout=1))
            busy = client.post(
                "/api/detect",
                content=b"not-a-valid-multipart-body",
                headers={"Content-Type": "multipart/form-data; boundary=broken"},
            )
            health = client.get("/api/health")
            release.set()
            first_response = first.result(timeout=2)

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(busy.status_code, 429)
        self.assertEqual(busy.headers["retry-after"], "1")
        self.assertIn("服务繁忙", busy.json()["detail"])
        self.assertEqual(health.status_code, 200)
        self.assertEqual(
            client.post(
                "/api/detect",
                files={"file": ("third.exe", b"data", "application/octet-stream")},
            ).status_code,
            200,
        )
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
        exposed = client.get(
            "/api/health",
            headers={"Origin": "http://127.0.0.1:5173"},
        ).headers["access-control-expose-headers"]
        self.assertIn("Content-Disposition", exposed)
        self.assertIn("Retry-After", exposed)
        self.assertIn("WWW-Authenticate", exposed)
        self.assertIn("X-Request-ID", exposed)

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

    def test_invalid_family_vocabulary_disables_family_model(self) -> None:
        invalid_vocabularies = (
            [],
            ["其他"],
            ["Example", "", "其他"],
            ["Example", "Example", "其他"],
            ["其他", "Example"],
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            (checkpoint_dir / "family_mlp.pt").write_bytes(b"not-loaded")
            labels_path = checkpoint_dir / "family_labels.json"

            for labels in invalid_vocabularies:
                with self.subTest(labels=labels):
                    labels_path.write_text(
                        json.dumps(labels, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    instance = Predictor.__new__(Predictor)
                    instance.family_model_load_error = None
                    instance.device = torch.device("cpu")
                    with (
                        patch("app.predictor.CHECKPOINTS_DIR", checkpoint_dir),
                        patch("app.predictor.torch.load", return_value={}),
                    ):
                        self.assertFalse(instance._try_load_family_model())
                    self.assertIn(
                        "family_labels.json",
                        instance.family_model_load_error,
                    )

    def test_bundled_family_checkpoint_loads_without_label_sidecar(self) -> None:
        predictor_module = sys.modules["app.predictor"]
        config = {
            "hidden_dims": [8, 4],
            "dropout": 0.0,
            "embed_dim": 4,
            "family_confidence_floor": 0.3,
        }
        labels = ["Example", "其他"]
        model = predictor_module.MalwareMLP(
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
            embed_dim=config["embed_dim"],
            num_classes=len(labels),
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            torch.save(
                build_family_checkpoint(model.state_dict(), labels),
                checkpoint_dir / "family_mlp.pt",
            )
            instance = Predictor.__new__(Predictor)
            instance.family_model_load_error = None
            instance.device = torch.device("cpu")
            with (
                patch("app.predictor.CHECKPOINTS_DIR", checkpoint_dir),
                patch("app.predictor.load_config", return_value=config),
            ):
                self.assertTrue(instance._load_family_model())

        self.assertEqual(instance.family_labels, labels)
        self.assertEqual(instance.family_confidence_floor, 0.3)

    def test_family_runtime_failure_degrades_without_blocking_detection(self) -> None:
        class FakeLightGBM:
            @staticmethod
            def predict(features):
                return np.full(len(features), 0.9)

        class IdentityScaler:
            @staticmethod
            def transform(features):
                return features

        class FakeMLP:
            @staticmethod
            def __call__(tensor):
                return torch.full((len(tensor), 1), 2.0)

        class BrokenFamily:
            @staticmethod
            def __call__(tensor):
                raise RuntimeError("broken family head")

        class InvalidFamily:
            @staticmethod
            def __call__(tensor):
                return torch.tensor([[float("nan"), 0.0]]).repeat(
                    len(tensor), 1
                )

        class PartiallyInvalidFamily:
            @staticmethod
            def __call__(tensor):
                return torch.tensor(
                    [[2.0, 0.0], [float("nan"), 0.0]]
                )

        def build_instance(family_model) -> Predictor:
            instance = Predictor.__new__(Predictor)
            instance.models_loaded = True
            instance.model_load_error = None
            instance.device = torch.device("cpu")
            instance.lgbm_model = FakeLightGBM()
            instance.scaler = IdentityScaler()
            instance.mlp_model = FakeMLP()
            instance.family_model = family_model
            instance.family_model_loaded = True
            instance.family_model_load_error = None
            instance.family_labels = ["Example", "其他"]
            instance.family_confidence_floor = 0.3
            instance._inference_slots = threading.BoundedSemaphore(1)
            return instance

        features = [np.array([0.1, 0.2], dtype=np.float32)]
        for family_model in (BrokenFamily(), InvalidFamily()):
            with self.subTest(family_model=type(family_model).__name__):
                instance = build_instance(family_model)
                result = instance.predict_features_ml_only(
                    ["sample.exe"], features
                )[0]
                self.assertEqual(result.verdict, "malicious")
                self.assertIsNone(result.family)
                self.assertIsNone(result.familyConfidence)
                self.assertFalse(instance.family_model_loaded)
                self.assertIn("运行时推理失败", instance.family_model_load_error)

        partial_instance = build_instance(PartiallyInvalidFamily())
        partial_results = partial_instance.predict_features_ml_only(
            ["first.exe", "second.exe"], features * 2
        )
        self.assertTrue(all(result.family is None for result in partial_results))
        self.assertFalse(partial_instance.family_model_loaded)

        single_instance = build_instance(BrokenFamily())
        self.assertEqual(
            single_instance._predict_family(np.array(features)),
            (None, None),
        )
        self.assertFalse(single_instance.family_model_loaded)

        decoder = build_instance(BrokenFamily())
        with self.assertRaisesRegex(ModelUnavailableError, "概率总和"):
            decoder._decode_family_probability(np.array([0.4, 0.4]))

        runtime_predictor = sys.modules["app.main"].predictor
        with (
            patch.object(runtime_predictor, "family_model_loaded", False),
            patch.object(
                runtime_predictor,
                "family_model_load_error",
                "运行时推理失败（RuntimeError）。",
            ),
        ):
            health = TestClient(app).get("/api/health").json()
        self.assertFalse(health["familyModelLoaded"])
        self.assertIn("运行时推理失败", health["familyModelLoadError"])

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

    def test_invalid_model_probabilities_fail_closed(self) -> None:
        class FakeLightGBM:
            def __init__(self, output) -> None:
                self.output = output

            def predict(self, features):
                return self.output

        class IdentityScaler:
            @staticmethod
            def transform(features):
                return features

        class FakeMLP:
            def __init__(self, logit: float = 0.0) -> None:
                self.logit = logit

            def __call__(self, tensor):
                return torch.full(
                    (len(tensor), 1), self.logit, dtype=torch.float32
                )

        instance = Predictor.__new__(Predictor)
        instance.models_loaded = True
        instance.model_load_error = None
        instance.device = torch.device("cpu")
        instance.family_model_loaded = False
        instance.scaler = IdentityScaler()
        instance.mlp_model = FakeMLP()
        instance._inference_slots = threading.BoundedSemaphore(1)
        features = [np.array([0.1, 0.2], dtype=np.float32)]

        for invalid_output in (
            np.array([np.nan]),
            np.array([1.1]),
            np.array([[0.9]]),
        ):
            with self.subTest(lightgbm_output=invalid_output):
                instance.lgbm_model = FakeLightGBM(invalid_output)
                with self.assertRaisesRegex(ModelUnavailableError, "LightGBM"):
                    instance.predict_features_ml_only(["sample.exe"], features)

        instance.extract_feature_vector = lambda content: features[0]
        instance.lgbm_model = FakeLightGBM(np.array([np.nan]))
        with self.assertRaisesRegex(ModelUnavailableError, "LightGBM"):
            instance._score(b"MZ")

        instance.lgbm_model = FakeLightGBM(np.array([0.9]))
        instance.mlp_model = FakeMLP(float("nan"))
        with self.assertRaisesRegex(ModelUnavailableError, "MLP"):
            instance.predict_features_ml_only(["sample.exe"], features)

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

    def test_runtime_model_failure_returns_503(self) -> None:
        client = TestClient(app)
        predictor = sys.modules["app.routers.detect"].predictor
        with (
            patch.object(predictor, "models_loaded", True),
            patch.object(
                predictor,
                "predict",
                side_effect=ModelUnavailableError("LightGBM 推理输出无效。"),
            ),
        ):
            response = client.post(
                "/api/detect",
                files={"file": ("sample.exe", b"MZ", "application/octet-stream")},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("LightGBM", response.json()["detail"])

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
        self.assertIn("apiKeyRequired", body)
        self.assertEqual(
            body["detectionConcurrency"],
            sys.modules["app.main"].settings.detection_concurrency,
        )
        self.assertEqual(
            body["inferenceConcurrency"],
            sys.modules["app.main"].predictor.inference_concurrency,
        )

        ready_response = client.get("/api/ready")
        self.assertEqual(ready_response.status_code, 200 if body["ready"] else 503)

        stats_response = client.get("/api/history/stats")
        self.assertEqual(stats_response.status_code, 200)
        self.assertIn("modelDisagreements", stats_response.json())

    def test_metrics_endpoint_never_returns_placeholder_scores(self) -> None:
        client = TestClient(app)
        metrics_module = sys.modules["app.routers.metrics"]
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "evaluation_manifest.json"
            with patch.object(metrics_module, "MANIFEST_FILE", manifest_path):
                missing = client.get("/api/metrics")
                self.assertEqual(missing.status_code, 404)
                self.assertIn("尚未生成", missing.json()["detail"])

                manifest_path.write_text("{broken-json", encoding="utf-8")
                invalid = client.get("/api/metrics")
                self.assertEqual(invalid.status_code, 503)
                self.assertIn("来源清单不可用", invalid.json()["detail"])

                manifest_path.write_text(
                    json.dumps(
                        {
                            "results": [
                                {
                                    "model": "invalid",
                                    "accuracy": 1.01,
                                    "precision": 0.8,
                                    "recall": 0.7,
                                    "f1": 0.75,
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_of_range = client.get("/api/metrics")
                self.assertEqual(out_of_range.status_code, 503)

                manifest_path.write_text(
                    json.dumps(
                        {
                            "results": [
                                {
                                    "model": "verified",
                                    "accuracy": 0.9,
                                    "precision": 0.8,
                                    "recall": 0.7,
                                    "f1": 0.75,
                                    "confusion_matrix": [[1, 0], [0, 1]],
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                valid = client.get("/api/metrics")
                self.assertEqual(valid.status_code, 200)
                self.assertEqual(valid.json()[0]["model"], "verified")

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
    def test_report_distinguishes_batch_from_unavailable_single_llm(self) -> None:
        unavailable = _result().model_copy(
            update={
                "llmReport": "[LLM 分析不可用] 未配置。",
                "llmVerdict": None,
                "llmConfidence": None,
            }
        )
        batch = unavailable.model_copy(update={"filename": "batch.exe"})

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "history.db"
            with patch("app.history.DB_PATH", db_path):
                history.init_db()
                single_id = history.record(unavailable, "single", "c" * 64)
                batch_id = history.record(batch, "batch", "d" * 64)

                single_report = history.render_report_html(history.get(single_id))
                batch_report = history.render_report_html(history.get(batch_id))

        self.assertIn("LLM 分析不可用或未形成有效结论", single_report)
        self.assertNotIn("本次为批量检测", single_report)
        self.assertIn("本次为批量检测，未运行 LLM 分析。", batch_report)

    def test_crud_stats_wal_and_html_escaping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "history.db"
            with patch("app.history.DB_PATH", db_path):
                history.init_db()
                first_id = history.record(_result(), "single", "a" * 64)
                second_id = history.record(
                    _result("benign.exe", "benign"), "batch", "b" * 64
                )

                client = TestClient(app)
                first_page = client.get(
                    "/api/history?limit=1&offset=0",
                    headers={"Origin": DEFAULT_CORS_ORIGINS[0]},
                )
                self.assertEqual(first_page.status_code, 200)
                self.assertEqual(first_page.headers["x-total-count"], "2")
                self.assertEqual(len(first_page.json()), 1)
                self.assertEqual(first_page.json()[0]["id"], second_id)
                self.assertIn(
                    "X-Total-Count",
                    first_page.headers["access-control-expose-headers"],
                )
                empty_page = client.get("/api/history?limit=1&offset=2")
                self.assertEqual(empty_page.json(), [])
                self.assertEqual(empty_page.headers["x-total-count"], "2")

                stats = history.stats()
                self.assertEqual(stats["total"], 2)
                self.assertEqual(stats["malicious"], 1)
                self.assertEqual(stats["benign"], 1)
                self.assertEqual(stats["single"], 1)
                self.assertEqual(stats["batch"], 1)
                self.assertEqual(stats["modelDisagreements"], 2)
                self.assertEqual(stats["llmCompared"], 2)
                self.assertEqual(stats["llmDisagreements"], 1)

                backup_response = client.get("/api/history/backup")
                self.assertEqual(backup_response.status_code, 200)
                self.assertIn(
                    "attachment;",
                    backup_response.headers["content-disposition"],
                )
                self.assertTrue(backup_response.content.startswith(b"SQLite format 3\x00"))
                downloaded = Path(directory) / "downloaded.db"
                downloaded.write_bytes(backup_response.content)
                with closing(sqlite3.connect(downloaded)) as backup_conn:
                    self.assertEqual(
                        backup_conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0],
                        2,
                    )
                    self.assertEqual(
                        backup_conn.execute("PRAGMA quick_check").fetchone()[0],
                        "ok",
                    )
                self.assertEqual(
                    list(Path(directory).glob(".history-backup-*.db")),
                    [],
                )
                with self.assertRaises(ValueError):
                    history.backup_to(db_path)
                with self.assertRaises(FileExistsError):
                    history.backup_to(downloaded)

                report = history.render_report_html(history.get(first_id))
                self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", report)
                self.assertNotIn("<script>alert(1)</script>", report)
                self.assertTrue(history.delete(second_id))
                self.assertEqual(history.clear(), 1)
                self.assertEqual(history.stats()["total"], 0)


if __name__ == "__main__":
    unittest.main()
