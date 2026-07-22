import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1] / "webapp" / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import history
from app.main import app
from app.predictor import ModelUnavailableError, Predictor
from app.schemas import AttckTag, DetectionResult


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


class PredictorReliabilityTests(unittest.TestCase):
    def test_model_load_exception_is_captured(self) -> None:
        instance = Predictor.__new__(Predictor)
        instance.model_load_error = None
        with patch.object(Predictor, "_load_models", side_effect=RuntimeError("bad checkpoint")):
            self.assertFalse(instance._try_load_models())
        self.assertEqual(instance.model_load_error, "RuntimeError: bad checkpoint")

    def test_unavailable_model_does_not_return_stub_by_default(self) -> None:
        instance = Predictor.__new__(Predictor)
        instance.models_loaded = False
        instance.stub_enabled = False
        instance.model_load_error = "checkpoint mismatch"
        with self.assertRaisesRegex(ModelUnavailableError, "checkpoint mismatch"):
            instance.predict("sample.exe", b"MZ")

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
        body = health_response.json()
        self.assertIn(body["mode"], ("real", "stub", "unavailable"))
        self.assertEqual(body["ready"], body["modelsLoaded"])

        ready_response = client.get("/api/ready")
        self.assertEqual(ready_response.status_code, 200 if body["ready"] else 503)

        stats_response = client.get("/api/history/stats")
        self.assertEqual(stats_response.status_code, 200)
        self.assertIn("modelDisagreements", stats_response.json())


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
