import json
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from src.llm.feature_summary import (
    MAX_INDICATORS_PER_KIND,
    _compile_time,
    _extract_string_indicators,
    summarize,
)
from src.llm.report import (
    ANALYSIS_VERSION,
    LLMAnalysis,
    _analysis_identity,
    _load_cached,
    _save_cache,
    generate_report,
)
from tests.pe_fixture import build_minimal_suspicious_pe


class FeatureSummaryTests(unittest.TestCase):
    def test_extracts_bounded_ascii_and_utf16_indicators(self) -> None:
        ascii_data = (
            b"http://example.test/payload\x00"
            b"connect server 203.0.113.9\x00"
            b"FileVersion 5.1.0.0\x00"
            b"HKLM\\Software\\Example\\Run\x00"
            b"powershell -enc AAAA\x00"
        )
        wide_data = "https://wide.example/path".encode("utf-16le")
        urls, ips, registry_paths, commands = _extract_string_indicators(
            ascii_data + wide_data
        )

        self.assertIn("http://example.test/payload", urls)
        self.assertIn("https://wide.example/path", urls)
        self.assertEqual(ips, ["203.0.113.9"])
        self.assertNotIn("5.1.0.0", ips)
        self.assertIn(r"HKLM\Software\Example\Run", registry_paths)
        self.assertTrue(any("powershell" in value for value in commands))
        self.assertTrue(all(len(values) <= MAX_INDICATORS_PER_KIND for values in (urls, ips, registry_paths, commands)))

    def test_compile_time_flags_future_and_invalid_values(self) -> None:
        _, future_anomaly = _compile_time(4_102_444_800)
        invalid_time, invalid_anomaly = _compile_time(0)
        self.assertIn("未来", future_anomaly)
        self.assertIsNone(invalid_time)
        self.assertIn("无效", invalid_anomaly)

    def test_minimal_pe_summary_contains_structural_anomalies(self) -> None:
        summary = summarize(build_minimal_suspicious_pe())
        self.assertIn(".ex0", summary.high_entropy_sections)
        self.assertIn(".ex0", summary.nonstandard_sections)
        self.assertIsNotNone(summary.compile_time_anomaly)
        self.assertNotIn("5.1.0.0", summary.ip_addresses)
        self.assertIn("编译时间异常", summary.to_prompt_text())


class ReportCacheTests(unittest.TestCase):
    def test_analysis_identity_tracks_model_config_and_prompt(self) -> None:
        config = {
            "provider_base_url": "https://provider.example/v1/chat/completions",
            "model": "model-a",
            "temperature": 0.0,
            "max_tokens": 100,
        }
        identity = _analysis_identity(config)
        self.assertEqual(identity, _analysis_identity(dict(reversed(config.items()))))
        self.assertNotEqual(identity, _analysis_identity({**config, "model": "model-b"}))
        with patch("src.llm.report.SYSTEM_PROMPT", "changed prompt"):
            self.assertNotEqual(identity, _analysis_identity(config))

    def test_cache_requires_current_analysis_identity_and_version(self) -> None:
        analysis = LLMAnalysis("malicious", 0.8, "test")
        identity = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            with patch("src.llm.report.CACHE_DIR", cache_dir):
                _save_cache("new", analysis, identity)
                self.assertEqual(_load_cached("new", identity), analysis)
                self.assertIsNone(_load_cached("new", "b" * 64))

                (cache_dir / "old.json").write_text(
                    json.dumps(
                        {
                            "verdict": "malicious",
                            "confidence": 0.8,
                            "narrative": "old",
                            "analysis_version": ANALYSIS_VERSION - 1,
                            "analysis_identity": identity,
                        }
                    ),
                    encoding="utf-8",
                )
                self.assertIsNone(_load_cached("old", identity))

    def test_invalid_llm_config_does_not_break_detection(self) -> None:
        summary = Mock()
        with patch("src.llm.report.load_config", side_effect=KeyError("model")):
            result = generate_report(b"sample", summary)
        self.assertIsNone(result.verdict)
        self.assertIsNone(result.confidence)
        self.assertIn("LLM 配置不可用", result.narrative)

    def test_concurrent_same_file_uses_one_provider_request(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"verdict":"malicious","confidence":80,'
                            '"narrative":"发现受限静态风险线索。"}'
                        )
                    }
                }
            ]
        }
        post = Mock()

        def delayed_post(*args, **kwargs):
            time.sleep(0.05)
            return response

        post.side_effect = delayed_post
        summary = Mock()
        summary.to_prompt_text.return_value = "bounded facts"
        config = {
            "provider_base_url": "https://provider.example/v1/chat/completions",
            "model": "test-model",
            "temperature": 0.0,
            "max_tokens": 100,
            "timeout_seconds": 1,
        }

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("src.llm.report.CACHE_DIR", Path(directory)),
                patch("src.llm.report.load_config", return_value=config),
                patch("src.llm.report.requests.post", post),
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            ):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(
                        pool.map(
                            lambda _: generate_report(b"same-file", summary),
                            range(2),
                        )
                    )

        self.assertEqual(post.call_count, 1)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0].verdict, "malicious")

    def test_corrupt_cache_is_treated_as_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            (cache_dir / "broken.json").write_text("{not-json", encoding="utf-8")
            with patch("src.llm.report.CACHE_DIR", cache_dir):
                self.assertIsNone(_load_cached("broken", "identity"))

if __name__ == "__main__":
    unittest.main()
