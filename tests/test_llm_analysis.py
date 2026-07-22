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
    _load_cached,
    _save_cache,
    generate_report,
)


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

    def test_real_demo_pe_summary_contains_structural_anomalies(self) -> None:
        sample = Path("demo_samples/suspicious_demo.exe").read_bytes()
        summary = summarize(sample)
        self.assertIn(".ex0", summary.high_entropy_sections)
        self.assertIn("fothk", summary.nonstandard_sections)
        self.assertIsNotNone(summary.compile_time_anomaly)
        self.assertNotIn("5.1.0.0", summary.ip_addresses)
        self.assertIn("编译时间异常", summary.to_prompt_text())


class ReportCacheTests(unittest.TestCase):
    def test_cache_requires_current_analysis_version(self) -> None:
        analysis = LLMAnalysis("malicious", 0.8, "test")
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            with patch("src.llm.report.CACHE_DIR", cache_dir):
                _save_cache("new", analysis)
                self.assertEqual(_load_cached("new"), analysis)

                (cache_dir / "old.json").write_text(
                    json.dumps(
                        {
                            "verdict": "malicious",
                            "confidence": 0.8,
                            "narrative": "old",
                            "analysis_version": ANALYSIS_VERSION - 1,
                        }
                    ),
                    encoding="utf-8",
                )
                self.assertIsNone(_load_cached("old"))

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
                self.assertIsNone(_load_cached("broken"))

if __name__ == "__main__":
    unittest.main()
