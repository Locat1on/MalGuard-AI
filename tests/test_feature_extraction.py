import hashlib
import re
import unittest
from collections import OrderedDict
from pathlib import Path

from src.features.extract import extract_features
from thrember.features import StringExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PE = PROJECT_ROOT / "demo_samples" / "suspicious_demo.exe"
DEMO_FEATURE_SHA256 = "b16a909b1dc7e8b2cefcaba6f69766b8be833945550c37796820e668cceb3f48"


class StringExtractionPatchTests(unittest.TestCase):
    def test_compiled_pattern_search_preserves_thrember_counts(self) -> None:
        content = (
            b"prefix powershell HTTP/1.1 https://example.test/path "
            b"CreateProcess hidden window\x00"
            b"second powershell string with C:/temp/file.exe\x00"
        )
        extractor = StringExtractor()
        raw = extractor.raw_features(content, None)
        strings = [value.decode() for value in extractor._allstrings.findall(content)]
        expected: dict[str, int] = {}
        for value in strings:
            for name, pattern in extractor._regexes.items():
                if re.search(pattern, value):
                    expected[name] = expected.get(name, 0) + 1

        self.assertEqual(raw["numstrings"], len(strings))
        self.assertEqual(
            raw["string_counts"],
            OrderedDict(sorted(expected.items())),
        )

    def test_demo_pe_feature_vector_contract_is_unchanged(self) -> None:
        vector = extract_features(DEMO_PE.read_bytes())
        self.assertEqual(vector.shape, (2568,))
        self.assertEqual(
            hashlib.sha256(vector.tobytes()).hexdigest(),
            DEMO_FEATURE_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
