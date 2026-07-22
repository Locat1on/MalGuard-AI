import hashlib
import re
import unittest
from collections import OrderedDict

from src.features.extract import extract_features
from tests.pe_fixture import build_minimal_suspicious_pe
from thrember.features import StringExtractor


FIXTURE_FEATURE_SHA256 = (
    "9328330b3c72ad711e16ce4e1052761dc44361fd946f69a2dba31c3d62cd0106"
)


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

    def test_minimal_pe_feature_vector_contract_is_stable(self) -> None:
        vector = extract_features(build_minimal_suspicious_pe())
        self.assertEqual(vector.shape, (2568,))
        self.assertEqual(
            hashlib.sha256(vector.tobytes()).hexdigest(),
            FIXTURE_FEATURE_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
