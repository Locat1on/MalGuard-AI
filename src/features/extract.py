"""Live-inference feature extraction: raw PE bytes -> EMBER2024 feature vector.

This is the module that makes "upload a real exe, get a detection result" work without a
sandbox: `thrember.PEFeatureExtractor` parses the file statically (via `pefile`) and produces
the same fixed-length vector used to build the training set, so the trained models can be
run directly on it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import src.features.thrember_patches  # noqa: F401 — applies the Authenticode parsing fix before use
from thrember import PEFeatureExtractor

_extractor = PEFeatureExtractor()

FEATURE_DIM = _extractor.dim

# The 2568-dim vector is a concatenation of 12 semantically distinct feature groups
# (byte histogram, import hashing, section info, ...). Exposed so models can treat each
# group as a separate branch instead of one flat vector. Order matches feature_vector()'s
# output order exactly, since both come from the same extractor.features list.
SEGMENT_DIMS: list[tuple[str, int]] = [(fe.name, fe.dim) for fe in _extractor.features]


def extract_features(file_bytes: bytes) -> np.ndarray:
    """Return the EMBER2024 feature vector (shape (FEATURE_DIM,), dtype float32) for a PE file's raw bytes."""
    return _extractor.feature_vector(file_bytes)


if __name__ == "__main__":
    # Smoke test against a real, benign system executable (static parsing only, never executed).
    sample = Path(r"C:\Windows\System32\notepad.exe").read_bytes()
    vec = extract_features(sample)
    print(f"feature dim: {FEATURE_DIM}")
    print(f"vector shape: {vec.shape}, dtype: {vec.dtype}")
    print(f"nonzero entries: {int((vec != 0).sum())} / {vec.shape[0]}")
