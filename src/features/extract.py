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


DOS_HEADER_SIZE = 64
PE_POINTER_OFFSET = 0x3C
PE_SIGNATURE = b"PE\x00\x00"
COFF_HEADER_SIZE = 20


def validate_pe_structure(file_bytes: bytes) -> None:
    """Reject arbitrary bytes before the permissive EMBER extractor can vectorize them."""
    if len(file_bytes) < DOS_HEADER_SIZE or file_bytes[:2] != b"MZ":
        raise ValueError("缺少有效的 DOS MZ 文件头")
    pe_offset = int.from_bytes(
        file_bytes[PE_POINTER_OFFSET : PE_POINTER_OFFSET + 4],
        byteorder="little",
        signed=False,
    )
    minimum_nt_header_end = pe_offset + len(PE_SIGNATURE) + COFF_HEADER_SIZE
    if pe_offset < DOS_HEADER_SIZE or minimum_nt_header_end > len(file_bytes):
        raise ValueError("PE 头偏移超出文件边界")
    if file_bytes[pe_offset : pe_offset + len(PE_SIGNATURE)] != PE_SIGNATURE:
        raise ValueError("缺少有效的 PE NT 签名")


def extract_features(file_bytes: bytes) -> np.ndarray:
    """Return the EMBER2024 feature vector for structurally valid PE bytes."""
    validate_pe_structure(file_bytes)
    return _extractor.feature_vector(file_bytes)


if __name__ == "__main__":
    # Smoke test against a real, benign system executable (static parsing only, never executed).
    sample = Path(r"C:\Windows\System32\notepad.exe").read_bytes()
    vec = extract_features(sample)
    print(f"feature dim: {FEATURE_DIM}")
    print(f"vector shape: {vec.shape}, dtype: {vec.dtype}")
    print(f"nonzero entries: {int((vec != 0).sum())} / {vec.shape[0]}")
