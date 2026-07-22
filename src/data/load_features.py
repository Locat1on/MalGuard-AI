"""Load vectorized EMBER2024 features and produce a fixed, reusable train/val/test split.

The split helpers share one index-building function, so LightGBM, MLP and family training use
identical train/validation boundaries. Callers can load train/validation or test independently
to avoid materializing data they do not need.

Memory note: X_train.dat is ~10.7 GB (1.04M rows x 2568 float32). thrember's
`read_vectorized_features` opens it as a memmap but then does `np.array(X)`, which forces the
whole thing into RAM; the boolean `y != -1` mask and `train_test_split` then each make more
full-size copies, several alive at once (peak ~20-25 GB). To avoid that, this module keeps X as
a read-only memmap, splits *indices* only, and materializes each output array exactly once by
fancy-indexing the memmap — cutting peak RAM to roughly the size of the arrays actually returned.
The returned splits are bit-identical to the previous np.array-based implementation (same rows,
same order — see `_labeled_train_val_indices`), so trained-model metrics and the scaler-row
alignment that `train_family.py` depends on are unaffected.
"""

import gc
import json
import os
from pathlib import Path

import numpy as np
import thrember
from sklearn.model_selection import train_test_split

DATA_DIR = os.environ.get(
    "EMBER2024_DATA_DIR",
    str(Path(__file__).resolve().parents[2] / "data" / "raw" / "ember2024"),
)
VAL_SIZE = 0.1
RANDOM_STATE = 42

_NDIM: int | None = None


def _feature_dim() -> int:
    """EMBER feature width (2568), cached — same value thrember uses to reshape the raw .dat."""
    global _NDIM
    if _NDIM is None:
        _NDIM = thrember.PEFeatureExtractor().dim
    return _NDIM


def _read_memmap(data_dir: str, subset: str) -> tuple[np.memmap, np.ndarray]:
    """X as a read-only memmap of shape (N, 2568) — NOT copied into RAM — plus y fully loaded
    (y is small: 4 bytes/row). Rows are materialized only when the caller fancy-indexes X."""
    path = Path(data_dir)
    ndim = _feature_dim()
    X = np.memmap(path / f"X_{subset}.dat", dtype=np.float32, mode="r").reshape(-1, ndim)
    y = np.array(np.memmap(path / f"y_{subset}.dat", dtype=np.int32, mode="r"))
    return X, y


def _labeled_train_val_indices(
    y_train_full: np.ndarray, val_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/val split over the *labeled* (`y != -1`) rows, returned as index arrays
    into the original (unmasked) row order.

    Splitting the index array (rather than the data) with the same length, stratify labels and
    random_state yields exactly the same partition train_test_split would have produced on the
    data itself — this is the single source of truth both load_split() and load_family_split()
    use, so their splits are identical by construction.
    """
    labeled = np.flatnonzero(y_train_full != -1)
    train_idx, val_idx = train_test_split(
        labeled,
        test_size=val_size,
        stratify=y_train_full[labeled],
        random_state=random_state,
    )
    return train_idx, val_idx


def load_train_val(
    data_dir: str = DATA_DIR,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize only the fixed labeled train/validation partition."""
    features, targets = _read_memmap(data_dir, "train")
    train_idx, val_idx = _labeled_train_val_indices(targets, val_size, random_state)
    X_train, X_val = features[train_idx], features[val_idx]
    y_train, y_val = targets[train_idx], targets[val_idx]
    del features
    gc.collect()
    return X_train, y_train, X_val, y_val


def load_test(data_dir: str = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Materialize only labeled rows from the official test split."""
    features, targets = _read_memmap(data_dir, "test")
    indices = np.flatnonzero(targets != -1)
    X_test, y_test = features[indices], targets[indices]
    del features
    gc.collect()
    return X_test, y_test


def load_split(
    data_dir: str = DATA_DIR,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed train/validation partition and official labeled test split."""
    X_train, y_train, X_val, y_val = load_train_val(data_dir, val_size, random_state)
    X_test, y_test = load_test(data_dir)
    return X_train, y_train, X_val, y_val, X_test, y_test


def _load_family_labels(data_dir: str, subset: str) -> np.ndarray:
    with open(Path(data_dir) / f"family_{subset}.json", encoding="utf-8") as file:
        return np.array(json.load(file), dtype=object)

def load_family_split(
    data_dir: str = DATA_DIR,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same split boundaries as `load_split()`, plus the per-row `family` label (str | None).

    `family_{train,test}.json` (written by src/data/extract_family_labels.py) are row-aligned
    with the raw X_train/X_test arrays *before* the `y != -1` mask. X, y and family are all
    indexed by the same train_idx/val_idx (shared with load_split via
    `_labeled_train_val_indices`), so the three stay aligned by construction — and aligned with
    load_split's arrays, which is what lets train_family.py reuse the scaler fit by train_mlp.py.

    Only malicious rows (`y == 1`) carry a real family name; benign rows always have
    `family is None`. Callers that want a family-classification training set should further
    filter by `y == 1` (see src/models/train_family.py) — this function just returns the full
    split so `y` stays available for that filter.

    Returns (X_train, y_train, family_train, X_val, y_val, family_val, X_test, y_test, family_test).
    """
    X_train_mm, y_train_full = _read_memmap(data_dir, "train")
    family_train_full = _load_family_labels(data_dir, "train")
    train_idx, val_idx = _labeled_train_val_indices(y_train_full, val_size, random_state)
    X_train, X_val = X_train_mm[train_idx], X_train_mm[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
    family_train, family_val = family_train_full[train_idx], family_train_full[val_idx]
    del X_train_mm
    gc.collect()

    X_test_mm, y_test_full = _read_memmap(data_dir, "test")
    family_test_full = _load_family_labels(data_dir, "test")
    test_idx = np.flatnonzero(y_test_full != -1)
    X_test, y_test = X_test_mm[test_idx], y_test_full[test_idx]
    family_test = family_test_full[test_idx]
    del X_test_mm
    gc.collect()

    return X_train, y_train, family_train, X_val, y_val, family_val, X_test, y_test, family_test


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_split()
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        malicious = int((y == 1).sum())
        benign = int((y == 0).sum())
        print(f"{name}: X={X.shape} malicious={malicious} benign={benign}")
