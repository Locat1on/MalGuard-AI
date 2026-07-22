"""Low-concurrency replacement for thrember.create_vectorized_features().

thrember's own vectorize_subset() calls multiprocessing.Pool() unconditionally, i.e. one
worker per CPU core. On Windows, each spawned worker re-imports the entire sklearn/scipy/
lightgbm/pefile/polars/matplotlib stack from scratch (spawn, not fork), and on a 32-core
machine that's 32 processes doing this at once — which exhausted the page file and crashed
with MemoryError / "DLL load failed" partway through vectorizing the train set.

This reuses thrember's own per-row vectorize() logic (imported, not reimplemented) but with
a small, fixed worker count, so only a handful of processes pay that import cost at a time.
Skips the challenge split since it was not downloaded.

Run: .venv\\Scripts\\python.exe src/data/vectorize_low_concurrency.py
"""

import multiprocessing
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import tqdm
from thrember.features import PEFeatureExtractor
from thrember.model import gather_feature_paths, raw_feature_iterator, vectorize_unpack

DATA_DIR = Path(
    os.environ.get(
        "EMBER2024_DATA_DIR",
        Path(__file__).resolve().parents[2] / "data" / "raw" / "ember2024",
    )
)
NUM_WORKERS = 4


def vectorize_subset_low_concurrency(
    X_path: Path,
    y_path: Path,
    raw_feature_paths: list[Path],
    extractor: PEFeatureExtractor,
    nrows: int,
    num_workers: int,
) -> None:
    # Allocate the memmap files up front (same as thrember.model.vectorize_subset).
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    del X, y

    argument_iterator = (
        (irow, raw_features_string, X_path, y_path, extractor, nrows, "label", {})
        for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths))
    )
    with multiprocessing.Pool(processes=num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(vectorize_unpack, argument_iterator), total=nrows):
            pass


def vectorize_split(data_dir: Path, subset: str, extractor: PEFeatureExtractor, num_workers: int) -> None:
    X_path = data_dir / f"X_{subset}.dat"
    y_path = data_dir / f"y_{subset}.dat"
    if X_path.exists() and y_path.exists():
        print(f"Skipping {subset} set — {X_path.name} already exists (delete it first to force a rebuild).")
        return

    print(f"Vectorizing {subset} set")
    feature_paths = gather_feature_paths(data_dir, subset)
    nrows = sum(1 for fp in feature_paths for _ in fp.open())
    print(f"  {nrows} rows across {len(feature_paths)} file(s)")
    vectorize_subset_low_concurrency(X_path, y_path, feature_paths, extractor, nrows, num_workers)


def main(num_workers: int = NUM_WORKERS) -> None:
    extractor = PEFeatureExtractor()
    vectorize_split(DATA_DIR, "train", extractor, num_workers)
    vectorize_split(DATA_DIR, "test", extractor, num_workers)
    try:
        vectorize_split(DATA_DIR, "challenge", extractor, num_workers)
    except ValueError:
        print("No challenge-set .jsonl files found — skipping (download it first if you want it vectorized).")
    print("Done.")


if __name__ == "__main__":
    main()
