"""Load vectorized EMBER2024 features and produce a fixed, reusable train/val/test split.

Both the LightGBM baseline and the MLP model must import `load_split()` rather than each
carving their own split, so their reported metrics are comparable on identical data.
"""

import numpy as np
import thrember
from sklearn.model_selection import train_test_split

DATA_DIR = r"D:\study\Integrated_Design\data\raw\ember2024"
VAL_SIZE = 0.1
RANDOM_STATE = 42


def load_split(
    data_dir: str = DATA_DIR,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_val, y_val, X_test, y_test).

    `y == -1` (unlabeled) rows are dropped from both train and test before splitting.
    The train/val split is stratified and uses a fixed random_state so repeated calls
    (from different training scripts) get the identical split.
    """
    X_train_full, y_train_full = thrember.read_vectorized_features(data_dir, subset="train")
    train_mask = y_train_full != -1
    X_train_full, y_train_full = X_train_full[train_mask], y_train_full[train_mask]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        stratify=y_train_full,
        random_state=random_state,
    )

    X_test, y_test = thrember.read_vectorized_features(data_dir, subset="test")
    test_mask = y_test != -1
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_split()
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        malicious = int((y == 1).sum())
        benign = int((y == 0).sum())
        print(f"{name}: X={X.shape} malicious={malicious} benign={benign}")
