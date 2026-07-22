"""Train the LightGBM baseline on the fixed EMBER2024 train/validation split.

Run: .venv\Scripts\python.exe src/models/train_lightgbm.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import load_config
from src.data.load_features import RANDOM_STATE, VAL_SIZE, load_train_val
from src.models.lightgbm_model import build_model
from src.reproducibility import PROJECT_ROOT, artifact_manifest, git_manifest, runtime_manifest, write_json_atomic

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "lightgbm.txt"
MANIFEST_PATH = CHECKPOINT_DIR / "lightgbm_training_manifest.json"


def train() -> None:
    source_git = git_manifest()
    params = load_config("lightgbm")
    started = time.perf_counter()
    X_train, y_train, X_val, y_val = load_train_val()
    model = build_model(X_train, y_train, X_val, y_val, params)

    y_pred = (model.predict(X_val) >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }
    print("LightGBM validation metrics:")
    for name, value in metrics.items():
        print(f"  {name + ':':10} {value:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(CHECKPOINT_PATH))
    manifest = {
        "model": "LightGBM EMBER static-feature baseline",
        "config": params,
        "seed": params["seed"],
        "split": {
            "validation_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": len(y_train),
            "validation_rows": len(y_val),
            "train_malicious": int((y_train == 1).sum()),
            "train_benign": int((y_train == 0).sum()),
        },
        "best_iteration": model.best_iteration,
        "validation_metrics": metrics,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "artifacts": artifact_manifest([CHECKPOINT_PATH]),
        "runtime": runtime_manifest(),
        "git": source_git,
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    print(f"saved model to {CHECKPOINT_PATH}")
    print(f"saved training manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    train()
