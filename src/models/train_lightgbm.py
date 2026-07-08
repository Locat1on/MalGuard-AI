"""Train the LightGBM baseline on EMBER2024 features (the accuracy-floor comparison for the MLP).

Run: .venv\\Scripts\\python.exe src/models/train_lightgbm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import load_config
from src.data.load_features import load_split
from src.models.lightgbm_model import build_model

CHECKPOINT_PATH = Path(r"D:\study\Integrated_Design\checkpoints\lightgbm.txt")


def train() -> None:
    params = load_config("lightgbm")
    X_train, y_train, X_val, y_val, _, _ = load_split()

    model = build_model(X_train, y_train, X_val, y_val, params)

    y_pred = (model.predict(X_val) >= 0.5).astype(int)
    print("LightGBM validation metrics:")
    print(f"  accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"  precision: {precision_score(y_val, y_pred):.4f}")
    print(f"  recall:    {recall_score(y_val, y_pred):.4f}")
    print(f"  f1:        {f1_score(y_val, y_pred):.4f}")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(CHECKPOINT_PATH))
    print(f"saved model to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
