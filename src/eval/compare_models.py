"""Evaluate LightGBM and MLP on the same held-out EMBER2024 test split, side by side.

Writes checkpoints/metrics.json in the shape the FastAPI backend's /api/metrics endpoint
expects (webapp/backend/app/routers/metrics.py reads this file directly).

Run: .venv\\Scripts\\python.exe src/eval/compare_models.py
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score

from src.config import load_config
from src.data.load_features import load_split
from src.models.mlp import MalwareMLP

CHECKPOINT_DIR = Path(r"D:\study\Integrated_Design\checkpoints")
METRICS_PATH = CHECKPOINT_DIR / "metrics.json"
CONFUSION_PLOT_PATH = CHECKPOINT_DIR / "confusion_matrices.png"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def predict_lightgbm(X_test: np.ndarray) -> np.ndarray:
    model = lgb.Booster(model_file=str(CHECKPOINT_DIR / "lightgbm.txt"))
    return (model.predict(X_test) >= 0.5).astype(int)


def predict_mlp(X_test: np.ndarray) -> np.ndarray:
    with open(CHECKPOINT_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_config = load_config("mlp")
    model = MalwareMLP(
        hidden_dims=mlp_config["hidden_dims"], dropout=mlp_config["dropout"], embed_dim=mlp_config["embed_dim"]
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "mlp.pt", map_location=device))
    model.eval()

    X_scaled = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_scaled)
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
    return preds


def main() -> None:
    _, _, _, _, X_test, y_test = load_split()

    y_pred_lgbm = predict_lightgbm(X_test)
    y_pred_mlp = predict_mlp(X_test)

    results = [
        {"model": "LightGBM (EMBER 静态特征基线)", **compute_metrics(y_test, y_pred_lgbm)},
        {"model": "MLP 深度模型 (本系统)", **compute_metrics(y_test, y_pred_mlp)},
    ]

    for row in results:
        print(row)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved metrics to {METRICS_PATH}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lgbm, display_labels=["benign", "malicious"], ax=axes[0])
    axes[0].set_title("LightGBM")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_mlp, display_labels=["benign", "malicious"], ax=axes[1])
    axes[1].set_title("MLP")
    fig.tight_layout()
    fig.savefig(CONFUSION_PLOT_PATH)
    print(f"saved confusion matrices to {CONFUSION_PLOT_PATH}")


if __name__ == "__main__":
    main()
