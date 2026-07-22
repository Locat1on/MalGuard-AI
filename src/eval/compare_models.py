"""Evaluate baseline, deep model, and deployed ensemble on the official test split.

Inference is chunked directly from the test memmap. The script writes the frontend-compatible
metrics list plus a provenance manifest with checkpoint hashes, protocol, environment and Git
state.

Run: .venv\Scripts\python.exe src/eval/compare_models.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import matplotlib
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.data.load_features import DATA_DIR, _read_memmap
from src.models.mlp import MalwareMLP
from src.reproducibility import PROJECT_ROOT, artifact_manifest, git_manifest, runtime_manifest, write_json_atomic

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
METRICS_PATH = CHECKPOINT_DIR / "metrics.json"
MANIFEST_PATH = CHECKPOINT_DIR / "evaluation_manifest.json"
CONFUSION_PLOT_PATH = CHECKPOINT_DIR / "confusion_matrices.png"
INFERENCE_BATCH_SIZE = 8192
THRESHOLD = 0.5


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def predict_lightgbm(
    features: np.memmap | np.ndarray,
    indices: np.ndarray,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> np.ndarray:
    model = lgb.Booster(model_file=str(CHECKPOINT_DIR / "lightgbm.txt"))
    scores = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        scores[start:stop] = model.predict(features[indices[start:stop]])
    return scores


def predict_mlp(
    features: np.memmap | np.ndarray,
    indices: np.ndarray,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> np.ndarray:
    with (CHECKPOINT_DIR / "scaler.pkl").open("rb") as file:
        scaler = pickle.load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("mlp")
    model = MalwareMLP(
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        embed_dim=config["embed_dim"],
    ).to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "mlp.pt", map_location=device, weights_only=True)
    )
    model.eval()

    scores = np.empty(len(indices), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            values = scaler.transform(features[indices[start:stop]]).astype(np.float32, copy=False)
            batch = torch.from_numpy(values).to(device, non_blocking=True)
            logits = model(batch)
            scores[start:stop] = torch.sigmoid(logits).float().cpu().numpy()
    return scores


def _staged_evaluation_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.evaluation{path.suffix}")


def _publish_evaluation_outputs(
    results: list[dict],
    manifest: dict,
    figure,
    metrics_path: Path = METRICS_PATH,
    manifest_path: Path = MANIFEST_PATH,
    plot_path: Path = CONFUSION_PLOT_PATH,
) -> None:
    """Publish the manifest last so it is the completion marker for one evaluation."""
    staged_metrics = _staged_evaluation_path(metrics_path)
    staged_manifest = _staged_evaluation_path(manifest_path)
    staged_plot = _staged_evaluation_path(plot_path)
    staged_paths = (staged_metrics, staged_manifest, staged_plot)
    try:
        write_json_atomic(staged_metrics, results)
        figure.savefig(staged_plot, dpi=160)
        write_json_atomic(staged_manifest, manifest)

        staged_metrics.replace(metrics_path)
        staged_plot.replace(plot_path)
        staged_manifest.replace(manifest_path)
    finally:
        for path in staged_paths:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def main() -> None:
    source_git = git_manifest()
    features, targets = _read_memmap(DATA_DIR, "test")
    indices = np.flatnonzero(targets != -1)
    y_test = targets[indices]
    print(
        f"official labeled test split: rows={len(indices)}, "
        f"malicious={(y_test == 1).sum()}, benign={(y_test == 0).sum()}"
    )

    lgbm_scores = predict_lightgbm(features, indices)
    mlp_scores = predict_mlp(features, indices)
    ensemble_scores = (lgbm_scores + mlp_scores) / 2
    evaluated = [
        ("LightGBM (EMBER 静态特征基线)", lgbm_scores),
        ("MLP 深度模型 (本系统)", mlp_scores),
        ("LightGBM + MLP 集成 (部署模型)", ensemble_scores),
    ]
    results = [
        {"model": name, **compute_metrics(y_test, scores)}
        for name, scores in evaluated
    ]
    for row in results:
        print(row)

    checkpoints = [
        CHECKPOINT_DIR / "lightgbm.txt",
        CHECKPOINT_DIR / "mlp.pt",
        CHECKPOINT_DIR / "scaler.pkl",
    ]
    manifest = {
        "protocol": {
            "dataset": "EMBER2024 Win64 official labeled test split",
            "feature_dimensions": features.shape[1],
            "threshold": THRESHOLD,
            "ensemble": "arithmetic mean of LightGBM and MLP malicious probabilities",
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "test_rows": len(indices),
            "test_malicious": int((y_test == 1).sum()),
            "test_benign": int((y_test == 0).sum()),
        },
        "results": [
            {
                **row,
                "confusion_matrix": confusion_matrix(
                    y_test, (scores >= THRESHOLD).astype(int), labels=[0, 1]
                ).tolist(),
            }
            for row, (_, scores) in zip(results, evaluated)
        ],
        "artifacts": artifact_manifest(checkpoints),
        "runtime": runtime_manifest(),
        "git": source_git,
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plot_titles = ["LightGBM", "MLP", "Ensemble"]
    for axis, title, (_, scores) in zip(axes, plot_titles, evaluated):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            (scores >= THRESHOLD).astype(int),
            display_labels=["benign", "malicious"],
            ax=axis,
            colorbar=False,
        )
        axis.set_title(title)
    fig.tight_layout()
    try:
        _publish_evaluation_outputs(results, manifest, fig)
    finally:
        plt.close(fig)
    print(f"saved metrics to {METRICS_PATH}")
    print(f"saved evaluation manifest to {MANIFEST_PATH}")
    print(f"saved confusion matrices to {CONFUSION_PLOT_PATH}")


if __name__ == "__main__":
    main()
