"""Train the production feature-group MLP with bounded host-memory use.

The EMBER feature files remain memory-mapped. StandardScaler is fitted incrementally and
features are standardized one batch at a time, so training never materializes a second
936000 x 2568 matrix in RAM.

Run: .venv\Scripts\python.exe src/models/train_mlp.py
"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import load_config
from src.data.load_features import (
    DATA_DIR,
    RANDOM_STATE,
    VAL_SIZE,
    _labeled_train_val_indices,
    _read_memmap,
)
from src.models.mlp import MalwareMLP
from src.models.training_utils import TorchStandardizer
from src.reproducibility import (
    PROJECT_ROOT,
    artifact_manifest,
    git_manifest,
    runtime_manifest,
    set_global_seed,
    write_json_atomic,
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "mlp.pt"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"
MANIFEST_PATH = CHECKPOINT_DIR / "mlp_training_manifest.json"
SCALER_FIT_BATCH_SIZE = 8192


class IndexedMemmapDataset(Dataset):
    def __init__(self, features: np.memmap, targets: np.ndarray, indices: np.ndarray):
        self.features = features
        self.targets = targets
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[int, float]:
        index = int(self.indices[item])
        return index, float(self.targets[index])


def fit_scaler_incrementally(
    features: np.memmap | np.ndarray,
    indices: np.ndarray,
    batch_size: int = SCALER_FIT_BATCH_SIZE,
) -> StandardScaler:
    scaler = StandardScaler()
    for start in range(0, len(indices), batch_size):
        scaler.partial_fit(features[indices[start : start + batch_size]])
    return scaler


def make_loader(
    dataset: IndexedMemmapDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    def collate(batch: list[tuple[int, float]]) -> tuple[torch.Tensor, torch.Tensor]:
        indices, targets = zip(*batch)
        values = np.asarray(
            dataset.features[np.asarray(indices, dtype=np.int64)],
            dtype=np.float32,
        )
        return torch.from_numpy(values), torch.tensor(targets, dtype=torch.float32)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    standardizer: TorchStandardizer,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch = standardizer(X_batch.to(device, non_blocking=True))
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(X_batch)
        all_preds.append((torch.sigmoid(logits) >= 0.5).long().cpu())
        all_targets.append(y_batch.long())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train() -> None:
    source_git = git_manifest()
    config = load_config("mlp")
    seed = int(config["seed"])
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and bool(config.get("use_amp", True))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    print(f"using device: {device}; seed={seed}; amp={use_amp}")
    print("opening fixed EMBER2024 training memmap...")

    features, targets = _read_memmap(DATA_DIR, "train")
    train_indices, val_indices = _labeled_train_val_indices(targets, VAL_SIZE, RANDOM_STATE)
    print(f"data ready: train={len(train_indices)}, val={len(val_indices)}")
    print("fitting StandardScaler incrementally...")
    scaler = fit_scaler_incrementally(features, train_indices)
    standardizer = TorchStandardizer(scaler, device)

    train_set = IndexedMemmapDataset(features, targets, train_indices)
    val_set = IndexedMemmapDataset(features, targets, val_indices)
    train_loader = make_loader(
        train_set, config["batch_size"], True, seed, device.type == "cuda"
    )
    val_loader = make_loader(
        val_set, config["batch_size"], False, seed, device.type == "cuda"
    )

    model = MalwareMLP(
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        embed_dim=config["embed_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    stale_epochs = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = standardizer(X_batch.to(device, non_blocking=True))
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(X_batch), y_batch)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            total_loss += loss.item() * X_batch.size(0)

        val_metrics = evaluate(model, val_loader, device, standardizer, use_amp)
        print(
            f"epoch {epoch:02d}  train_loss={total_loss / len(train_set):.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['f1']:.4f}"
        )
        if best_metrics is None or val_metrics["f1"] > best_metrics["f1"]:
            best_metrics = val_metrics
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), MODEL_PATH)
            with SCALER_PATH.open("wb") as file:
                pickle.dump(scaler, file)
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                print(f"early stopping at epoch {epoch} (best val_f1={best_metrics['f1']:.4f})")
                break

    if best_metrics is None:
        raise RuntimeError("training completed without a validation result")
    manifest = {
        "model": "MalwareMLP feature-group attention fusion",
        "config": config,
        "seed": seed,
        "split": {
            "validation_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": len(train_indices),
            "validation_rows": len(val_indices),
            "train_malicious": int((targets[train_indices] == 1).sum()),
            "train_benign": int((targets[train_indices] == 0).sum()),
        },
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "artifacts": artifact_manifest([MODEL_PATH, SCALER_PATH]),
        "runtime": runtime_manifest(),
        "git": source_git,
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    print(f"saved best model (val_f1={best_metrics['f1']:.4f}) to {MODEL_PATH}")
    print(f"saved training manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    train()
