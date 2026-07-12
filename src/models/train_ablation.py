"""Run controlled, memory-safe ablations for the EMBER2024 feature-group MLP.

The existing scaler.pkl is reused because train_mlp.py fitted it on the same fixed
training split. Features remain memmapped and are standardized one batch at a time,
so the three-model experiment does not duplicate the 10+ GB training matrix.

Run: .venv\Scripts\python.exe src/models/train_ablation.py
"""

import gc
import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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
from src.features.extract import FEATURE_DIM
from src.models.mlp import MalwareMLP

CHECKPOINT_DIR = Path(r"D:\study\Integrated_Design\checkpoints\ablation")
SCALER_PATH = Path(r"D:\study\Integrated_Design\checkpoints\scaler.pkl")
SEED = 42


class GroupMeanMLP(MalwareMLP):
    """Production group encoder with uniform mean fusion instead of attention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        segments = torch.split(x, self.segment_sizes, dim=1)
        padded = torch.stack(
            [F.pad(seg, (0, self.max_seg_size - seg.size(1))) for seg in segments], dim=1
        )
        embeds = torch.einsum("bnm,nmp->bnp", padded, self.branch_weights)
        embeds = embeds + self.branch_biases.unsqueeze(0)
        embeds = self.branch_bn(embeds.reshape(batch_size, -1))
        embeds = F.relu(embeds)
        embeds = self.dropout_layer(embeds).reshape(batch_size, self.num_branches, self.embed_dim)
        return self.classifier(embeds.mean(dim=1)).squeeze(-1)


class PlainMLP(nn.Module):
    """Flat MLP with the same embedding and classifier widths as group models."""

    def __init__(self, hidden_dims: list[int], dropout: float, embed_dim: int):
        super().__init__()
        layers, previous = [], FEATURE_DIM
        for width in [embed_dim, *hidden_dims]:
            layers += [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


class IndexedMemmapDataset(Dataset):
    """Indices into one feature memmap; collate_fn does the batched scaling."""

    def __init__(self, features: np.memmap, targets: np.ndarray, indices: np.ndarray):
        self.features = features
        self.targets = targets
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[int, float]:
        index = int(self.indices[item])
        return index, float(self.targets[index])


@dataclass(frozen=True)
class Variant:
    name: str
    checkpoint_name: str
    model_type: type[nn.Module]


VARIANTS = [
    Variant("普通 MLP（2568 维直接输入）", "plain_mlp.pt", PlainMLP),
    Variant("分组 MLP（均值融合）", "group_mean_mlp.pt", GroupMeanMLP),
    Variant("分组 MLP（注意力融合）", "group_attention_mlp.pt", MalwareMLP),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: IndexedMemmapDataset, scaler, batch_size: int, shuffle: bool) -> DataLoader:
    def collate(batch: list[tuple[int, float]]) -> tuple[torch.Tensor, torch.Tensor]:
        indices, targets = zip(*batch)
        X = scaler.transform(dataset.features[np.asarray(indices)])
        return torch.from_numpy(X), torch.tensor(targets, dtype=torch.float32)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=collate,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    scores, targets = [], []
    for X_batch, y_batch in loader:
        scores.append(torch.sigmoid(model(X_batch.to(device))).cpu())
        targets.append(y_batch)
    y_score = torch.cat(scores).numpy()
    y_true = torch.cat(targets).numpy().astype(int)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def train_variant(variant: Variant, config: dict, datasets: tuple, scaler, device: torch.device) -> dict:
    set_seed(SEED)
    train_set, val_set, test_set = datasets
    model = variant.model_type(
        hidden_dims=config["hidden_dims"], dropout=config["dropout"], embed_dim=config["embed_dim"]
    ).to(device)
    train_loader = make_loader(train_set, scaler, config["batch_size"], shuffle=True)
    val_loader = make_loader(val_set, scaler, config["batch_size"], shuffle=False)
    test_loader = make_loader(test_set, scaler, config["batch_size"], shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    checkpoint_path = CHECKPOINT_DIR / variant.checkpoint_name
    best_f1, stale_epochs = -1.0, 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        validation = evaluate(model, val_loader, device)
        print(
            f"{variant.name} | epoch {epoch:02d} | "
            f"loss={total_loss / len(train_set):.4f} | val_f1={validation['f1']:.4f}"
        )
        if validation["f1"] > best_f1:
            best_f1, stale_epochs = validation["f1"], 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return {
        "model": variant.name,
        "checkpoint": checkpoint_path.name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_f1": best_f1,
        "test": evaluate(model, test_loader, device),
    }


def train() -> None:
    config = load_config("mlp")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing train-fitted scaler: {SCALER_PATH}")
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)
    if scaler.n_features_in_ != FEATURE_DIM:
        raise ValueError(f"Scaler dimension {scaler.n_features_in_} does not match {FEATURE_DIM} features")

    print(f"using device: {device}")
    print("opening fixed EMBER2024 memmaps...")
    train_features, train_targets = _read_memmap(DATA_DIR, "train")
    train_indices, val_indices = _labeled_train_val_indices(train_targets, VAL_SIZE, RANDOM_STATE)
    test_features, test_targets = _read_memmap(DATA_DIR, "test")
    test_indices = np.flatnonzero(test_targets != -1)
    datasets = (
        IndexedMemmapDataset(train_features, train_targets, train_indices),
        IndexedMemmapDataset(train_features, train_targets, val_indices),
        IndexedMemmapDataset(test_features, test_targets, test_indices),
    )
    print(f"data ready: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    results = []
    for variant in VARIANTS:
        results.append(train_variant(variant, config, datasets, scaler, device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    payload = {
        "protocol": {
            "dataset": "EMBER2024 Win64 static PE features (2568 dimensions)",
            "train_validation_split": f"stratified {1 - VAL_SIZE:.0%}/{VAL_SIZE:.0%}, random_state={RANDOM_STATE}",
            "test_set": "official labeled test split; used once after validation checkpoint selection",
            "seed": SEED,
            "scaler": "existing scaler.pkl, fitted on the fixed training split only",
            "training": {key: config[key] for key in ("epochs", "batch_size", "learning_rate", "patience", "hidden_dims", "dropout", "embed_dim")},
        },
        "results": results,
    }
    output_path = CHECKPOINT_DIR / "ablation_metrics.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved metrics to {output_path}")
    for result in results:
        metrics = result["test"]
        print(f"{result['model']}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, auc={metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    train()