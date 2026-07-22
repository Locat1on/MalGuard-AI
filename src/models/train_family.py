"""Train the optional malware-family classifier on malicious EMBER2024 samples.

The production path keeps feature matrices memory-mapped, standardizes one batch at a time,
and uses the same fixed train/validation boundaries and scaler as the binary MLP. The official
test split is opened only after early stopping has selected the best checkpoint.

Run: .venv\Scripts\python.exe src/models/train_family.py
"""

import gc
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.data.load_features import (
    DATA_DIR,
    RANDOM_STATE,
    VAL_SIZE,
    _labeled_train_val_indices,
    _load_family_labels,
    _read_memmap,
)
from src.models.family_checkpoint import (
    OTHER_LABEL,
    build_family_checkpoint,
    unpack_family_checkpoint,
)
from src.models.mlp import MalwareMLP
from src.models.training_utils import TorchStandardizer
from src.reproducibility import (
    artifact_manifest,
    git_manifest,
    runtime_manifest,
    set_global_seed,
    write_json_atomic,
)

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"
MODEL_PATH = CHECKPOINT_DIR / "family_mlp.pt"
LABELS_PATH = CHECKPOINT_DIR / "family_labels.json"
STAGED_MODEL_PATH = CHECKPOINT_DIR / ".family_mlp.training.pt"
STAGED_LABELS_PATH = CHECKPOINT_DIR / ".family_labels.training.json"
METRICS_PATH = CHECKPOINT_DIR / "family_metrics.json"
CLASSIFICATION_REPORT_PATH = CHECKPOINT_DIR / "family_classification_report.txt"
DISTRIBUTION_PATH = CHECKPOINT_DIR / "family_distribution.json"
CONFUSION_PLOT_PATH = CHECKPOINT_DIR / "family_confusion_matrix.png"
MANIFEST_PATH = CHECKPOINT_DIR / "family_training_manifest.json"

REMAINDER_LABEL = "其余已建模家族"
MAX_CONFUSION_CLASSES = 30


def build_vocab(families: np.ndarray, min_count: int) -> list[str]:
    """Return frequent family names in descending training-support order."""
    counts = Counter(family for family in families if family is not None)
    return [name for name, count in counts.most_common() if count >= min_count]


def encode(families: np.ndarray, vocab_index: dict[str, int], other_idx: int) -> np.ndarray:
    return np.array(
        [vocab_index.get(family, other_idx) for family in families], dtype=np.int64
    )


def class_distribution(y_encoded: np.ndarray, labels: list[str]) -> dict[str, int]:
    counts = Counter(y_encoded.tolist())
    return {
        labels[index]: count
        for index, count in sorted(counts.items(), key=lambda item: -item[1])
    }


def select_malicious_with_family(
    X: np.ndarray,
    y: np.ndarray,
    family: np.ndarray,
    vocab_index: dict[str, int],
    other_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility helper for existing controlled experiments.

    Production training uses ``select_family_indices`` so it never materializes these feature
    rows as one large array.
    """
    keep = (y == 1) & (family != None)  # noqa: E711 - elementwise object-array check
    return X[keep], encode(family[keep], vocab_index, other_idx)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    scaler,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Compatibility loader for existing controlled experiments."""
    X_scaled = scaler.transform(X)
    dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class IndexedFamilyDataset(Dataset):
    """Labels aligned to selected absolute rows in a feature memmap."""

    def __init__(
        self,
        features: np.memmap | np.ndarray,
        indices: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        if len(indices) != len(labels):
            raise ValueError("indices and labels must have equal length")
        self.features = features
        self.indices = indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[int, int]:
        return int(self.indices[item]), int(self.labels[item])


def select_family_indices(
    targets: np.ndarray,
    families: np.ndarray,
    candidate_indices: np.ndarray,
    vocab_index: dict[str, int],
    other_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select malicious labeled rows without indexing the feature matrix."""
    family_values = families[candidate_indices]
    has_family = np.fromiter(
        (value is not None for value in family_values),
        dtype=bool,
        count=len(candidate_indices),
    )
    selected = candidate_indices[(targets[candidate_indices] == 1) & has_family]
    return selected, encode(families[selected], vocab_index, other_idx)


def make_indexed_loader(
    dataset: IndexedFamilyDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    def collate(batch: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        indices, targets = zip(*batch)
        values = np.asarray(
            dataset.features[np.asarray(indices, dtype=np.int64)],
            dtype=np.float32,
        )
        return torch.from_numpy(values), torch.tensor(targets, dtype=torch.long)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate,
        pin_memory=pin_memory,
    )


@torch.inference_mode()
def predict_proba(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility probability collector used by the long-tail experiment."""
    model.eval()
    all_proba, all_targets = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        all_proba.append(torch.softmax(logits, dim=1).cpu())
        all_targets.append(y_batch)
    return torch.cat(all_proba).numpy(), torch.cat(all_targets).numpy()


@torch.inference_mode()
def predict_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    standardizer: TorchStandardizer,
    use_amp: bool,
    top_k: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Collect only labels and an aggregate top-k hit rate, not a full N x C matrix."""
    model.eval()
    all_predictions, all_targets = [], []
    top_k_hits = 0
    sample_count = 0
    for X_batch, y_batch in loader:
        X_batch = standardizer(X_batch.to(device, non_blocking=True))
        targets_device = y_batch.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(X_batch)
        predictions = logits.argmax(dim=1)
        effective_k = min(top_k, logits.shape[1])
        top_indices = logits.topk(effective_k, dim=1).indices
        top_k_hits += int(
            top_indices.eq(targets_device.unsqueeze(1)).any(dim=1).sum().item()
        )
        sample_count += len(y_batch)
        all_predictions.append(predictions.cpu())
        all_targets.append(y_batch)
    return (
        torch.cat(all_predictions).numpy(),
        torch.cat(all_targets).numpy(),
        top_k_hits / sample_count,
    )


@torch.inference_mode()
def evaluate_f1(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> float:
    proba, y_true = predict_proba(model, loader, device)
    y_pred = proba.argmax(axis=1)
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )
    )


def collapse_confusion_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    max_classes: int = MAX_CONFUSION_CLASSES,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Keep frequent test classes readable and map all remaining modeled classes together."""
    if len(labels) <= max_classes:
        return y_true, y_pred, labels

    selected = [
        index for index, _ in Counter(y_true.tolist()).most_common(max_classes)
    ]
    selected_lookup = {old: new for new, old in enumerate(selected)}
    remainder_index = len(selected)

    def collapse(values: np.ndarray) -> np.ndarray:
        return np.fromiter(
            (selected_lookup.get(int(value), remainder_index) for value in values),
            dtype=np.int64,
            count=len(values),
        )

    display_labels = [labels[index] for index in selected] + [REMAINDER_LABEL]
    return collapse(y_true), collapse(y_pred), display_labels


def _atomic_save_family_checkpoint(
    model: nn.Module, labels: list[str], path: Path
) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(build_family_checkpoint(model.state_dict(), labels), temp_path)
    temp_path.replace(path)


def train() -> None:
    source_git = git_manifest()
    config = load_config("family")
    min_count = int(config["min_count"])
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and bool(config.get("use_amp", True))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    print(f"using device: {device}; seed={seed}; amp={use_amp}")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"{SCALER_PATH} not found; run src/models/train_mlp.py first."
        )
    with SCALER_PATH.open("rb") as file:
        scaler = pickle.load(file)
    standardizer = TorchStandardizer(scaler, device)

    features, targets = _read_memmap(DATA_DIR, "train")
    families = _load_family_labels(DATA_DIR, "train")
    if len(features) != len(families):
        raise ValueError("family_train.json is not aligned with X_train.dat")
    train_candidates, val_candidates = _labeled_train_val_indices(
        targets, VAL_SIZE, RANDOM_STATE
    )
    malicious_train = train_candidates[targets[train_candidates] == 1]
    vocab = build_vocab(families[malicious_train], min_count)
    vocab_index = {name: index for index, name in enumerate(vocab)}
    other_idx = len(vocab)
    labels = vocab + [OTHER_LABEL]
    num_classes = len(labels)

    train_indices, y_train = select_family_indices(
        targets, families, train_candidates, vocab_index, other_idx
    )
    val_indices, y_val = select_family_indices(
        targets, families, val_candidates, vocab_index, other_idx
    )
    print(
        f"family vocab={num_classes} classes; train={len(train_indices)}; "
        f"validation={len(val_indices)}"
    )

    train_set = IndexedFamilyDataset(features, train_indices, y_train)
    val_set = IndexedFamilyDataset(features, val_indices, y_val)
    train_loader = make_indexed_loader(
        train_set,
        config["batch_size"],
        True,
        seed,
        device.type == "cuda",
    )
    val_loader = make_indexed_loader(
        val_set,
        config["batch_size"],
        False,
        seed,
        device.type == "cuda",
    )

    model = MalwareMLP(
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        embed_dim=config["embed_dim"],
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1: float | None = None
    best_epoch = 0
    stale_epochs = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    STAGED_MODEL_PATH.unlink(missing_ok=True)
    STAGED_LABELS_PATH.unlink(missing_ok=True)

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

        y_val_pred, y_val_eval, _ = predict_labels(
            model, val_loader, device, standardizer, use_amp=False
        )
        val_f1 = float(
            f1_score(
                y_val_eval,
                y_val_pred,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        )
        print(
            f"epoch {epoch:02d} train_loss={total_loss / len(train_set):.4f} "
            f"val_macro_f1={val_f1:.4f}"
        )
        if best_f1 is None or val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            stale_epochs = 0
            _atomic_save_family_checkpoint(model, labels, STAGED_MODEL_PATH)
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                print(
                    f"early stopping at epoch {epoch} "
                    f"(best val_macro_f1={best_f1:.4f})"
                )
                break

    if best_f1 is None:
        raise RuntimeError("training completed without a validation result")
    staged_checkpoint = torch.load(
        STAGED_MODEL_PATH, map_location=device, weights_only=True
    )
    staged_state_dict, staged_labels = unpack_family_checkpoint(
        staged_checkpoint, STAGED_LABELS_PATH
    )
    if staged_labels != labels:
        raise RuntimeError("staged family checkpoint labels changed during training")
    model.load_state_dict(staged_state_dict)

    del train_loader, val_loader, train_set, val_set, features, targets, families
    gc.collect()

    test_features, test_targets = _read_memmap(DATA_DIR, "test")
    test_families = _load_family_labels(DATA_DIR, "test")
    if len(test_features) != len(test_families):
        raise ValueError("family_test.json is not aligned with X_test.dat")
    test_candidates = np.flatnonzero(test_targets != -1)
    test_indices, y_test = select_family_indices(
        test_targets, test_families, test_candidates, vocab_index, other_idx
    )
    test_set = IndexedFamilyDataset(test_features, test_indices, y_test)
    test_loader = make_indexed_loader(
        test_set,
        config["batch_size"],
        False,
        seed,
        device.type == "cuda",
    )
    y_pred, y_test_eval, top3 = predict_labels(
        model, test_loader, device, standardizer, use_amp=False, top_k=3
    )
    class_indices = list(range(num_classes))
    top1 = float(accuracy_score(y_test_eval, y_pred))
    macro_f1 = float(
        f1_score(
            y_test_eval,
            y_pred,
            labels=class_indices,
            average="macro",
            zero_division=0,
        )
    )
    weighted_f1 = float(
        f1_score(
            y_test_eval,
            y_pred,
            labels=class_indices,
            average="weighted",
            zero_division=0,
        )
    )
    majority_class = Counter(y_test_eval.tolist()).most_common(1)[0][0]
    baseline_accuracy = float((y_test_eval == majority_class).mean())
    other_share = float((y_test_eval == other_idx).mean())

    metrics = {
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "majority_class_baseline_accuracy": baseline_accuracy,
        "other_share_test": other_share,
        "num_classes": num_classes,
        "min_count": min_count,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_indices),
    }
    distribution = {
        "train": class_distribution(y_train, labels),
        "val": class_distribution(y_val, labels),
        "test": class_distribution(y_test_eval, labels),
    }
    report_text = classification_report(
        y_test_eval,
        y_pred,
        labels=class_indices,
        target_names=labels,
        zero_division=0,
    )

    confusion_true, confusion_pred, confusion_labels = collapse_confusion_classes(
        y_test_eval, y_pred, labels
    )
    figure_size = max(8.0, min(14.0, len(confusion_labels) * 0.45))
    fig, axis = plt.subplots(figsize=(figure_size, figure_size))
    ConfusionMatrixDisplay.from_predictions(
        confusion_true,
        confusion_pred,
        labels=list(range(len(confusion_labels))),
        display_labels=confusion_labels,
        ax=axis,
        xticks_rotation="vertical",
        colorbar=False,
    )
    axis.set_title("Family classifier - frequent test classes")
    axis.tick_params(axis="both", labelsize=6)
    fig.tight_layout()

    write_json_atomic(STAGED_LABELS_PATH, labels)
    write_json_atomic(METRICS_PATH, metrics)
    write_json_atomic(DISTRIBUTION_PATH, distribution)
    CLASSIFICATION_REPORT_PATH.write_text(report_text, encoding="utf-8")
    fig.savefig(CONFUSION_PLOT_PATH, dpi=160)
    plt.close(fig)

    # Publish only after training, official-test evaluation, and report generation succeed.
    # The bundled model already contains its label order, so an interruption between these
    # two replacements cannot make runtime family names refer to the wrong output neurons.
    STAGED_MODEL_PATH.replace(MODEL_PATH)
    STAGED_LABELS_PATH.replace(LABELS_PATH)

    artifacts = [
        MODEL_PATH,
        LABELS_PATH,
        METRICS_PATH,
        CLASSIFICATION_REPORT_PATH,
        DISTRIBUTION_PATH,
        CONFUSION_PLOT_PATH,
        SCALER_PATH,
    ]
    manifest = {
        "model": "MalwareMLP family classifier",
        "config": config,
        "seed": seed,
        "split": {
            "validation_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
            "train_samples": len(train_indices),
            "validation_samples": len(val_indices),
            "test_samples": len(test_indices),
        },
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "test_metrics": metrics,
        "confusion_matrix_scope": {
            "max_frequent_classes": MAX_CONFUSION_CLASSES,
            "displayed_classes": len(confusion_labels),
            "remainder_label": REMAINDER_LABEL,
        },
        "duration_seconds": round(time.perf_counter() - started, 2),
        "artifacts": artifact_manifest(artifacts),
        "runtime": runtime_manifest(),
        "git": source_git,
    }
    write_json_atomic(MANIFEST_PATH, manifest)

    print(
        f"test top1={top1:.4f}; top3={top3:.4f}; macro_f1={macro_f1:.4f}; "
        f"weighted_f1={weighted_f1:.4f}"
    )
    print(f"saved best model (val_macro_f1={best_f1:.4f}) to {MODEL_PATH}")
    print(f"saved training manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    train()
