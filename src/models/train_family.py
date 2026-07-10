"""Train an MLP multiclass family-attribution model over malicious EMBER2024 samples.

Malware family names are extremely long-tailed (EMBER2024 has 6787 distinct families dataset-
wide, most with a handful of samples). Rather than model every family, this restricts the
target space to families with at least `min_count` samples in the malicious training subset
and buckets everything else into a catch-all "其他" class — a low-confidence prediction is
still useful ("this doesn't look like any family we have good coverage of") without pretending
we can reliably distinguish families we've seen only a handful of times.

Benign rows never carry a family label (see the raw EMBER2024 schema) and are excluded
entirely — this is a malicious-only classifier, layered on top of (not replacing) the binary
LightGBM/MLP detectors.

Reuses the multi-branch attention-fusion architecture from src/models/mlp.py (same one used by
the binary detector) with its output layer widened to `num_classes`. An earlier version of this
script used LightGBM, whose native multiclass objective builds one tree per class per boosting
round — with ~440 classes that meant ~200 iterations x 440 classes ~= 88,000 trees, measured at
~2.2 hours to train. A neural net's output layer only grows by a `(hidden_dim, num_classes)`
matmul, so training here takes minutes, not hours.

Reuses checkpoints/scaler.pkl (fit by train_mlp.py on the full binary-training X_train) rather
than fitting a new scaler: load_family_split() and load_split() call train_test_split() with
the same (X_train_full, y_train_full, test_size, stratify, random_state), so X_train is bit-
identical between the two — the malicious-with-family subset trained on here is a strict subset
of the exact array that scaler.pkl was already fit on. Run train_mlp.py first if scaler.pkl
doesn't exist yet.

Writes:
  checkpoints/family_mlp.pt                    — the trained model's state_dict
  checkpoints/family_labels.json               — ordered list of class names; index i ==
                                                  predicted class i (last entry is always "其他")
  checkpoints/family_metrics.json              — top-1/top-3 accuracy, macro/weighted F1, and a
                                                  majority-class baseline for context, on the
                                                  held-out test split
  checkpoints/family_classification_report.txt — per-class precision/recall/F1/support (test)
  checkpoints/family_distribution.json         — per-class sample counts in train/val/test,
                                                  for documenting how long-tailed the label is
  checkpoints/family_confusion_matrix.png      — test-set confusion matrix

Run: .venv\\Scripts\\python.exe src/models/train_family.py
"""

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    top_k_accuracy_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.data.load_features import load_family_split
from src.models.mlp import MalwareMLP

CHECKPOINT_DIR = Path(r"D:\study\Integrated_Design\checkpoints")
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"
MODEL_PATH = CHECKPOINT_DIR / "family_mlp.pt"
LABELS_PATH = CHECKPOINT_DIR / "family_labels.json"
METRICS_PATH = CHECKPOINT_DIR / "family_metrics.json"
CLASSIFICATION_REPORT_PATH = CHECKPOINT_DIR / "family_classification_report.txt"
DISTRIBUTION_PATH = CHECKPOINT_DIR / "family_distribution.json"
CONFUSION_PLOT_PATH = CHECKPOINT_DIR / "family_confusion_matrix.png"
OTHER_LABEL = "其他"


def build_vocab(families: np.ndarray, min_count: int) -> list[str]:
    """Family names with >= min_count occurrences in `families`, most frequent first."""
    counts = Counter(f for f in families if f is not None)
    return [name for name, count in counts.most_common() if count >= min_count]


def encode(families: np.ndarray, vocab_index: dict[str, int], other_idx: int) -> np.ndarray:
    return np.array([vocab_index.get(f, other_idx) for f in families], dtype=np.int64)


def class_distribution(y_encoded: np.ndarray, labels: list[str]) -> dict[str, int]:
    """Sample counts per class name, most frequent first — documents how long-tailed the
    family label is (only meaningful info if computed *after* dropping the None-family rows
    that select_malicious_with_family() already excludes).
    """
    counts = Counter(y_encoded.tolist())
    return {labels[idx]: count for idx, count in sorted(counts.items(), key=lambda kv: -kv[1])}


def select_malicious_with_family(
    X: np.ndarray, y: np.ndarray, family: np.ndarray, vocab_index: dict[str, int], other_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Malicious rows that carry a family label, encoded to a class index.

    Rows with no family label at all (ClarAVy couldn't attribute one to that sample) carry no
    training signal and are dropped rather than forced into a class — they're neither a real
    family nor an honest example of "recognized-but-uncommon".
    """
    keep = (y == 1) & (family != None)  # noqa: E711 -- elementwise None-check on an object array
    return X[keep], encode(family[keep], vocab_index, other_idx)


def make_loader(X: np.ndarray, y: np.ndarray, scaler, batch_size: int, shuffle: bool) -> DataLoader:
    X_scaled = scaler.transform(X)
    dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_proba, all_targets = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        all_proba.append(torch.softmax(logits, dim=1).cpu())
        all_targets.append(y_batch)
    return torch.cat(all_proba).numpy(), torch.cat(all_targets).numpy()


@torch.no_grad()
def evaluate_f1(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> float:
    proba, y_true = predict_proba(model, loader, device)
    y_pred = proba.argmax(axis=1)
    return float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0))


def train() -> None:
    config = load_config("family")
    min_count = config.pop("min_count")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"{SCALER_PATH} not found — run src/models/train_mlp.py first so its fitted "
            "StandardScaler is available for this model to reuse."
        )
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    X_train, y_train, family_train, X_val, y_val, family_val, X_test, y_test, family_test = load_family_split()

    vocab = build_vocab(family_train[y_train == 1], min_count)
    vocab_index = {name: i for i, name in enumerate(vocab)}
    other_idx = len(vocab)
    labels = vocab + [OTHER_LABEL]
    num_classes = len(labels)
    print(f"family vocab: {len(vocab)} families with >= {min_count} samples in the training set (+ '{OTHER_LABEL}' catch-all)")

    X_tr, y_tr = select_malicious_with_family(X_train, y_train, family_train, vocab_index, other_idx)
    X_va, y_va = select_malicious_with_family(X_val, y_val, family_val, vocab_index, other_idx)
    X_te, y_te = select_malicious_with_family(X_test, y_test, family_test, vocab_index, other_idx)
    print(f"train={X_tr.shape[0]} val={X_va.shape[0]} test={X_te.shape[0]} malicious rows with a family label")

    train_loader = make_loader(X_tr, y_tr, scaler, config["batch_size"], shuffle=True)
    val_loader = make_loader(X_va, y_va, scaler, config["batch_size"], shuffle=False)
    test_loader = make_loader(X_te, y_te, scaler, config["batch_size"], shuffle=False)

    model = MalwareMLP(
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        embed_dim=config["embed_dim"],
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    epochs_without_improvement = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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

        val_f1 = evaluate_f1(model, val_loader, device, num_classes)
        print(f"epoch {epoch:02d}  train_loss={total_loss / len(train_loader.dataset):.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                print(f"early stopping at epoch {epoch} (best val_macro_f1={best_f1:.4f})")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    proba, y_te_eval = predict_proba(model, test_loader, device)
    y_pred = proba.argmax(axis=1)
    class_indices = list(range(num_classes))

    top1 = float(accuracy_score(y_te_eval, y_pred))
    top3 = float(top_k_accuracy_score(y_te_eval, proba, k=3, labels=class_indices))
    macro_f1 = float(f1_score(y_te_eval, y_pred, labels=class_indices, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_te_eval, y_pred, labels=class_indices, average="weighted", zero_division=0))
    other_share = float((y_te_eval == other_idx).mean())

    # Context for the accuracy numbers above: with a long-tailed, "其他"-heavy label space,
    # top-1 accuracy alone can look deceptively good just from always guessing the majority
    # class. This is the number any real model needs to beat.
    majority_class = Counter(y_te_eval.tolist()).most_common(1)[0][0]
    baseline_acc = float((y_te_eval == majority_class).mean())

    print(f"test top-1 accuracy: {top1:.4f}  (majority-class baseline: {baseline_acc:.4f})")
    print(f"test top-3 accuracy: {top3:.4f}")
    print(f"test macro-F1: {macro_f1:.4f}  weighted-F1: {weighted_f1:.4f}")
    print(f"share of test rows that are genuinely '{OTHER_LABEL}': {other_share:.4f}")

    report_text = classification_report(y_te_eval, y_pred, labels=class_indices, target_names=labels, zero_division=0)

    distribution = {
        "train": class_distribution(y_tr, labels),
        "val": class_distribution(y_va, labels),
        "test": class_distribution(y_te_eval, labels),
    }

    fig_size = max(8.0, len(labels) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    # labels=class_indices pins the matrix to the full class set (in vocab order) regardless of
    # which classes actually show up in y_te/y_pred — without it, a vocab family with zero test
    # occurrences would silently shrink the matrix and misalign it against display_labels.
    ConfusionMatrixDisplay.from_predictions(
        y_te_eval, y_pred, labels=class_indices, display_labels=labels, ax=ax, xticks_rotation="vertical"
    )
    ax.set_title("Family classifier — test set confusion matrix")
    fig.tight_layout()

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "top1_accuracy": top1,
                "top3_accuracy": top3,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "majority_class_baseline_accuracy": baseline_acc,
                "other_share_test": other_share,
                "num_classes": num_classes,
                "min_count": min_count,
                "train_samples": int(X_tr.shape[0]),
                "val_samples": int(X_va.shape[0]),
                "test_samples": int(X_te.shape[0]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(DISTRIBUTION_PATH, "w", encoding="utf-8") as f:
        json.dump(distribution, f, ensure_ascii=False, indent=2)
    fig.savefig(CONFUSION_PLOT_PATH)
    plt.close(fig)

    print(f"saved best model (val_macro_f1={best_f1:.4f}) to {MODEL_PATH}")
    print(f"saved labels to {LABELS_PATH}")
    print(f"saved metrics to {METRICS_PATH}")
    print(f"saved per-class report to {CLASSIFICATION_REPORT_PATH}")
    print(f"saved class distribution to {DISTRIBUTION_PATH}")
    print(f"saved confusion matrix to {CONFUSION_PLOT_PATH}")


if __name__ == "__main__":
    train()
