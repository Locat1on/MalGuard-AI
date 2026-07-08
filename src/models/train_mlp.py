"""Train the MLP deep model on EMBER2024 features.

Run: .venv\\Scripts\\python.exe src/models/train_mlp.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.data.load_features import load_split
from src.models.mlp import MalwareMLP

CHECKPOINT_DIR = Path(r"D:\study\Integrated_Design\checkpoints")
MODEL_PATH = CHECKPOINT_DIR / "mlp.pt"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"


def make_loader(X, y, scaler: StandardScaler, batch_size: int, shuffle: bool) -> DataLoader:
    X_scaled = scaler.transform(X)
    dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu()
        all_preds.append(preds)
        all_targets.append(y_batch.long())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def train() -> None:
    config = load_config("mlp")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    X_train, y_train, X_val, y_val, _, _ = load_split()

    scaler = StandardScaler().fit(X_train)
    train_loader = make_loader(X_train, y_train, scaler, config["batch_size"], shuffle=True)
    val_loader = make_loader(X_val, y_val, scaler, config["batch_size"], shuffle=False)

    model = MalwareMLP(
        hidden_dims=config["hidden_dims"], dropout=config["dropout"], embed_dim=config["embed_dim"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

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

        val_metrics = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:02d}  train_loss={total_loss / len(train_loader.dataset):.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            with open(SCALER_PATH, "wb") as f:
                pickle.dump(scaler, f)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                print(f"early stopping at epoch {epoch} (best val_f1={best_f1:.4f})")
                break

    print(f"saved best model (val_f1={best_f1:.4f}) to {MODEL_PATH}")


if __name__ == "__main__":
    train()
