"""Small training-only helpers shared by the binary and family MLPs."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class TorchStandardizer:
    """Apply a fitted scikit-learn scaler on the training device in float32."""

    def __init__(self, scaler: StandardScaler, device: torch.device) -> None:
        if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
            raise ValueError("StandardScaler must be fitted before training")
        self.mean = torch.as_tensor(
            np.asarray(scaler.mean_, dtype=np.float32),
            device=device,
        )
        self.scale = torch.as_tensor(
            np.asarray(scaler.scale_, dtype=np.float32),
            device=device,
        )

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.device != self.mean.device:
            raise ValueError("batch and standardizer must be on the same device")
        return batch.sub_(self.mean).div_(self.scale)
