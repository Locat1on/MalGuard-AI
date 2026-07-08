"""LightGBM baseline over the EMBER2024 feature vector (the classical-ML accuracy-floor model).

Mirrors mlp.py's role: this module defines the model, train_lightgbm.py orchestrates
loading data / training / evaluating / saving. Hyperparameters live in configs/lightgbm.yaml,
not here — see src/config.py.
"""

import lightgbm as lgb
import numpy as np


def build_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    return lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=50)],
    )
