"""Load hyperparameter configs from configs/*.yaml so tuning doesn't require editing training scripts."""

from pathlib import Path

import yaml

CONFIGS_DIR = Path(r"D:\study\Integrated_Design\configs")


def load_config(name: str) -> dict:
    """Load configs/{name}.yaml, e.g. load_config('lightgbm') -> configs/lightgbm.yaml."""
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
