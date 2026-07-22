"""Versioned family-checkpoint contract shared by training and inference."""

import json
from collections.abc import Mapping
from pathlib import Path

FAMILY_CHECKPOINT_FORMAT_VERSION = 1
OTHER_LABEL = "其他"


def validate_family_labels(value: object) -> list[str]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(
            "family_labels.json 必须至少包含一个已知家族和末尾的“其他”类。"
        )
    if any(not isinstance(label, str) or not label.strip() for label in value):
        raise ValueError("family_labels.json 只能包含非空字符串。")
    if len(set(value)) != len(value):
        raise ValueError("family_labels.json 包含重复标签。")
    if value[-1] != OTHER_LABEL:
        raise ValueError("family_labels.json 的最后一个标签必须是“其他”。")
    return value


def build_family_checkpoint(
    state_dict: Mapping[str, object], labels: list[str]
) -> dict:
    """Bundle class order with weights so one atomic file defines runtime semantics."""
    return {
        "format_version": FAMILY_CHECKPOINT_FORMAT_VERSION,
        "labels": validate_family_labels(labels),
        "state_dict": state_dict,
    }


def unpack_family_checkpoint(
    checkpoint: object,
    legacy_labels_path: Path,
) -> tuple[Mapping[str, object], list[str]]:
    """Read the bundled format or fall back to the existing state-dict + JSON pair."""
    if (
        isinstance(checkpoint, Mapping)
        and checkpoint.get("format_version") == FAMILY_CHECKPOINT_FORMAT_VERSION
    ):
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("family_mlp.pt 缺少有效的 state_dict。")
        return state_dict, validate_family_labels(checkpoint.get("labels"))

    if not isinstance(checkpoint, Mapping):
        raise ValueError("family_mlp.pt 不是有效的模型 checkpoint。")
    if not legacy_labels_path.exists():
        raise FileNotFoundError("旧格式 family_mlp.pt 需要 family_labels.json。")
    labels = json.loads(legacy_labels_path.read_text(encoding="utf-8"))
    return checkpoint, validate_family_labels(labels)
