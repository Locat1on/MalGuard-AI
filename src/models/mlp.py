"""Feature-group attention-fusion classifier over the EMBER2024 feature vector.

The 2568-dim EMBER vector concatenates 12 semantically distinct feature groups (byte
histogram, import hashing, section info, ...). Rather than treating it as one flat vector,
each group is first encoded by its own small branch network, then the branch embeddings are
combined via a learned attention-weighted fusion — the same fusion pattern used in the
multi-modal malware detection literature (e.g. DMLDroid's dynamic weighted fusion, HMSF-ADM's
heterogeneous semantic fusion), adapted here to EMBER's feature-group structure since we only
have a single static feature vector, not separate image/API-sequence modalities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from src.features.extract import FEATURE_DIM, SEGMENT_DIMS


class FeatureGroupBranch(nn.Module):
    """Encodes one EMBER feature group (e.g. byte histogram) into a fixed-size embedding."""

    def __init__(self, input_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MalwareMLP(nn.Module):
    """Binary malicious/benign classifier with attention-weighted fusion over EMBER feature groups.

    hidden_dims/dropout/embed_dim are hyperparameters — see configs/mlp.yaml, loaded via src/config.py.
    """

    def __init__(self, hidden_dims: list[int], dropout: float, embed_dim: int, input_dim: int = FEATURE_DIM):
        super().__init__()
        if input_dim != FEATURE_DIM:
            raise ValueError(f"input_dim must equal FEATURE_DIM ({FEATURE_DIM}) for feature-group splitting")

        self.segment_sizes = [dim for _, dim in SEGMENT_DIMS]
        self.branches = nn.ModuleList(
            [FeatureGroupBranch(dim, embed_dim, dropout) for _, dim in SEGMENT_DIMS]
        )
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

        classifier_layers: list[nn.Module] = []
        prev_dim = embed_dim
        for hidden_dim in hidden_dims:
            classifier_layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim
        classifier_layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        segments = torch.split(x, self.segment_sizes, dim=1)
        branch_embeds = torch.stack(
            [branch(seg) for branch, seg in zip(self.branches, segments)], dim=1
        )  # (batch, num_branches, embed_dim)

        attn_logits = self.attention(branch_embeds).squeeze(-1)  # (batch, num_branches)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)  # (batch, num_branches, 1)
        fused = (branch_embeds * attn_weights).sum(dim=1)  # (batch, embed_dim)

        return self.classifier(fused).squeeze(-1)
