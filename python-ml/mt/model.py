"""
Multi-Tower / Two-Tower Model

Architecture:
  One independent MLP (Multi-Layer Perceptron) tower for each entity type.
  Input -> LazyLinear(HIDDEN) -> ReLU -> Dropout -> Linear(EMBED) -> L2 Norm

Why Multi-Tower:
  - 100% Graph-free during inference (perfect for production cold-starts).
  - Extremely fast execution and O(1) memory overhead.
  - All entities are projected into the same euclidean space (comparable via
    Cosine Similarity).
"""

from __future__ import annotations
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mt.config import (
    HIDDEN_DIM, EMBED_DIM, DROPOUT,
    NODE_TYPES,
)

# We use a separate weights file for the Multi-Tower to avoid conflict with GNN
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "mt_weights.pt")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_AUTOCAST = device.type in ("mps", "cuda")
_DTYPE    = torch.bfloat16 if _AUTOCAST else torch.float32


class MultiTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Instead of message passing, we just have independent towers.
        # We use nn.ModuleDict to store a Tower (Sequential module) for each node type.
        
        self.towers = nn.ModuleDict()
        for t in NODE_TYPES:
            # LazyLinear(-1, HIDDEN) avoids needing to know exact input feature dims
            self.towers[t] = nn.Sequential(
                nn.LazyLinear(HIDDEN_DIM),
                nn.LayerNorm(HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, EMBED_DIM),
            )

    def forward(self, x: torch.Tensor, node_type: str) -> torch.Tensor:
        """
        Encode entities using their specific tower.
        Returns L2-normalised embeddings [N, EMBED_DIM].
        """
        h = self.towers[node_type](x)
        return F.normalize(h, dim=-1)

    def forward_all(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Encode a batch of multiple entity types at once (used during training).
        Returns {node_type: L2-normalised embeddings [N, EMBED_DIM]}.
        """
        return {
            t: self.forward(x, t)
            for t, x in x_dict.items()
        }


# ── Persistence ─────────────────────────────────────────────────────────────

def save_model(model: MultiTower, path: str = MODEL_WEIGHTS_PATH) -> None:
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")


def load_model(path: str = MODEL_WEIGHTS_PATH) -> Optional[MultiTower]:
    if not os.path.exists(path):
        return None
    model = MultiTower().to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    except (RuntimeError, Exception):
        print("Warning: saved weights incompatible with current architecture — discarding.")
        os.remove(path)
        return None
    model.eval()
    return model
