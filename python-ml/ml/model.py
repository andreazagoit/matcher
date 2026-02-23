"""
HetEncoder — Heterogeneous Graph Transformer using PyG HGTConv.

Architecture:
  Type-specific input projections (User/Event/Space features → HIDDEN_DIM)
  → HGT_LAYERS × HGTConv  (graph-aware cross-type attention)
  → Output projection (HIDDEN_DIM → EMBED_DIM) + L2 normalisation

Two forward modes:
  forward_graph(x_dict, edge_index_dict)  — training and ml:sync (full graph)
  forward_single(x, node_type)            — /embed endpoint (new entities, no graph)
"""

from __future__ import annotations
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

from ml.config import (
    USER_DIM, EVENT_DIM, SPACE_DIM, TAG_DIM,
    HIDDEN_DIM, EMBED_DIM, DROPOUT,
    NODE_TYPES, METADATA,
    HGT_HEADS, HGT_LAYERS,
    MODEL_WEIGHTS_PATH,
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_AUTOCAST = device.type in ("mps", "cuda")
_DTYPE    = torch.bfloat16 if _AUTOCAST else torch.float32

_INPUT_DIM: dict[str, int] = {
    "user":  USER_DIM,
    "event": EVENT_DIM,
    "space": SPACE_DIM,
    "tag":   TAG_DIM,
}


class HetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Per-type input projections: raw features → shared hidden space
        self.input_proj = nn.ModuleDict({
            t: nn.Linear(_INPUT_DIM[t], HIDDEN_DIM) for t in NODE_TYPES
        })
        self.input_norm = nn.ModuleDict({
            t: nn.LayerNorm(HIDDEN_DIM) for t in NODE_TYPES
        })
        self.drop = nn.Dropout(DROPOUT)

        # HGT attention layers — cross-type message passing
        self.convs = nn.ModuleList([
            HGTConv(HIDDEN_DIM, HIDDEN_DIM, METADATA, heads=HGT_HEADS)
            for _ in range(HGT_LAYERS)
        ])

        # Shared output projection → embedding space
        self.out_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
        )

    # ── Shared input encoding ──────────────────────────────────────────────────

    def _encode_inputs(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            t: self.drop(F.relu(self.input_norm[t](self.input_proj[t](x))))
            for t, x in x_dict.items()
        }

    # ── Full-graph forward (training + ml:sync) ────────────────────────────────

    def forward_graph(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Run HGT over the full heterogeneous graph.
        Returns {node_type: L2-normalised embeddings [N, EMBED_DIM]}.
        """
        h = self._encode_inputs(x_dict)
        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            # Residual: add previous state without ReLU — HGT attention already
            # applies softmax normalization internally; extra ReLU kills negative
            # contributions from cross-type attention.
            h = {t: h_new[t] + h[t] for t in h}
        return {t: F.normalize(self.out_proj(v), dim=-1) for t, v in h.items()}

    # ── Encoder-only forward (/embed — new entities, no graph) ────────────────

    def forward_single(self, x: torch.Tensor, node_type: str) -> torch.Tensor:
        """
        Encode a single entity without the graph (used by /embed).
        Returns an L2-normalised embedding [EMBED_DIM].
        The embedding is consistent with forward_graph in the shared feature space
        but lacks cross-type context. ml:sync re-embeds with the full graph.
        """
        h = self.drop(F.relu(self.input_norm[node_type](self.input_proj[node_type](x))))
        return F.normalize(self.out_proj(h), dim=-1)


# ── Persistence ────────────────────────────────────────────────────────────────

def save_model(model: HetEncoder, path: str = MODEL_WEIGHTS_PATH) -> None:
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")


def load_model(path: str = MODEL_WEIGHTS_PATH) -> Optional[HetEncoder]:
    if not os.path.exists(path):
        return None
    model = HetEncoder().to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    except (RuntimeError, Exception):
        print("Warning: saved weights incompatible with current architecture — discarding.")
        os.remove(path)
        return None
    model.eval()
    return model
