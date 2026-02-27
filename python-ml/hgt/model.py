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

from hgt.config import (
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

        # Per-type output projection → embedding space
        self.out_proj = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(HIDDEN_DIM, EMBED_DIM),
                nn.LayerNorm(EMBED_DIM),
            ) for t in NODE_TYPES
        })

        # DistMult Relational Decoder Weights
        # Initialized with ones so that it natively starts as a pure dot-product
        # and slowly learns relationship-specific deviations.
        self.decoder_weights = nn.ParameterDict()
        for src, rel, dst in METADATA[1]:
            key = f"{src}__{rel}__{dst}"
            self.decoder_weights[key] = nn.Parameter(torch.ones(EMBED_DIM))

        # Adding discovery validation relations for validation phase
        discovery_rels = [
            ("event", "similarity", "event"),
            ("space", "similarity", "space"),
            ("tag", "similarity", "tag")
        ]
        for src, rel, dst in discovery_rels:
            key = f"{src}__{rel}__{dst}"
            self.decoder_weights[key] = nn.Parameter(torch.ones(EMBED_DIM))

    # ── Relational Decoder (DistMult) ──────────────────────────────────────────
    
    def score(self, src_type: str, rel_type: str, dst_type: str, a_emb: torch.Tensor, i_emb: torch.Tensor) -> torch.Tensor:
        """
        DistMult scoring for a batch of (anchor, item) pairs.
        a_emb: [B, D]
        i_emb: [B, D]
        """
        key = f"{src_type}__{rel_type}__{dst_type}"
        w = self.decoder_weights.get(key, None)
        if w is None:
            # Fallback to pure dot product if relation not found
            return torch.sum(a_emb * i_emb, dim=-1)
        return torch.sum(a_emb * w * i_emb, dim=-1)

    def score_batch(self, src_type: str, rel_type: str, dst_type: str, a_emb: torch.Tensor, i_emb_mat: torch.Tensor) -> torch.Tensor:
        """
        DistMult scoring for evaluating 1 anchor against N items.
        a_emb: [1, D]
        i_emb_mat: [N, D]
        """
        key = f"{src_type}__{rel_type}__{dst_type}"
        w = self.decoder_weights.get(key, None)
        if w is None:
            return torch.sum(a_emb * i_emb_mat, dim=-1)
        return torch.sum((a_emb * w) * i_emb_mat, dim=-1)

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
        Returns {node_type: embeddings [N, EMBED_DIM]}.
        """
        h = self._encode_inputs(x_dict)
        for conv in self.convs:
            # HGTConv accepts edge_index_dict as second argument.
            h_new = conv(h, edge_index_dict)
            h = {t: self.drop(h_new[t]) + h[t] for t in h}
        return {t: self.out_proj[t](v) for t, v in h.items()}

    # ── Encoder-only forward (/embed — new entities, no graph) ────────────────

    def forward_single(self, x: torch.Tensor, node_type: str) -> torch.Tensor:
        """
        Encode a single entity without the graph (used by /embed).
        Returns an embedding [EMBED_DIM].
        The embedding is consistent with forward_graph in the shared feature space
        but lacks cross-type context. ml:sync re-embeds with the full graph.
        """
        h = self.drop(F.relu(self.input_norm[node_type](self.input_proj[node_type](x))))
        return self.out_proj[node_type](h)


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
