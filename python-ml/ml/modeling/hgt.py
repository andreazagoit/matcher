"""
Heterogeneous Graph Transformer — lightweight (HGT-lite).

Architecture
────────────
Each entity type has its own input encoder that projects type-specific features
into a shared hidden space (HIDDEN_DIM). A single HGT-style cross-type attention
layer then refines each node's representation by attending over its direct
neighbors (used during training; transparent at inference).

                 UserEncoder(60→128)  ─┐
                EventEncoder(51→128)  ─┤─→  HGTAttentionLayer  →  OutputProj(128→256)  →  L2-norm
                SpaceEncoder(43→128)  ─┘

Inference
─────────
Only the type-specific encoder + output projection run — no graph needed.
The `/embed` endpoint stays stateless.

Training
────────
Anchors can be any entity type (user / event / space). For each positive
triplet (anchor, item), one hop of HGT attention is applied in both directions,
making message-passing aware of the specific (anchor_type, item_type) edge.
This creates a training signal that rewards proximity in the shared embedding
space for all heterogeneous pairs.

Loss: InfoNCE (in-batch negatives) + contrastive margin on explicit negatives.
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ml.config import (
    USER_DIM, EVENT_DIM, SPACE_DIM,
    EMBED_DIM, HIDDEN_DIM,
    DROPOUT,
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# bfloat16 autocast: M1 executes bfloat16 natively (~30% speedup, no gradient scaler needed)
_AUTOCAST_ENABLED = device.type in ("mps", "cuda")
_AUTOCAST_DTYPE   = torch.bfloat16 if _AUTOCAST_ENABLED else torch.float32

_DIM_MAP = {"user": USER_DIM, "event": EVENT_DIM, "space": SPACE_DIM}


# ─── Building blocks ───────────────────────────────────────────────────────────

def _make_encoder(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_DIM),
        nn.LayerNorm(HIDDEN_DIM),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.LayerNorm(HIDDEN_DIM),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
    )


class HGTAttentionLayer(nn.Module):
    """
    One hop of HGT-style cross-type attention.

    For each directed edge (src_type → dst_type) a distinct set of
    key/query/value projections is learned, making message-passing
    aware of both node and edge heterogeneity.

    Edge types supported:
        all pairwise combinations between {user, event, space}
    """

    EDGE_TYPES = [
        ("user", "user"),
        ("user", "event"),
        ("user", "space"),
        ("event", "user"),
        ("event", "event"),
        ("event", "space"),
        ("space", "user"),
        ("space", "event"),
        ("space", "space"),
    ]

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_heads: int = 4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        def _proj_dict():
            return nn.ModuleDict({
                f"{s}__{d}": nn.Linear(hidden_dim, hidden_dim, bias=False)
                for s, d in self.EDGE_TYPES
            })

        self.W_k = _proj_dict()   # key   (source-side)
        self.W_q = _proj_dict()   # query (target-side)
        self.W_v = _proj_dict()   # value (source-side)

        # Per-target-type output normalisation
        self.W_out = nn.ModuleDict({
            t: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for t in ["user", "event", "space"]
        })
        self.norms = nn.ModuleDict({
            t: nn.LayerNorm(hidden_dim)
            for t in ["user", "event", "space"]
        })

    def _edge_key(self, src_type: str, dst_type: str) -> str:
        return f"{src_type}__{dst_type}"

    def forward(
        self,
        src_h: torch.Tensor,   # (B, hidden_dim)  source node embeddings
        dst_h: torch.Tensor,   # (B, hidden_dim)  target node embeddings
        src_type: str,
        dst_type: str,
    ) -> torch.Tensor:
        """
        Returns updated dst_h after one hop of src → dst attention.
        Shapes: (B, hidden_dim) → (B, hidden_dim)
        """
        key_name = self._edge_key(src_type, dst_type)
        if key_name not in self.W_k:
            return dst_h   # unsupported edge type → identity

        B = src_h.size(0)
        H, D = self.num_heads, self.head_dim

        def reshape(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, H, D)

        k = reshape(self.W_k[key_name](src_h))  # (B, H, D)
        q = reshape(self.W_q[key_name](dst_h))  # (B, H, D)
        v = reshape(self.W_v[key_name](src_h))  # (B, H, D)

        # Per-sample attention: sigmoid(q·k / scale) — one weight per head per pair.
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / self.scale)  # (B, H, 1)
        msg  = (attn * v).view(B, self.hidden_dim)                             # (B, hidden_dim)

        out = self.W_out[dst_type](msg)
        return self.norms[dst_type](dst_h + out)   # residual


# ─── Main model ────────────────────────────────────────────────────────────────

class HetEncoder(nn.Module):
    """
    Heterogeneous encoder (HGT-lite).

    Each entity type is encoded through its own MLP encoder, then
    two stacked HGT attention layers refine representations via cross-type
    message passing (used during training; skipped at inference), then
    an output projection maps to the shared EMBED_DIM space.

                UserEncoder(60→256)  ─┐
               EventEncoder(51→256)  ─┤─→  HGTLayer1  →  HGTLayer2  →  OutputProj(256→256)  →  L2-norm
               SpaceEncoder(43→256)  ─┘
    """

    def __init__(self):
        super().__init__()

        # Type-specific input encoders
        self.encoders = nn.ModuleDict({
            "user":  _make_encoder(USER_DIM),
            "event": _make_encoder(EVENT_DIM),
            "space": _make_encoder(SPACE_DIM),
        })

        # Two stacked graph attention layers (used during training)
        self.hgt  = HGTAttentionLayer(HIDDEN_DIM)
        self.hgt2 = HGTAttentionLayer(HIDDEN_DIM)

        # Shared output projection
        self.output_proj = nn.Linear(HIDDEN_DIM, EMBED_DIM)
        self.output_norm = nn.LayerNorm(EMBED_DIM)

    def _encode_hidden(self, entity_type: str, x: torch.Tensor) -> torch.Tensor:
        """Project features → hidden space. x: (B, input_dim) → (B, HIDDEN_DIM)."""
        return self.encoders[entity_type](x)

    def _hidden_to_embed(self, h: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.output_proj(h))

    def _hgt_hop(
        self,
        layer: HGTAttentionLayer,
        h_anchor: torch.Tensor,
        h_item: torch.Tensor,
        anchor_types: list[str],
        item_types: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One full hop of bidirectional HGT attention for all (anchor_type, item_type) pairs.
        Returns updated (h_anchor, h_item) tensors.
        """
        h_anchor_out = h_anchor.clone()
        for atype in set(anchor_types):
            for itype in set(item_types):
                mask = torch.tensor(
                    [(a == atype and i == itype) for a, i in zip(anchor_types, item_types)],
                    device=h_anchor.device,
                )
                if not mask.any():
                    continue
                h_anchor_out[mask] = layer(
                    src_h=h_item[mask], dst_h=h_anchor[mask],
                    src_type=itype, dst_type=atype,
                )
                h_item[mask] = layer(
                    src_h=h_anchor[mask], dst_h=h_item[mask],
                    src_type=atype, dst_type=itype,
                )
        return h_anchor_out, h_item

    def forward(
        self,
        anchor_feats: torch.Tensor,     # (B, anchor_dim)
        anchor_types: list[str],        # length B
        item_feats: torch.Tensor,       # (B, item_dim)
        item_types: list[str],          # length B
        use_graph: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (anchor_emb, item_emb), each (B, EMBED_DIM), L2-normalised.

        When use_graph=True two stacked hops of HGT attention are applied:
          hop 1: raw encoder output → first refinement
          hop 2: first refinement → second refinement (2-hop neighbourhood)
        """
        # Anchors/items may have mixed types and padded dims; encode per type.
        h_anchor = torch.zeros(len(anchor_types), HIDDEN_DIM, device=anchor_feats.device)
        for atype in set(anchor_types):
            mask = torch.tensor([t == atype for t in anchor_types], device=anchor_feats.device)
            dim = _DIM_MAP[atype]
            h_anchor[mask] = self._encode_hidden(atype, anchor_feats[mask][:, :dim])

        h_item = torch.zeros(len(item_types), HIDDEN_DIM, device=anchor_feats.device)
        for itype in set(item_types):
            mask = torch.tensor([t == itype for t in item_types], device=anchor_feats.device)
            dim = _DIM_MAP[itype]
            h_item[mask] = self._encode_hidden(itype, item_feats[mask][:, :dim])

        if use_graph:
            h_anchor, h_item = self._hgt_hop(self.hgt,  h_anchor, h_item, anchor_types, item_types)
            h_anchor, h_item = self._hgt_hop(self.hgt2, h_anchor, h_item, anchor_types, item_types)

        anchor_emb = F.normalize(self._hidden_to_embed(h_anchor), dim=-1)
        item_emb   = F.normalize(self._hidden_to_embed(h_item),   dim=-1)
        return anchor_emb, item_emb

    def encode(self, entity_type: str, features: list[float]) -> torch.Tensor:
        """Encode a single entity. Stateless — no graph propagation."""
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            h = self._encode_hidden(entity_type, x)
            return F.normalize(self._hidden_to_embed(h), dim=-1).squeeze(0)


# ─── Loss ──────────────────────────────────────────────────────────────────────

def contrastive_loss(
    anchor_emb: torch.Tensor,
    item_emb: torch.Tensor,
    labels: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    margin: float = 0.3,
    temperature: float = 0.1,
    fn_mask: Optional[torch.Tensor] = None,
    _components: Optional[dict] = None,
    anchor_types: Optional[list[str]] = None,
    item_types: Optional[list[str]] = None,
    cross_type_temperature: float = 0.15,
    cross_type_loss_scale: float = 3.0,
) -> torch.Tensor:
    """
    Weighted combination of:
      - Margin loss on positive pairs (scaled by interaction weight)
      - Margin loss on explicit negatives (uniform weight)
      - Per-task InfoNCE: one independent InfoNCE loss per unique
        (anchor_type, item_type) pair found in the batch.

    Per-task InfoNCE is the key to cross-type alignment.  A single mixed
    cross-type sub-batch causes contradictory gradients: user_i gets pushed
    away from space_j even when they share tags, simply because space_j
    appears as a negative for a user→event pair in the same batch.
    Computing InfoNCE within each task (user→event, user→space, event→user …)
    gives each task a clean, unambiguous signal.

      same-type tasks  → temperature (sharp, fast convergence)
      cross-type tasks → cross_type_temperature (softer, avoids vanishing
                         signal when cosines are near 0 in early epochs)
                         × cross_type_loss_scale (compensate fewer pairs)

    weights:  per-sample float in [0.05, 1.0].
    fn_mask:  (P, P) bool tensor of known off-diagonal true positives to
              mask from InfoNCE denominator (false-negative rejection).
    anchor_types / item_types: string lists — when provided, InfoNCE is
              split per (anchor_type, item_type) task.
    _components: optional dict filled with {"pos", "neg", "info_same", "info_cross"}.
    """
    if weights is None:
        weights = torch.ones_like(labels)

    sim = (anchor_emb * item_emb).sum(dim=-1)   # cosine sim (L2-normalised)

    pos_mask = labels == 1
    neg_mask = labels == 0

    loss = torch.tensor(0.0, device=anchor_emb.device)
    pos_val = neg_val = info_same_val = info_cross_val = 0.0

    # Weighted margin: strong positives must be close; weak ones get lenient push
    if pos_mask.any():
        pos_t = (F.relu(1.0 - margin - sim[pos_mask]) * weights[pos_mask]).mean()
        loss  = loss + pos_t
        pos_val = pos_t.item()

    if neg_mask.any():
        neg_t = F.relu(sim[neg_mask] - margin).mean()
        loss  = loss + neg_t
        neg_val = neg_t.item()

    # ── Per-task InfoNCE ───────────────────────────────────────────────────────
    n_pos = int(pos_mask.sum().item())
    if n_pos >= 2:
        pos_anc = anchor_emb[pos_mask]
        pos_itm = item_emb[pos_mask]
        pos_w   = weights[pos_mask]

        def _infonce_for(sub_mask: torch.Tensor, temp: float) -> Optional[torch.Tensor]:
            n = int(sub_mask.sum().item())
            if n < 2:
                return None
            anc = pos_anc[sub_mask]
            itm = pos_itm[sub_mask]
            w   = pos_w[sub_mask]
            logits = torch.matmul(anc, itm.T) / temp   # (n, n)
            # Propagate fn_mask into this sub-view.
            if fn_mask is not None:
                sub_idx = sub_mask.nonzero(as_tuple=True)[0]
                if sub_idx.numel() >= 2:
                    sub_fn = fn_mask[sub_idx][:, sub_idx]
                    if sub_fn.shape == logits.shape:
                        logits = logits.masked_fill(sub_fn, -float("inf"))
            targets = torch.arange(n, device=anchor_emb.device)
            return (F.cross_entropy(logits, targets, reduction="none") * w).mean()

        if anchor_types is not None and item_types is not None:
            # Extract types for positive-subset indices.
            _pa = [anchor_types[i] for i in range(len(anchor_types)) if pos_mask[i]]
            _pi = [item_types[i]   for i in range(len(item_types))   if pos_mask[i]]

            # One InfoNCE per unique (anchor_type, item_type) task — preserves
            # clean per-task semantics and balances cross-type gradient weight.
            seen_tasks: set[tuple[str, str]] = set()
            for at, it in zip(_pa, _pi):
                if (at, it) in seen_tasks:
                    continue
                seen_tasks.add((at, it))

                task_mask = torch.tensor(
                    [a == at and i == it for a, i in zip(_pa, _pi)],
                    dtype=torch.bool, device=anchor_emb.device,
                )
                is_cross = at != it
                temp     = cross_type_temperature if is_cross else temperature
                t_task   = _infonce_for(task_mask, temp)
                if t_task is not None:
                    scale = cross_type_loss_scale if is_cross else 1.0
                    loss  = loss + scale * t_task
                    if is_cross:
                        info_cross_val += t_task.item()
                    else:
                        info_same_val  += t_task.item()
        else:
            # No type info → single InfoNCE over all positives (original behaviour).
            all_mask = torch.ones(n_pos, dtype=torch.bool, device=anchor_emb.device)
            t_all = _infonce_for(all_mask, temperature)
            if t_all is not None:
                loss          = loss + t_all
                info_same_val = t_all.item()

    if _components is not None:
        _components["pos"]        = pos_val
        _components["neg"]        = neg_val
        _components["info_same"]  = info_same_val
        _components["info_cross"] = info_cross_val

    return loss


# ─── Training dataset ──────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    Stores heterogeneous training triplets.
    Each triplet: (anchor_type, anchor_id, anchor_vec, item_type, item_id, item_vec, label, weight)

    Entity IDs are threaded through so the training loop can build an in-batch
    positive mask for InfoNCE false-negative rejection.
    """

    def __init__(self, triplets: list[tuple]):
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        anchor_type, anchor_id, anchor_vec, item_type, item_id, item_vec, label, weight = self.triplets[idx]
        return (
            anchor_type,
            anchor_id,
            torch.tensor(anchor_vec, dtype=torch.float32),
            item_type,
            item_id,
            torch.tensor(item_vec,   dtype=torch.float32),
            float(label),
            float(weight),
        )


def _collate(batch):
    # batch element layout (from TripletDataset.__getitem__):
    #   b[0] anchor_type  b[1] anchor_id  b[2] anchor_tensor
    #   b[3] item_type    b[4] item_id    b[5] item_tensor
    #   b[6] label        b[7] weight
    anchor_types = [b[0] for b in batch]
    anchor_ids   = [b[1] for b in batch]
    item_types   = [b[3] for b in batch]
    item_ids     = [b[4] for b in batch]
    labels  = torch.tensor([b[6] for b in batch], dtype=torch.float32)
    weights = torch.tensor([b[7] for b in batch], dtype=torch.float32)

    # Anchors/items may have different dims — pad each side independently.
    max_anchor_dim = max(b[2].size(0) for b in batch)
    anchors = torch.zeros(len(batch), max_anchor_dim)
    for i, b in enumerate(batch):
        anchors[i, : b[2].size(0)] = b[2]

    max_item_dim = max(b[5].size(0) for b in batch)
    items = torch.zeros(len(batch), max_item_dim)
    for i, b in enumerate(batch):
        items[i, : b[5].size(0)] = b[5]

    return anchors, anchor_types, anchor_ids, items, item_types, item_ids, labels, weights


# ─── Batch encoding ────────────────────────────────────────────────────────────

def encode_all(
    model: HetEncoder,
    features: dict[str, tuple[str, list[float]]],
    batch_size: int = 1024,
) -> dict[str, torch.Tensor]:
    """
    Batch encode all entities grouped by type (efficient, no padding overhead).

    Args:
        features: {entity_id: (entity_type, feature_vec)}

    Returns:
        {entity_id: L2-normalised embedding tensor (EMBED_DIM,)}
    """
    result: dict[str, torch.Tensor] = {}
    by_type: dict[str, list[tuple[str, list[float]]]] = {}
    for eid, (etype, fvec) in features.items():
        by_type.setdefault(etype, []).append((eid, fvec))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for etype, items in by_type.items():
                for i in range(0, len(items), batch_size):
                    batch = items[i : i + batch_size]
                    ids   = [x[0] for x in batch]
                    vecs  = torch.tensor([x[1] for x in batch], dtype=torch.float32).to(device)
                    h     = model._encode_hidden(etype, vecs)
                    embs  = F.normalize(model._hidden_to_embed(h), dim=-1)
                    for j, eid in enumerate(ids):
                        result[eid] = embs[j]
    finally:
        if was_training:
            model.train()
    return result


# ─── Similarity search ─────────────────────────────────────────────────────────

def find_similar(
    model: HetEncoder,
    query_type: str,
    query_features: list[float],
    candidates: dict[str, tuple[str, list[float]]],
    target_type: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """
    Returns top-N most similar entities to the query.

    Args:
        model:          trained HetEncoder
        query_type:     entity type of the query ("user" | "event" | "space")
        query_features: feature vector of the source entity
        candidates:     {entity_id: (entity_type, features)}
        target_type:    if set, filter to this type
        limit:          max results

    Returns:
        list of {"id": ..., "type": ..., "score": ...} sorted by score desc
    """
    model.eval()
    query_emb = model.encode(query_type, query_features)

    filtered = {
        eid: (etype, fvec)
        for eid, (etype, fvec) in candidates.items()
        if target_type is None or etype == target_type
    }

    if not filtered:
        return []

    ids    = list(filtered.keys())
    types  = [filtered[eid][0] for eid in ids]
    scores = []

    # Encode in groups per type to avoid padding overhead
    for etype in set(types):
        group_ids  = [eid for eid in ids if filtered[eid][0] == etype]
        group_vecs = torch.tensor(
            [filtered[eid][1] for eid in group_ids], dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            h = model._encode_hidden(etype, group_vecs)
            embs = F.normalize(model._hidden_to_embed(h), dim=-1)
            sims = F.cosine_similarity(query_emb.unsqueeze(0), embs).cpu().tolist()
        scores.extend(zip(group_ids, [etype] * len(group_ids), sims))

    ranked = sorted(scores, key=lambda x: x[2], reverse=True)
    return [
        {"id": eid, "type": etype, "score": round(float(s), 4)}
        for eid, etype, s in ranked[:limit]
    ]
