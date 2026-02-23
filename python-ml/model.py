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
import copy
import math
import os
import random
import time
from collections import defaultdict
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from config import (
    USER_DIM, EVENT_DIM, SPACE_DIM,
    EMBED_DIM, HIDDEN_DIM,
    LEARNING_RATE, EPOCHS, BATCH_SIZE, DROPOUT, MODEL_WEIGHTS_PATH,
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
    projected to the shared EMBED_DIM space.

    Optionally, a HGTAttentionLayer refines the representation via
    one hop of cross-type attention (used during training; skipped at inference).
    """

    def __init__(self):
        super().__init__()

        # Type-specific input encoders
        self.encoders = nn.ModuleDict({
            "user":  _make_encoder(USER_DIM),
            "event": _make_encoder(EVENT_DIM),
            "space": _make_encoder(SPACE_DIM),
        })

        # Graph attention layer (used during training)
        self.hgt = HGTAttentionLayer(HIDDEN_DIM)

        # Shared output projection
        self.output_proj = nn.Linear(HIDDEN_DIM, EMBED_DIM)
        self.output_norm = nn.LayerNorm(EMBED_DIM)

    def _encode_hidden(self, entity_type: str, x: torch.Tensor) -> torch.Tensor:
        """Project features → hidden space. x: (B, input_dim) → (B, HIDDEN_DIM)."""
        return self.encoders[entity_type](x)

    def _hidden_to_embed(self, h: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.output_proj(h))

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

        When use_graph=True one hop of HGT attention is applied in both directions:
          - anchor attends over item
          - item attends back over anchor
        """
        # Anchors/items may have mixed types and padded dims; we encode per type.
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
            # Apply one-hop attention pair-wise by (anchor_type, item_type).
            h_anchor_refined = h_anchor.clone()
            for atype in set(anchor_types):
                for itype in set(item_types):
                    mask = torch.tensor(
                        [(a == atype and i == itype) for a, i in zip(anchor_types, item_types)],
                        device=anchor_feats.device,
                    )
                    if not mask.any():
                        continue
                    h_anchor_refined[mask] = self.hgt(
                        src_h=h_item[mask], dst_h=h_anchor[mask],
                        src_type=itype, dst_type=atype,
                    )
                    h_item[mask] = self.hgt(
                        src_h=h_anchor[mask], dst_h=h_item[mask],
                        src_type=atype, dst_type=itype,
                    )
            h_anchor = h_anchor_refined

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
    temperature: float = 0.07,
    fn_mask: Optional[torch.Tensor] = None,
    _components: Optional[dict] = None,
) -> torch.Tensor:
    """
    Weighted combination of:
      - Margin loss on positive pairs (scaled by interaction weight)
      - Margin loss on explicit negatives (uniform weight)
      - InfoNCE on positive pairs with in-batch negatives (scaled by weight)

    weights:  per-sample float in [0.05, 1.0].
              Positive weights reflect recency × interaction type.
              Negative weights are 1.0 — push all negatives equally.
    fn_mask:  optional (P, P) bool tensor of known off-diagonal true positives
              in the positive subset of the batch.  These cells are masked out of
              the InfoNCE denominator to avoid penalising bidirectional or
              co-positive pairs that shouldn't count as negatives.
    _components: optional dict; if provided, filled with scalar breakdown
                 {"pos": float, "neg": float, "info": float} for logging.
    """
    if weights is None:
        weights = torch.ones_like(labels)

    sim = (anchor_emb * item_emb).sum(dim=-1)   # cosine sim (embeddings are L2-normalised)

    pos_mask = labels == 1
    neg_mask = labels == 0

    loss = torch.tensor(0.0, device=anchor_emb.device)
    pos_val = neg_val = info_val = 0.0

    # Weighted margin: strong positives must be close; weak ones get lenient push
    if pos_mask.any():
        pos_t = (F.relu(1.0 - margin - sim[pos_mask]) * weights[pos_mask]).mean()
        loss  = loss + pos_t
        pos_val = pos_t.item()

    if neg_mask.any():
        neg_t = F.relu(sim[neg_mask] - margin).mean()
        loss  = loss + neg_t
        neg_val = neg_t.item()

    # Weighted InfoNCE: positive pairs weighted by interaction strength.
    # fn_mask suppresses known off-diagonal true positives (bidirectional edges,
    # co-engaged pairs) so they don't contribute false-negative gradient.
    if pos_mask.sum() >= 2:
        pos_anc = anchor_emb[pos_mask]
        pos_itm = item_emb[pos_mask]
        logits  = torch.matmul(pos_anc, pos_itm.T) / temperature  # (P, P)
        if fn_mask is not None and fn_mask.shape == logits.shape:
            logits = logits.masked_fill(fn_mask, -float("inf"))
        targets = torch.arange(logits.size(0), device=logits.device)
        infonce = F.cross_entropy(logits, targets, reduction="none")
        info_t  = (infonce * weights[pos_mask]).mean()
        loss    = loss + info_t
        info_val = info_t.item()

    if _components is not None:
        _components["pos"]  = pos_val
        _components["neg"]  = neg_val
        _components["info"] = info_val

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
    batch_size: int = 512,
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


# ─── Hard negative mining ───────────────────────────────────────────────────────

def mine_hard_negatives(
    model: HetEncoder,
    anchor_features_by_type: dict[str, dict[str, list[float]]],
    item_features: dict[str, tuple[str, list[float]]],
    positive_set: set[tuple[str, str, str, str]],
    n_per_anchor: int = 3,
    max_anchors_per_type: int = 2_000,
) -> list[tuple]:
    """
    Mine hard negatives for every anchor type (user / event / space).

    For each (anchor_type, target_type) pair found in positive_set, encodes all
    target items and sampled anchors, then selects the top-N most similar
    non-interacted items as hard negatives (label=0, weight=1.0).

    Negatives are type-constrained: for a user→event positive, hard negatives are
    events only — matching the type-constrained training regime.

    Args:
        anchor_features_by_type: {entity_type: {entity_id: feature_vec}}
        item_features:           full corpus {entity_id: (entity_type, feature_vec)}
        positive_set:            4-tuple keys (anchor_type, anchor_id, item_type, item_id)
        n_per_anchor:            hard negatives per anchor per target type
        max_anchors_per_type:    cap per anchor type to keep mining fast

    Returns list of 8-tuples (anchor_type, anchor_id, anchor_vec,
                               item_type, item_id, item_vec, 0, 1.0).
    """
    # Encode all corpus items once, then split by type for fast per-task lookup.
    all_item_embs = encode_all(model, item_features)
    emb_mat_by_type: dict[str, tuple[list[str], torch.Tensor]] = {}
    items_by_type: dict[str, list[str]] = {}
    for iid, (itype, _) in item_features.items():
        items_by_type.setdefault(itype, []).append(iid)
    for itype, iids in items_by_type.items():
        mat = torch.stack([all_item_embs[iid] for iid in iids])
        emb_mat_by_type[itype] = (iids, mat)

    # Determine which (anchor_type → target_type) pairs exist in positive_set.
    anchor_to_targets: dict[str, set[str]] = defaultdict(set)
    for atype, _aid, itype, _iid in positive_set:
        anchor_to_targets[atype].add(itype)

    # Build per-anchor positive lookup once (avoid re-iterating positive_set per anchor).
    anchor_positives: dict[tuple[str, str], set[str]] = defaultdict(set)
    for atype, aid, _itype, iid in positive_set:
        anchor_positives[(atype, aid)].add(iid)

    hard_neg_triplets: list[tuple] = []

    for anchor_type, feats in anchor_features_by_type.items():
        target_types = anchor_to_targets.get(anchor_type)
        if not target_types:
            continue

        aid_list = list(feats.keys())
        if len(aid_list) > max_anchors_per_type:
            aid_list = random.sample(aid_list, max_anchors_per_type)

        # Encode this anchor type in batch.
        anchor_feats_typed = {aid: (anchor_type, feats[aid]) for aid in aid_list}
        all_anchor_embs = encode_all(model, anchor_feats_typed)

        for target_type in target_types:
            if target_type not in emb_mat_by_type:
                continue
            target_ids, item_emb_mat = emb_mat_by_type[target_type]
            item_id_to_idx = {iid: i for i, iid in enumerate(target_ids)}

            for aid in aid_list:
                a_emb = all_anchor_embs[aid].unsqueeze(0)
                sims  = F.cosine_similarity(a_emb, item_emb_mat).cpu().clone()

                # Mask out known positives and the anchor itself.
                for exc_id in anchor_positives.get((anchor_type, aid), set()) | {aid}:
                    idx = item_id_to_idx.get(exc_id)
                    if idx is not None:
                        sims[idx] = -float("inf")

                valid_k = min(n_per_anchor, int((sims > -float("inf")).sum().item()))
                if valid_k == 0:
                    continue

                top_idxs = torch.topk(sims, k=valid_k).indices.tolist()
                a_feat   = feats[aid]
                for idx in top_idxs:
                    iid         = target_ids[idx]
                    _, ivec     = item_features[iid]
                    hard_neg_triplets.append(
                        (anchor_type, aid, a_feat, target_type, iid, ivec, 0, 1.0)
                    )

    return hard_neg_triplets


# ─── Validation metrics ─────────────────────────────────────────────────────────

def evaluate_recall_ndcg(
    model: HetEncoder,
    val_data: dict,
    k: int = 10,
    max_users: int = 500,
    allowed_item_types: Optional[set[str]] = None,
    fixed_eval_anchors: Optional[list] = None,
) -> dict:
    """
    Compute Recall@K and NDCG@K on held-out validation pairs.

    Supports both legacy user-only validation and typed multi-anchor validation.

    Args:
        val_data:  dict from build_training_data — {anchor_features, item_features, val_pairs}
        k:         ranking cut-off (default 10)
        max_users: sample this many val users for speed (default 500)
        allowed_item_types: optional candidate filter (e.g. {"event", "space"})

    Returns:
        {
          "recall@k": float, "ndcg@k": float,
          "macro_recall@k": float, "macro_ndcg@k": float,
          "micro_recall@k": float, "micro_ndcg@k": float,
          "per_task": dict, "k": int, "n_anchors": int
        }
    """
    item_features = val_data["item_features"]
    raw_pairs = val_data["val_pairs"]

    # Backward compatibility: legacy schema {user_features, val_pairs=(uid,iid)}.
    if "anchor_features" in val_data:
        anchor_features = val_data["anchor_features"]
        val_pairs = raw_pairs
        seen_train_by_anchor = {
            key: set(items)
            for key, items in (val_data.get("seen_train_by_anchor") or {}).items()
        }
    else:
        anchor_features = {("user", uid): feats for uid, feats in val_data["user_features"].items()}
        val_pairs = [("user", uid, "unknown", iid) for uid, iid in raw_pairs]
        seen_train_by_anchor = {
            ("user", uid): set(items)
            for uid, items in (val_data.get("seen_train_by_user") or {}).items()
        }

    # Group held-out pairs by (anchor_type, anchor_id, target_type)
    anchor_holdouts_by_task: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for atype, aid, itype, iid in val_pairs:
        target_type = itype
        if target_type == "unknown" and iid in item_features:
            target_type = item_features[iid][0]
        anchor_holdouts_by_task[(atype, aid)][target_type].add(iid)

    if fixed_eval_anchors is not None:
        # Use the caller-supplied fixed subset for a consistent metric across epochs.
        eval_anchors = [
            a for a in fixed_eval_anchors
            if a in anchor_holdouts_by_task and a in anchor_features
        ]
    else:
        eval_anchors = [akey for akey in anchor_holdouts_by_task if akey in anchor_features]
        if len(eval_anchors) > max_users:
            eval_anchors = random.sample(eval_anchors, max_users)

    if not eval_anchors:
        return {"recall@k": 0.0, "ndcg@k": 0.0, "k": k, "n_users": 0}

    if allowed_item_types:
        item_features = {
            iid: (itype, ivec)
            for iid, (itype, ivec) in item_features.items()
            if itype in allowed_item_types
        }
    if not item_features:
        return {"recall@k": 0.0, "ndcg@k": 0.0, "k": k, "n_users": 0}

    # Encode candidate corpus items in batch
    all_item_embs = encode_all(model, item_features)
    item_ids = list(item_features.keys())
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    item_emb_mat = torch.stack([all_item_embs[iid] for iid in item_ids])  # (n_items, EMBED_DIM)

    recalls: list[float] = []  # aggregate per-anchor across all targets
    ndcgs:   list[float] = []
    per_task_recalls: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_task_ndcgs: dict[tuple[str, str], list[float]] = defaultdict(list)

    for anchor_key in eval_anchors:
        atype, aid = anchor_key
        holdouts_by_type = {
            t: {iid for iid in ids if iid in item_id_to_idx}
            for t, ids in anchor_holdouts_by_task[anchor_key].items()
        }
        all_holdout_ids = set().union(*holdouts_by_type.values()) if holdouts_by_type else set()
        if not all_holdout_ids:
            continue
        a_emb = model.encode(atype, anchor_features[anchor_key]).unsqueeze(0)  # (1, EMBED_DIM)
        sims = F.cosine_similarity(a_emb, item_emb_mat)                         # (n_items,)

        # Exclude trivial candidates: self + items already seen in train.
        for seen_id in seen_train_by_anchor.get(anchor_key, set()):
            idx = item_id_to_idx.get(seen_id)
            if idx is not None:
                sims[idx] = -float("inf")
        if atype in ("user", "event", "space"):
            self_idx = item_id_to_idx.get(aid)
            if self_idx is not None:
                sims[self_idx] = -float("inf")

        top_k_idxs = torch.topk(sims, k=min(k, len(item_ids))).indices.tolist()
        top_k_ids  = {item_ids[i] for i in top_k_idxs}

        # Overall anchor metric across all target types.
        hits = len(top_k_ids & all_holdout_ids)
        recall = hits / len(all_holdout_ids)
        recalls.append(recall)
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, idx in enumerate(top_k_idxs)
            if item_ids[idx] in all_holdout_ids
        )
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(all_holdout_ids), k)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcgs.append(ndcg)

        # Per-task metrics by (anchor_type -> target_type).
        for target_type, holdout_ids in holdouts_by_type.items():
            if not holdout_ids:
                continue
            task_hits = len(top_k_ids & holdout_ids)
            task_recall = task_hits / len(holdout_ids)
            task_dcg = sum(
                1.0 / math.log2(rank + 2)
                for rank, idx in enumerate(top_k_idxs)
                if item_ids[idx] in holdout_ids
            )
            task_ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(holdout_ids), k)))
            task_ndcg = task_dcg / task_ideal_dcg if task_ideal_dcg > 0 else 0.0
            key = (atype, target_type)
            per_task_recalls[key].append(task_recall)
            per_task_ndcgs[key].append(task_ndcg)

    recall_at_k = (sum(recalls) / len(recalls)) if recalls else 0.0
    ndcg_at_k = (sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0

    per_task = {
        f"{src}->{dst}": {
            "recall@k": sum(vals_r) / len(vals_r),
            "ndcg@k": sum(per_task_ndcgs[(src, dst)]) / len(per_task_ndcgs[(src, dst)]),
            "n_users": len(vals_r),
        }
        for (src, dst), vals_r in per_task_recalls.items()
    }

    macro_recall = (
        sum(v["recall@k"] for v in per_task.values()) / len(per_task)
        if per_task else recall_at_k
    )
    macro_ndcg = (
        sum(v["ndcg@k"] for v in per_task.values()) / len(per_task)
        if per_task else ndcg_at_k
    )
    total_n = sum(v["n_users"] for v in per_task.values())
    micro_recall = (
        sum(v["recall@k"] * v["n_users"] for v in per_task.values()) / total_n
        if total_n > 0 else recall_at_k
    )
    micro_ndcg = (
        sum(v["ndcg@k"] * v["n_users"] for v in per_task.values()) / total_n
        if total_n > 0 else ndcg_at_k
    )

    return {
        "recall@k": recall_at_k,
        "ndcg@k": ndcg_at_k,
        "macro_recall@k": macro_recall,
        "macro_ndcg@k": macro_ndcg,
        "micro_recall@k": micro_recall,
        "micro_ndcg@k": micro_ndcg,
        "per_task": per_task,
        "k": k,
        "n_anchors": len(recalls),
        "n_users": len(recalls),
    }


# ─── Training ──────────────────────────────────────────────────────────────────

def train(
    triplets: list[tuple],
    val_data: Optional[dict] = None,
    positive_set: Optional[set] = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = 15,
    eval_every: int = 5,
    hard_neg_fn: Optional[Callable] = None,
    hard_neg_every: int = 0,
    checkpoint_path: Optional[str] = MODEL_WEIGHTS_PATH,
    checkpoint_every: int = 10,
) -> HetEncoder:
    """
    Train HetEncoder from 8-tuple
    (anchor_type, anchor_id, anchor_vec, item_type, item_id, item_vec, label, weight) triplets.

    Args:
        triplets:          training data from build_training_data()["train_triplets"]
        val_data:          optional — enables Recall@K / NDCG@K evaluation
        positive_set:      set of (anchor_type, anchor_id, item_type, item_id) 4-tuples;
                           used to mask known off-diagonal positives in InfoNCE
        epochs:            max training epochs
        batch_size:        mini-batch size
        patience:          early stopping — stop if val Recall@K doesn't improve for
                           this many epochs (0 = disabled)
        eval_every:        evaluate validation metrics every N epochs
        hard_neg_fn:       callable(model) → list of hard neg 8-tuples
        hard_neg_every:    0 = disabled
        checkpoint_path:   path to save best checkpoint (None = no auto-save)
        checkpoint_every:  also save a periodic checkpoint every N epochs as crash recovery
    """
    if not triplets:
        print("No training data. Returning untrained model.")
        return HetEncoder().to(device)

    model = HetEncoder().to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"  ↻  Fine-tuning from checkpoint: {checkpoint_path}", flush=True)
        except RuntimeError:
            print(
                "  !  Checkpoint incompatible with current architecture."
                " Training from scratch.",
                flush=True,
            )
    else:
        print("  •  No checkpoint found. Training from scratch.", flush=True)

    if device.type == "cuda":
        # torch.compile MPS inductor has a dynamic-shape bug; safe only on CUDA
        model = torch.compile(model, mode="default")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    base_triplets    = triplets[:]
    current_triplets = base_triplets[:]
    dataset = TripletDataset(current_triplets)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)

    best_recall   = -1.0
    best_weights: Optional[dict] = None
    no_improve_ep = 0

    n_batches   = len(loader)
    epoch_times: list[float] = []

    # Pre-sample a fixed eval subset once so early-stopping uses a consistent
    # metric across epochs (avoids stopping on sampling noise).
    _fixed_eval_anchors: Optional[list] = None
    if val_data is not None and eval_every > 0:
        _all_val_anc = list((val_data.get("anchor_features") or {}).keys())
        if _all_val_anc:
            _fixed_eval_anchors = random.sample(_all_val_anc, min(300, len(_all_val_anc)))

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── Hard negative refresh ──────────────────────────────────────────
        if hard_neg_fn is not None and hard_neg_every > 0 and epoch > 0 and epoch % hard_neg_every == 0:
            print(f"\n  ↺  Mining hard negatives (epoch {epoch})...", flush=True)
            model.eval()
            hard_negs = hard_neg_fn(model)
            model.train()
            current_triplets = base_triplets + hard_negs
            dataset = TripletDataset(current_triplets)
            loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
            n_batches = len(loader)
            print(
                f"  ↺  +{len(hard_negs):,} hard negs  "
                f"→ {len(current_triplets):,} total triplets",
                flush=True,
            )

        # ── Training step ──────────────────────────────────────────────────
        model.train()
        total_loss = total_pos = total_neg = total_info = 0.0

        for batch_idx, (batch_anchors, anchor_types, anchor_ids, batch_items, item_types, item_ids, batch_labels, batch_weights) in enumerate(loader):
            batch_anchors  = batch_anchors.to(device)
            batch_items    = batch_items.to(device)
            batch_labels   = batch_labels.to(device)
            batch_weights  = batch_weights.to(device)

            # ── InfoNCE false-negative mask ────────────────────────────────
            # For each pair of positives in the batch, flag off-diagonal cells
            # where (anchor_i, item_j) is a known true positive so InfoNCE
            # doesn't penalise bidirectional or co-positive pairs.
            fn_mask: Optional[torch.Tensor] = None
            if positive_set is not None:
                _pos_flags = (batch_labels == 1).cpu().tolist()
                _pos_idx   = [i for i, m in enumerate(_pos_flags) if m]
                P = len(_pos_idx)
                if P >= 2:
                    _fm = torch.zeros(P, P, dtype=torch.bool)
                    for _pi, _i in enumerate(_pos_idx):
                        for _pj, _j in enumerate(_pos_idx):
                            if _pi != _pj and (
                                anchor_types[_i], anchor_ids[_i],
                                item_types[_j], item_ids[_j],
                            ) in positive_set:
                                _fm[_pi, _pj] = True
                    if _fm.any():
                        fn_mask = _fm.to(device)

            components: dict = {}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=_AUTOCAST_DTYPE, enabled=_AUTOCAST_ENABLED):
                # DropGraph: 15% of steps train without graph attention to keep
                # the output projection aligned with the graph-free inference path.
                _use_graph = random.random() > 0.15
                anchor_emb, item_emb = model(
                    batch_anchors,
                    anchor_types,
                    batch_items,
                    item_types,
                    use_graph=_use_graph,
                )
                loss = contrastive_loss(
                    anchor_emb, item_emb, batch_labels, batch_weights,
                    fn_mask=fn_mask,
                    _components=components,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_pos  += components.get("pos",  0.0)
            total_neg  += components.get("neg",  0.0)
            total_info += components.get("info", 0.0)

            # In-epoch progress bar (updates in-place, no newline)
            if (batch_idx + 1) % max(1, n_batches // 20) == 0 or batch_idx == n_batches - 1:
                done = batch_idx + 1
                bar  = "█" * round(done / n_batches * 16) + "░" * (16 - round(done / n_batches * 16))
                avg  = total_loss / done
                print(
                    f"\r  ep {epoch+1:>4}/{epochs}  [{bar}] {done}/{n_batches}"
                    f"  loss: {avg:.4f}",
                    end="",
                    flush=True,
                )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = total_loss / n_batches

        # ETA estimate
        if len(epoch_times) >= 3:
            remaining = epochs - (epoch + 1)
            eta_s     = remaining * (sum(epoch_times[-5:]) / len(epoch_times[-5:]))
            eta_str   = f"  ETA {eta_s/60:.0f}m{eta_s%60:.0f}s"
        else:
            eta_str = ""

        # Loss breakdown suffix
        breakdown = (
            f"  [pos {total_pos/n_batches:.4f}"
            f"  neg {total_neg/n_batches:.4f}"
            f"  nce {total_info/n_batches:.4f}]"
        )

        # ── Validation ─────────────────────────────────────────────────────
        if val_data is not None and eval_every > 0 and (epoch + 1) % eval_every == 0:
            metrics = evaluate_recall_ndcg(
                model, val_data, k=10, fixed_eval_anchors=_fixed_eval_anchors,
            )
            # Micro average (weighted by n) as the early-stopping signal.
            # Macro would be dominated by trivially-easy same-type tasks
            # (event→event, space→space) whose recall is near 1.0 from epoch 1
            # due to raw feature similarity — masking lack of progress on the
            # product-critical user→event / user→space tasks.
            r_at_k = metrics.get("micro_recall@k", metrics.get("macro_recall@k", metrics["recall@k"]))
            n_at_k = metrics.get("micro_ndcg@k",   metrics.get("macro_ndcg@k",   metrics["ndcg@k"]))

            is_best = r_at_k > best_recall
            star    = "  ★ best" if is_best else ""

            print(
                f"\r  ep {epoch+1:>4}/{epochs}"
                f"  loss: {avg_loss:.4f}{breakdown}"
                f"  {epoch_time:.1f}s{eta_str}"
                f"  Recall@10: {r_at_k:.4f}"
                f"  NDCG@10: {n_at_k:.4f}"
                f"  (n={metrics.get('n_anchors', metrics['n_users'])}){star}",
                flush=True,
            )
            per_task = metrics.get("per_task") or {}
            if per_task:
                parts = []
                for task_name in sorted(per_task.keys()):
                    task = per_task[task_name]
                    parts.append(
                        f"{task_name} R@10:{task['recall@k']:.4f} "
                        f"N@10:{task['ndcg@k']:.4f} (n={task['n_users']})"
                    )
                print(f"       {' | '.join(parts)}", flush=True)

            if is_best:
                best_recall   = r_at_k
                best_weights  = copy.deepcopy(model.state_dict())
                no_improve_ep = 0
                if checkpoint_path:
                    torch.save(best_weights, checkpoint_path)
            else:
                no_improve_ep += eval_every
                if patience > 0 and no_improve_ep >= patience:
                    print(
                        f"\n  ✗  Early stopping at epoch {epoch+1}"
                        f"  (best Recall@10={best_recall:.4f},"
                        f" no improvement for {patience} epochs)",
                        flush=True,
                    )
                    break

            model.train()
        else:
            # Log every epoch when val is active but eval is skipped this epoch;
            # log every 10 epochs when no val data at all.
            should_log = val_data is not None or (epoch + 1) % 10 == 0
            if should_log:
                print(
                    f"\r  ep {epoch+1:>4}/{epochs}"
                    f"  loss: {avg_loss:.4f}{breakdown}"
                    f"  {epoch_time:.1f}s{eta_str}",
                    flush=True,
                )

        # Periodic crash-recovery checkpoint (independent of val improvement)
        if (
            checkpoint_path
            and checkpoint_every > 0
            and (epoch + 1) % checkpoint_every == 0
        ):
            _recovery_path = checkpoint_path + ".latest"
            torch.save(model.state_dict(), _recovery_path)

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"\n  ✓  Restored best checkpoint  (Recall@10={best_recall:.4f})", flush=True)

    return model


# ─── Persistence ───────────────────────────────────────────────────────────────

def save_model(model: HetEncoder, path: str = MODEL_WEIGHTS_PATH) -> None:
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path: str = MODEL_WEIGHTS_PATH) -> Optional[HetEncoder]:
    if not os.path.exists(path):
        return None
    model = HetEncoder().to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError:
        # Architecture changed since last save — discard old weights
        print("Warning: saved weights are incompatible with current architecture. Discarding.")
        os.remove(path)
        return None
    model.eval()
    return model


# ─── Similarity search ─────────────────────────────────────────────────────────

def find_similar(
    model: HetEncoder,
    query_type: str,
    query_features: "list[float]",
    candidates: "dict[str, tuple[str, list[float]]]",
    target_type: Optional[str] = None,
    limit: int = 10,
) -> "list[dict]":
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
