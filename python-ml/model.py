"""
Heterogeneous Graph Transformer — lightweight (HGT-lite).

Architecture
────────────
Each entity type has its own input encoder that projects type-specific features
into a shared hidden space (HIDDEN_DIM). A single HGT-style cross-type attention
layer then refines each node's representation by attending over its direct
neighbors (used during training; transparent at inference).

                 UserEncoder(60→128)  ─┐
                EventEncoder(43→128)  ─┤─→  HGTAttentionLayer  →  OutputProj(128→64)  →  L2-norm
                SpaceEncoder(42→128)  ─┘

Inference
─────────
Only the type-specific encoder + output projection run — no graph needed.
The `/embed` endpoint stays stateless.

Training
────────
For each positive triplet (user, item), one hop of HGT attention is applied:
the user embedding is updated by attending over the item embedding (simulating
the edge user→item). This creates a training signal that rewards proximity
in the shared embedding space.

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
        user  → event
        user  → space
        user  → user   (mutual match / conversation)
    """

    EDGE_TYPES = [
        ("user", "event"),
        ("user", "space"),
        ("user", "user"),
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

        # Scaled dot-product attention per head (pair-wise within batch)
        # attn[i,j] = softmax over j of (q_i · k_j / scale)
        q_exp = q.unsqueeze(2)   # (B, H, 1, D)
        k_exp = k.unsqueeze(1)   # (B, H, D)  → broadcast (1, B, H, D) not needed; just per-sample
        # For a single-hop local update we compute per-sample (not across-batch),
        # using the single src→dst pair: attention weight = sigmoid(q·k / scale)
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
        anchor_feats: torch.Tensor,     # (B, USER_DIM)     — always user
        item_feats: torch.Tensor,       # (B, item_dim)
        item_types: list[str],          # length B
        use_graph: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (anchor_emb, item_emb), each (B, EMBED_DIM), L2-normalised.

        When use_graph=True one hop of HGT attention is applied:
          - anchor (user) attends over item
          - item attends back over anchor
        """
        h_anchor = self._encode_hidden("user", anchor_feats)

        # Items may have different types and dims.
        # _collate pads all items to the widest dim in the batch, so we must
        # slice each group back to its true input dimension before encoding.
        unique_types = list(set(item_types))
        if len(unique_types) == 1:
            dim    = _DIM_MAP[unique_types[0]]
            h_item = self._encode_hidden(unique_types[0], item_feats[:, :dim])
        else:
            h_item = torch.zeros(len(item_types), HIDDEN_DIM, device=anchor_feats.device)
            for etype in unique_types:
                mask = torch.tensor([t == etype for t in item_types], device=anchor_feats.device)
                dim  = _DIM_MAP[etype]
                h_item[mask] = self._encode_hidden(etype, item_feats[mask][:, :dim])

        if use_graph:
            # For mixed item types we apply attention per-type group
            h_anchor_refined = h_anchor.clone()
            for etype in set(item_types):
                mask = torch.tensor([t == etype for t in item_types], device=anchor_feats.device)
                h_anchor_refined[mask] = self.hgt(
                    src_h=h_item[mask], dst_h=h_anchor[mask],
                    src_type=etype, dst_type="user",
                )
                h_item[mask] = self.hgt(
                    src_h=h_anchor[mask], dst_h=h_item[mask],
                    src_type="user", dst_type=etype,
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
    _components: Optional[dict] = None,
) -> torch.Tensor:
    """
    Weighted combination of:
      - Margin loss on positive pairs (scaled by interaction weight)
      - Margin loss on explicit negatives (uniform weight)
      - InfoNCE on positive pairs with in-batch negatives (scaled by weight)

    weights:     per-sample float in [0.05, 1.0].
                 Positive weights reflect recency × interaction type.
                 Negative weights are 1.0 — push all negatives equally.
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

    # Weighted InfoNCE: positive pairs weighted by interaction strength
    if pos_mask.sum() >= 2:
        pos_anc = anchor_emb[pos_mask]
        pos_itm = item_emb[pos_mask]
        logits  = torch.matmul(pos_anc, pos_itm.T) / temperature  # (P, P)
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
    Each triplet: (anchor_vec, item_vec, item_type, label, weight)
    """

    def __init__(self, triplets: list[tuple]):
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        anchor_vec, item_vec, item_type, label, weight = self.triplets[idx]
        return (
            torch.tensor(anchor_vec, dtype=torch.float32),
            torch.tensor(item_vec,   dtype=torch.float32),
            item_type,
            float(label),
            float(weight),
        )


def _collate(batch):
    anchors   = torch.stack([b[0] for b in batch])
    item_type = [b[2] for b in batch]
    labels    = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    weights   = torch.tensor([b[4] for b in batch], dtype=torch.float32)

    # Items may have different dims — pad to max dim in batch
    max_dim = max(b[1].size(0) for b in batch)
    items = torch.zeros(len(batch), max_dim)
    for i, b in enumerate(batch):
        items[i, : b[1].size(0)] = b[1]

    return anchors, items, item_type, labels, weights


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

    model.eval()
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
    return result


# ─── Hard negative mining ───────────────────────────────────────────────────────

def mine_hard_negatives(
    model: HetEncoder,
    user_features: dict[str, list[float]],
    item_features: dict[str, tuple[str, list[float]]],
    positive_set: set[tuple[str, str]],
    n_per_user: int = 3,
    max_users: int = 2_000,
) -> list[tuple]:
    """
    Find hard negatives: items with high similarity to a user but not in positive_set.

    Encodes all items and sampled users in batch, then for each user selects
    the top-N most similar non-interacted items as hard negatives (label=0, weight=1.0).

    Returns list of 5-tuples (anchor_vec, item_vec, item_type, 0, 1.0).
    """
    # Encode all items once
    all_item_embs  = encode_all(model, item_features)
    item_ids       = list(item_features.keys())
    item_emb_mat   = torch.stack([all_item_embs[iid] for iid in item_ids])  # (n_items, EMBED_DIM)
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    # Sample users for speed (encoding 10K users is fine but 100K would be slow)
    uid_list = list(user_features.keys())
    if len(uid_list) > max_users:
        uid_list = random.sample(uid_list, max_users)

    # Encode sampled users
    user_feats_typed = {uid: ("user", user_features[uid]) for uid in uid_list}
    all_user_embs    = encode_all(model, user_feats_typed)

    # Build per-user positive set (fast lookup)
    user_positives: dict[str, set[str]] = defaultdict(set)
    for uid, iid in positive_set:
        if uid in all_user_embs:
            user_positives[uid].add(iid)

    hard_neg_triplets: list[tuple] = []
    for uid in uid_list:
        u_emb = all_user_embs[uid].unsqueeze(0)                          # (1, EMBED_DIM)
        sims  = F.cosine_similarity(u_emb, item_emb_mat).cpu().clone()  # (n_items,)

        # Mask out this user's positives and themselves
        for iid in user_positives.get(uid, set()) | {uid}:
            if iid in item_id_to_idx:
                sims[item_id_to_idx[iid]] = -float("inf")

        valid_k = min(n_per_user, int((sims > -float("inf")).sum().item()))
        if valid_k == 0:
            continue

        top_idxs = torch.topk(sims, k=valid_k).indices.tolist()
        u_feat   = user_features[uid]
        for idx in top_idxs:
            iid         = item_ids[idx]
            itype, ivec = item_features[iid]
            hard_neg_triplets.append((u_feat, ivec, itype, 0, 1.0))

    return hard_neg_triplets


# ─── Validation metrics ─────────────────────────────────────────────────────────

def evaluate_recall_ndcg(
    model: HetEncoder,
    val_data: dict,
    k: int = 10,
    max_users: int = 500,
) -> dict:
    """
    Compute Recall@K and NDCG@K on held-out validation pairs.

    For each user in val_data, ranks ALL items in the corpus by cosine similarity
    to the user's embedding, then checks how many held-out positive items appear
    in the top-K results.

    Args:
        val_data:  dict from build_training_data — {user_features, item_features, val_pairs}
        k:         ranking cut-off (default 10)
        max_users: sample this many val users for speed (default 500)

    Returns:
        {"recall@k": float, "ndcg@k": float, "k": int, "n_users": int}
    """
    user_features = val_data["user_features"]
    item_features = val_data["item_features"]
    val_pairs     = val_data["val_pairs"]

    # Group held-out pairs by user
    user_holdouts: dict[str, set[str]] = defaultdict(set)
    for uid, iid in val_pairs:
        user_holdouts[uid].add(iid)

    eval_users = [uid for uid in user_holdouts if uid in user_features]
    if len(eval_users) > max_users:
        eval_users = random.sample(eval_users, max_users)

    if not eval_users:
        return {"recall@k": 0.0, "ndcg@k": 0.0, "k": k, "n_users": 0}

    # Encode all corpus items in batch
    all_item_embs = encode_all(model, item_features)
    item_ids      = list(item_features.keys())
    item_emb_mat  = torch.stack([all_item_embs[iid] for iid in item_ids])  # (n_items, EMBED_DIM)

    recalls: list[float] = []
    ndcgs:   list[float] = []

    for uid in eval_users:
        holdout_ids = user_holdouts[uid]
        u_emb = model.encode("user", user_features[uid]).unsqueeze(0)  # (1, EMBED_DIM)
        sims  = F.cosine_similarity(u_emb, item_emb_mat)               # (n_items,)

        top_k_idxs = torch.topk(sims, k=min(k, len(item_ids))).indices.tolist()
        top_k_ids  = {item_ids[i] for i in top_k_idxs}

        hits   = len(top_k_ids & holdout_ids)
        recall = hits / len(holdout_ids)
        recalls.append(recall)

        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, idx in enumerate(top_k_idxs)
            if item_ids[idx] in holdout_ids
        )
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(holdout_ids), k)))
        ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    return {
        "recall@k": sum(recalls) / len(recalls),
        "ndcg@k":   sum(ndcgs)   / len(ndcgs),
        "k":        k,
        "n_users":  len(recalls),
    }


# ─── Training ──────────────────────────────────────────────────────────────────

def train(
    triplets: list[tuple],
    val_data: Optional[dict] = None,
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
    Train HetEncoder from 5-tuple (anchor_vec, item_vec, item_type, label, weight) triplets.

    Args:
        triplets:          training data from build_training_data()["train_triplets"]
        val_data:          optional — enables Recall@K / NDCG@K evaluation
        epochs:            max training epochs
        batch_size:        mini-batch size
        patience:          early stopping — stop if val Recall@K doesn't improve for
                           this many epochs (0 = disabled)
        eval_every:        evaluate validation metrics every N epochs
        hard_neg_fn:       callable(model) → list of hard neg 5-tuples
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

        for batch_idx, (batch_anchors, batch_items, item_types, batch_labels, batch_weights) in enumerate(loader):
            batch_anchors  = batch_anchors.to(device)
            batch_items    = batch_items.to(device)
            batch_labels   = batch_labels.to(device)
            batch_weights  = batch_weights.to(device)

            components: dict = {}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=_AUTOCAST_DTYPE, enabled=_AUTOCAST_ENABLED):
                anchor_emb, item_emb = model(batch_anchors, batch_items, item_types, use_graph=True)
                loss = contrastive_loss(
                    anchor_emb, item_emb, batch_labels, batch_weights,
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
            metrics = evaluate_recall_ndcg(model, val_data, k=10, max_users=300)
            r_at_k  = metrics["recall@k"]
            n_at_k  = metrics["ndcg@k"]

            is_best = r_at_k > best_recall
            star    = "  ★ best" if is_best else ""

            print(
                f"\r  ep {epoch+1:>4}/{epochs}"
                f"  loss: {avg_loss:.4f}{breakdown}"
                f"  {epoch_time:.1f}s{eta_str}"
                f"  Recall@10: {r_at_k:.4f}"
                f"  NDCG@10: {n_at_k:.4f}"
                f"  (n={metrics['n_users']}){star}",
                flush=True,
            )

            if is_best:
                best_recall   = r_at_k
                best_weights  = copy.deepcopy(model.state_dict())
                no_improve_ep = 0
                if checkpoint_path:
                    torch.save(best_weights, checkpoint_path)
                    print(f"       ✓ Saved best checkpoint → {checkpoint_path}", flush=True)
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
        else:
            # Plain epoch log (every 10 epochs when no val, every epoch otherwise)
            if val_data is None and (epoch + 1) % 10 == 0:
                print(
                    f"\r  ep {epoch+1:>4}/{epochs}"
                    f"  loss: {avg_loss:.4f}{breakdown}"
                    f"  {epoch_time:.1f}s{eta_str}",
                    flush=True,
                )
            elif val_data is not None:
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
            print(f"  ⟳  Recovery checkpoint → {_recovery_path}", flush=True)

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
