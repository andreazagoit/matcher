"""
Train the HGT model on the heterogeneous graph.

Loss: BPR (Bayesian Personalized Ranking) — pairwise ranking loss for
full-graph training. For each positive (user, item) edge we sample 1 random
negative; BPR pushes pos_score > neg_score via logsigmoid of their difference.
BCE works better for mini-batch (as in the PyG example), but BPR is more
robust for full-graph setups where embeddings are recomputed every step.

Usage:
  python -m ml.train
  python -m ml.train --epochs 100 --lr 1e-3 --patience 30 --eval-every 5
"""

from __future__ import annotations
import argparse
import copy
import math
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from hgt.config import (
    LEARNING_RATE, EPOCHS, NEGATIVE_SAMPLES,
    MODEL_WEIGHTS_PATH, TRAINING_DATA_DIR,
)
from hgt.graph import build_graph_data
from hgt.model import HetEncoder, save_model, load_model, device, _AUTOCAST, _DTYPE
from hgt.config import (
    MODEL_WEIGHTS_PATH, TRAINING_DATA_DIR,
    EMBED_DIM, HIDDEN_DIM, HGT_HEADS, HGT_LAYERS,
    NODE_TYPES, METADATA
)
import os

def load_model_and_graph(weights_path: str = MODEL_WEIGHTS_PATH, data_dir: str = TRAINING_DATA_DIR):
    print(f"Loading data from {data_dir}...")
    bundle = build_graph_data(data_dir=data_dir)
    
    model = HetEncoder().to(device)
    
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        
    model.eval()
    return model, bundle

# ── Configuration ─────────────────────────────────────────────────────────────

# Edges used for BCE ranking loss — the only training signal.
# All other edge types still contribute via HGT message passing.
_BPR_EDGES: list[tuple[str, str, str]] = [
    ("user",  "attends",             "event"),
    ("user",  "joins",               "space"),
    ("user",  "similar_to",          "user"),
    ("user",  "likes",               "tag"),
    ("event", "hosted_by",           "space"),
    ("event", "tagged_with",         "tag"),
    ("space", "tagged_with_space",   "tag"),
    # Reverse edges for full-spectrum supervision
    ("event", "rev_attends",         "user"),
    ("space", "rev_joins",           "user"),
    ("tag",   "rev_likes",           "user"),
    ("space", "rev_hosted_by",       "event"),
    ("tag",   "rev_tagged_with_event", "event"),
    ("tag",   "rev_tagged_with_space", "space"),
]

_EDGES_PER_TYPE = 512   # positive pairs sampled per edge type per step


# Margin from original BPR conceptually.
# Using standard MarginRankingLoss optimizes ranking constraints efficiently.
_margin_loss = torch.nn.MarginRankingLoss(margin=0.5)

def _bpr_loss(
    anchor: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluates relative ranking via MarginRankingLoss.
    Constraints: anchor * pos > anchor * neg + margin
    """
    n = anchor.size(0)
    n_neg = neg.size(0) // n
    
    # [n, n_neg] expansions
    pos_score = (anchor * pos).sum(-1, keepdim=True).expand(n, n_neg)
    neg_score = (anchor.unsqueeze(1) * neg.view(n, n_neg, -1)).sum(-1)
    
    # Target tensor of ones (pos_score should be larger than neg_score)
    ones = torch.ones_like(pos_score)
    return _margin_loss(pos_score.flatten(), neg_score.flatten(), ones.flatten())


# ── Training step ─────────────────────────────────────────────────────────────

def _train_step(
    model: HetEncoder,
    data,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=_DTYPE, enabled=_AUTOCAST):
        emb = model.forward_graph(
            {t: data[t].x.to(device) for t in data.node_types},
            {et: data[et].edge_index.to(device) for et in data.edge_types},
        )

    total_loss = torch.tensor(0.0, device=device)

    for src_t, rel, dst_t in _BPR_EDGES:
        et = (src_t, rel, dst_t)
        if et not in data.edge_types:
            continue
        edges = data[et].edge_index.to(device)
        if edges.size(1) == 0:
            continue

        n   = min(_EDGES_PER_TYPE, edges.size(1))
        idx = torch.randperm(edges.size(1), device=device)[:n]

        a_emb = F.normalize(emb[src_t][edges[0, idx]], dim=-1)   # [n, D]
        p_emb = F.normalize(emb[dst_t][edges[1, idx]], dim=-1)   # [n, D]

        # ── Hard Negative Mining
        # Instead of 5 purely randoms, we do:
        # Half random global (easy negatives, teach basic distinction)
        # Half in-batch positives (hard negatives: popular items currently active but unlinked to this anchor)
        
        n_hard = max(1, NEGATIVE_SAMPLES // 2)
        n_easy = max(1, NEGATIVE_SAMPLES - n_hard)
        
        # Easy: Uniform random over all items in catalog
        easy_idx = torch.randint(0, emb[dst_t].size(0), (n * n_easy,), device=device)
        
        # Hard: Sample from the batch's positive items recursively 
        # (Items that other users liked in this batch, very prevalent/popular overall)
        batch_pos_pool = edges[1, idx]
        hard_idx = batch_pos_pool[torch.randint(0, n, (n * n_hard,), device=device)]
        
        neg_idx = torch.cat([easy_idx, hard_idx], dim=0)
        
        n_emb   = F.normalize(emb[dst_t][neg_idx], dim=-1)

        total_loss = total_loss + _bpr_loss(a_emb, p_emb, n_emb)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item()


# ── Evaluation ────────────────────────────────────────────────────────────────

def _ndcg(top_k: list[int], ids: list[str], holdout: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(r + 2)
        for r, i in enumerate(top_k)
        if ids[i] in holdout
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(holdout), k)))
    return dcg / ideal if ideal > 0 else 0.0


@torch.no_grad()
def evaluate(
    model: HetEncoder,
    train_data,
    val_data: dict,
    node_ids: dict[str, list[str]],
    k: int = 10,
    max_anchors: int = 384,
    min_val_items: int = 2,
    fixed_anchors: Optional[list] = None,
) -> dict:
    """
    Evaluate Recall@K / NDCG@K using full-graph embeddings.
    Mirrors production: embeddings come from forward_graph (same as ml:sync).

    min_val_items: skip anchors with fewer than this many val items.
    Anchors with only 1 val item produce binary 0/1 recall per anchor,
    which introduces too much noise for meaningful epoch-level tracking.
    """
    model.eval()
    emb = model.forward_graph(
        {t: train_data[t].x.to(device) for t in train_data.node_types},
        {et: train_data[et].edge_index.to(device) for et in train_data.edge_types},
    )

    # Pre-build ID → index maps
    id_to_idx = {
        t: {iid: i for i, iid in enumerate(ids)}
        for t, ids in node_ids.items()
    }

    # Group val pairs by anchor
    by_anchor: dict[tuple, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for atype, aid, itype, iid in val_data["val_pairs"]:
        by_anchor[(atype, aid)][itype].add(iid)

    seen_train = val_data.get("seen_train_by_anchor", {})

    def _n_val(anchor):
        return sum(len(v) for v in by_anchor[anchor].values())

    if fixed_anchors is not None:
        all_anchors = [a for a in fixed_anchors if _n_val(a) >= min_val_items]
    else:
        all_anchors = [a for a in by_anchor if _n_val(a) >= min_val_items]
        if len(all_anchors) > max_anchors:
            all_anchors = random.sample(all_anchors, max_anchors)

    per_task_r: dict[tuple, list[float]] = defaultdict(list)
    per_task_n: dict[tuple, list[float]] = defaultdict(list)

    for atype, aid in all_anchors:
        aidx = id_to_idx.get(atype, {}).get(aid)
        if aidx is None:
            continue
        a_emb = emb[atype][aidx].cpu().unsqueeze(0)   # [1, D]
        seen  = seen_train.get((atype, aid), set())

        for dst_type, pos_ids in by_anchor[(atype, aid)].items():
            if not pos_ids or dst_type not in emb:
                continue
            dst_ids    = node_ids[dst_type]
            dst_idx_map = id_to_idx[dst_type]
            mat        = emb[dst_type].cpu()           # [N, D]

            sims = F.cosine_similarity(a_emb, mat).clone()

            # Mask: seen train items + self
            for exc in seen | {aid}:
                idx = dst_idx_map.get(exc)
                if idx is not None:
                    sims[idx] = -float("inf")

            valid_ids = [iid for iid in pos_ids if iid in dst_idx_map]
            if not valid_ids:
                continue

            k_eff  = min(k, int((sims > -float("inf")).sum()))
            top_k  = torch.topk(sims, k=k_eff).indices.tolist()
            top_k_ids = {dst_ids[i] for i in top_k}

            r = len(top_k_ids & set(valid_ids)) / len(valid_ids)
            n = _ndcg(top_k, dst_ids, set(valid_ids), k)
            per_task_r[(atype, dst_type)].append(r)
            per_task_n[(atype, dst_type)].append(n)

    # Ensure all 16 combinations (4x4 matrix) are reported for full observability
    node_types = ["user", "event", "space", "tag"]
    for a in node_types:
        for b in node_types:
            key = (a, b)
            if key not in per_task_r:
                per_task_r[key] = []
                per_task_n[key] = []

    per_task = {
        f"{a}->{b}": {
            "recall@k": sum(rs) / len(rs) if rs else 0.0,
            "ndcg@k":   sum(per_task_n[(a, b)]) / len(per_task_n[(a, b)]) if rs else 0.0,
            "n_anchors": len(rs),
        }
        for (a, b), rs in sorted(per_task_r.items())
    }

    total_n   = sum(v["n_anchors"] for v in per_task.values())
    micro_r   = (
        sum(v["recall@k"] * v["n_anchors"] for v in per_task.values()) / total_n
        if total_n else 0.0
    )
    
    # Primary signal: mean recall across CORE interaction types
    core_keys = ["user->event", "user->space", "user->tag"]
    active_core = [per_task[k]["recall@k"] for k in core_keys if k in per_task]
    primary_r = sum(active_core) / len(active_core) if active_core else micro_r

    return {
        "primary_recall": primary_r,
        "micro_recall@k": micro_r,
        "per_task":        per_task,
        "n_anchors":       total_n,
        "k":               k,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    eval_every: int = 5,
    patience: int = 6,
    resume: bool = False,
    data_dir: str = TRAINING_DATA_DIR,
    weights_path: str = MODEL_WEIGHTS_PATH,
) -> HetEncoder:
    """patience: stop after this many evals with no improvement (0 = disabled)."""
    print(f"Loading graph data from {data_dir} …")
    bundle     = build_graph_data(data_dir=data_dir)
    train_data = bundle["train_data"]
    val_data   = bundle["val_data"]
    node_ids   = bundle["node_ids"]

    model = load_model(weights_path) if resume else None
    if model is None:
        model = HetEncoder().to(device)
        print("Starting fresh model.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # LambdaLR: linear warmup for first 5 epochs, then cosine annealing
    _warmup = 5
    def _lr_lambda(ep: int) -> float:
        if ep < _warmup:
            return 0.1 + 0.9 * (ep / _warmup)
        progress = (ep - _warmup) / max(1, epochs - _warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # Fix eval anchors once — ensures consistent metric across all epochs
    # Only include anchors with >=2 val items: single-item anchors produce
    # binary 0/1 recall per anchor which dominates the metric with noise.
    anchors_with_val: set[tuple] = set()
    for atype, aid, _itype, _iid in val_data["val_pairs"]:
        anchors_with_val.add((atype, aid))

    by_anchor_count: dict[tuple, int] = defaultdict(int)
    for atype, aid, _, _ in val_data["val_pairs"]:
        by_anchor_count[(atype, aid)] += 1

    # Balanced anchor sampling: take up to 128 anchors per type to ensure stable 'n' for all tasks.
    by_type: dict[str, list] = defaultdict(list)
    for atype, aid in val_data["anchor_features"]:
        if (atype, aid) in anchors_with_val and by_anchor_count[(atype, aid)] >= 1: # lowered to 1 to include rare links
            by_type[atype].append((atype, aid))
            
    rng = random.Random(0)
    balanced_anchors = []
    for atype, candidates in by_type.items():
        n_to_take = min(128, len(candidates))
        balanced_anchors.extend(rng.sample(candidates, n_to_take))
        
    fixed_eval_anchors = balanced_anchors
    print(f"  Eval anchors: {len(fixed_eval_anchors)} (balanced: {[f'{t}:{len(v)}' for t, v in by_type.items() if any(a[0]==t for a in balanced_anchors)]})")

    best_score    = -1.0
    best_state    = None
    patience_ctr  = 0
    t0            = time.time()

    for ep in range(1, epochs + 1):
        loss = _train_step(model, train_data, optimizer)
        scheduler.step()

        if ep % eval_every == 0 or ep == epochs:
            ev      = evaluate(model, train_data, val_data, node_ids,
                               fixed_anchors=fixed_eval_anchors)
            pt      = ev["per_task"]
            elapsed = time.time() - t0

            task_lines = []
            for k in sorted(pt.keys()):
                line = f"{k:14} │ R@10: {pt[k]['recall@k']:.4f} │ N@10: {pt[k]['ndcg@k']:.4f} │ n: {pt[k]['n_anchors']}"
                task_lines.append(line)
            
            task_str = "\n         ".join(task_lines)
            
            print(
                f"[{ep:>4}/{epochs}] loss={loss:.4f}  "
                f"primary_R@10={ev['primary_recall']:.4f}  micro_R@10={ev['micro_recall@k']:.4f}  "
                f"({elapsed:.1f}s)\n         {task_str}"
            )

            if ev["primary_recall"] > best_score:
                best_score   = ev["primary_recall"]
                best_state   = copy.deepcopy(model.state_dict())
                patience_ctr = 0
                save_model(model, weights_path)
                print(f"  ✓ New best R@10 = {best_score:.4f}")
            else:
                patience_ctr += 1
                if patience > 0 and patience_ctr >= patience:
                    print(f"  Early stop: no improvement for {patience} evals ({patience * eval_every} epochs).")
                    break
        else:
            if ep % max(1, eval_every // 5) == 0:
                print(f"[{ep:>4}/{epochs}] loss={loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nTraining complete. Best cross-type R@10 = {best_score:.4f}")
    return model


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train the HGT matcher model.")
    p.add_argument("--epochs",      type=int,   default=EPOCHS,         help="Max training epochs")
    p.add_argument("--lr",          type=float, default=LEARNING_RATE,  help="Learning rate")
    p.add_argument("--eval-every",  type=int,   default=5,              help="Evaluate every N epochs")
    p.add_argument("--patience",    type=int,   default=6,              help="Early stop after N evals without improvement (0=disabled)")
    p.add_argument("--resume",      action="store_true",                help="Resume from saved weights")
    p.add_argument("--data-dir",    default=TRAINING_DATA_DIR,          help="Training data directory")
    p.add_argument("--weights",     default=MODEL_WEIGHTS_PATH,         help="Model weights path")
    args = p.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        eval_every=args.eval_every,
        patience=args.patience,
        resume=args.resume,
        data_dir=args.data_dir,
        weights_path=args.weights,
    )


if __name__ == "__main__":
    main()
