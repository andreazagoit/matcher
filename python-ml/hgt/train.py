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
    LEARNING_RATE, EPOCHS,
    MODEL_WEIGHTS_PATH, TRAINING_DATA_DIR,
)
from hgt.graph import build_graph_data
from hgt.model import HetEncoder, save_model, load_model, device, _AUTOCAST, _DTYPE


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
]

_EDGES_PER_TYPE = 512   # positive pairs sampled per edge type per step


# ── BPR loss ──────────────────────────────────────────────────────────────────

def _bpr_loss(
    anchor: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> torch.Tensor:
    """Standard BPR: positive item must rank higher than a random negative."""
    return -F.logsigmoid((anchor * pos).sum(-1) - (anchor * neg).sum(-1)).mean()


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

        a_emb = F.normalize(emb[src_t][edges[0, idx]], dim=-1)
        p_emb = F.normalize(emb[dst_t][edges[1, idx]], dim=-1)

        neg_idx = torch.randint(0, emb[dst_t].size(0), (n,), device=device)
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
    max_anchors: int = 300,
    fixed_anchors: Optional[list] = None,
) -> dict:
    """
    Evaluate Recall@K / NDCG@K using full-graph embeddings.
    Mirrors production: embeddings come from forward_graph (same as ml:sync).
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

    if fixed_anchors is not None:
        all_anchors = [a for a in fixed_anchors if a in by_anchor and any(by_anchor[a].values())]
    else:
        all_anchors = [a for a in by_anchor if any(by_anchor[a].values())]
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

    per_task = {
        f"{a}->{b}": {
            "recall@k": sum(rs) / len(rs),
            "ndcg@k":   sum(per_task_n[(a, b)]) / len(per_task_n[(a, b)]),
            "n_anchors": len(rs),
        }
        for (a, b), rs in per_task_r.items()
    }

    total_n   = sum(v["n_anchors"] for v in per_task.values())
    micro_r   = (
        sum(v["recall@k"] * v["n_anchors"] for v in per_task.values()) / total_n
        if total_n else 0.0
    )
    # Primary signal: mean cross-type recall (user→event, user→space)
    cross_rs  = [per_task[k]["recall@k"] for k in ("user->event", "user->space") if k in per_task]
    primary_r = sum(cross_rs) / len(cross_rs) if cross_rs else micro_r

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
    resume: bool = False,
    data_dir: str = TRAINING_DATA_DIR,
    weights_path: str = MODEL_WEIGHTS_PATH,
) -> HetEncoder:
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
    anchors_with_val: set[tuple] = set()
    for atype, aid, _itype, _iid in val_data["val_pairs"]:
        anchors_with_val.add((atype, aid))
    all_anchors = [k for k in val_data["anchor_features"] if k in anchors_with_val]
    rng = random.Random(0)
    fixed_eval_anchors = (
        rng.sample(all_anchors, min(300, len(all_anchors)))
        if len(all_anchors) > 300 else all_anchors
    )

    best_score = -1.0
    best_state = None
    t0         = time.time()

    for ep in range(1, epochs + 1):
        loss = _train_step(model, train_data, optimizer)
        scheduler.step()

        if ep % eval_every == 0 or ep == epochs:
            ev      = evaluate(model, train_data, val_data, node_ids,
                               fixed_anchors=fixed_eval_anchors)
            pt      = ev["per_task"]
            elapsed = time.time() - t0

            task_str = "  ".join(
                f"{k}: R@10={v['recall@k']:.3f} N@10={v['ndcg@k']:.3f} (n={v['n_anchors']})"
                for k, v in sorted(pt.items())
            )
            print(
                f"[{ep:>4}/{epochs}] loss={loss:.4f}  "
                f"primary_R@10={ev['primary_recall']:.4f}  micro_R@10={ev['micro_recall@k']:.4f}  "
                f"({elapsed:.0f}s)\n         {task_str}"
            )

            if ev["primary_recall"] > best_score:
                best_score = ev["primary_recall"]
                best_state = copy.deepcopy(model.state_dict())
                save_model(model, weights_path)
                print(f"  ✓ New best R@10 = {best_score:.4f}")
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
    p.add_argument("--resume",      action="store_true",                help="Resume from saved weights")
    p.add_argument("--data-dir",    default=TRAINING_DATA_DIR,          help="Training data directory")
    p.add_argument("--weights",     default=MODEL_WEIGHTS_PATH,         help="Model weights path")
    args = p.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        eval_every=args.eval_every,
        resume=args.resume,
        data_dir=args.data_dir,
        weights_path=args.weights,
    )


if __name__ == "__main__":
    main()
