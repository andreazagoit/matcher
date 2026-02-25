"""
Train script for the Multi-Tower architecture.
Uses the exact same BPR loss as the GNN model to pull associated entities together
and push unassociated entities apart.

Run from matcher/python-ml:
  .venv/bin/python -m multi_tower.train
"""

import argparse
import math
import time
from typing import Optional

import torch
import torch.nn.functional as F

from mt.model import MultiTower, save_model, load_model, device, MODEL_WEIGHTS_PATH
from hgt.graph import build_graph_data
from mt.config import NODE_TYPES

# Edges used for BCE ranking loss — the training signal to align the independent towers.
_BPR_EDGES: list[tuple[str, str, str]] = [
    # behavioural
    ("user",  "attends",           "event"),
    ("user",  "joins",             "space"),
    # tag alignment — critical for the multi-tower to learn tag relevance
    ("user",  "likes",             "tag"),
    ("event", "tagged_with",       "tag"),
    ("space", "tagged_with_space", "tag"),
]

_EDGES_PER_TYPE = 512


def pointwise_bce_loss_multi_tower(
    model: MultiTower,
    data,
    out_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, int, int]:
    """
    Computes pointwise BCE loss, scoring positive edges against 1 and random negatives against 0.
    """
    loss = torch.tensor(0.0, device=device)
    total_samples = 0
    correct = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    for edge_type in _BPR_EDGES:
        src_type, rel, dst_type = edge_type
        if edge_type not in data.edge_types:
            continue

        edge_index = data[edge_type].edge_index.to(device)
        E = edge_index.size(1)
        if E == 0:
            continue

        # Sample positive edges
        batch_size = min(_EDGES_PER_TYPE, E)
        idx = torch.randint(0, E, (batch_size,), device=device)
        src_pos = edge_index[0, idx]
        dst_pos = edge_index[1, idx]

        # Sample negative destinations
        num_dst_nodes = out_dict[dst_type].size(0)
        dst_neg = torch.randint(0, num_dst_nodes, (batch_size,), device=device)

        # Get embeddings
        src_emb = out_dict[src_type][src_pos]
        pos_emb = out_dict[dst_type][dst_pos]
        neg_emb = out_dict[dst_type][dst_neg]

        # Calculate Cosine Similarities and Scale (temperature ~0.05 for BCE)
        temp = 0.05
        pos_score = (src_emb * pos_emb).sum(dim=-1) / temp
        neg_score = (src_emb * neg_emb).sum(dim=-1) / temp

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        
        loss += criterion(scores, labels)
        
        # Accuracy: pos score > 0 (prob > 50%), neg score < 0 (prob < 50%)
        correct += ((pos_score > 0).sum() + (neg_score < 0).sum()).item()
        total_samples += batch_size * 2

    return loss, correct, total_samples


def evaluate(
    model: MultiTower,
    train_data,
    data,
    batch_size: int = 256,
) -> tuple[float, float, str]:
    """
    Evaluates Recall@10 on the primary relations (attends, joins).
    """
    model.eval()
    with torch.no_grad():
        out = model.forward_all({t: train_data[t].x.to(device) for t in train_data.node_types})

    def _eval_edge_type(edge_type: tuple[str,str,str]) -> tuple[float, float, int]:
        src, rel, dst = edge_type
        if edge_type not in data:
            return 0.0, 0.0, 0
        ei = data[edge_type]
        N_src = out[src].size(0)

        # Build list of active users
        active_u = torch.unique(ei[0])
        hits, dcg_sum, n_users = 0.0, 0.0, 0

        # Sub-sample active users for speed
        if len(active_u) > batch_size:
            active_u = active_u[torch.randperm(len(active_u))[:batch_size]]

        src_embs = out[src][active_u]
        dst_embs = out[dst]
        scores = src_embs @ dst_embs.T

        for i, u in enumerate(active_u):
            # Ground truth items for this user
            mask = (ei[0] == u)
            gt_items = ei[1, mask]
            if len(gt_items) == 0:
                continue

            # Ranked top 10
            top10 = torch.topk(scores[i], k=10).indices
            relevant = sum(1 for item in gt_items if item in top10)
            hits += 1 if relevant > 0 else 0
            n_users += 1

        r10 = hits / max(1, n_users)
        return r10, 0.0, n_users

    details = []
    tot_r, tot_n = 0.0, 0

    # Primary metrics: user->event and user->space
    for et in [("user", "attends", "event"), ("user", "joins", "space")]:
        r, _, n = _eval_edge_type(et)
        tot_r += r * n
        tot_n += n
        details.append(f"{et[0]}->{et[2]}: R@10={r:.3f} (n={n})")

    primary_r = tot_r / max(1, tot_n)

    # Evaluate Tag alignment indirectly (user -> tag directly via closest vector search)
    r_tag, _, n_tag = _eval_edge_type(("user", "likes", "tag"))
    details.append(f"user->tag: R@10={r_tag:.3f} (n={n_tag})")

    return primary_r, primary_r, "  ".join(details)


def train(
    data_dir: str = "training-data",
    epochs: int = 150,
    lr: float = 0.005,
    eval_every: int = 5,
    resume: bool = False,
    weights_path: str = MODEL_WEIGHTS_PATH,
) -> MultiTower:
    print(f"Loading graph data from {data_dir} …")
    bundle     = build_graph_data(data_dir=data_dir)
    train_data = bundle["train_data"]
    val_data   = bundle["val_data"]

    model = load_model(weights_path) if resume else None
    if model is None:
        model = MultiTower().to(device)
        print("Starting fresh Multi-Tower model.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine annealing without warmup (lazy learning is faster without message passing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_r = -1.0
    t0 = time.time()

    # Need a dummy forward to initialize Lazy modules
    with torch.no_grad():
         model.forward_all({t: train_data[t].x.to(device) for t in train_data.node_types})

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 1. Compute embeddings for ALL nodes
        # In multi-tower, this is just passing all nodes through their MLPs
        out_dict = model.forward_all({t: train_data[t].x.to(device) for t in train_data.node_types})

        # 2. Compute loss based on edges
        loss, correct, total = pointwise_bce_loss_multi_tower(model, train_data, out_dict)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc = correct / max(1, total)

        if ep % eval_every == 0 or ep == epochs:
            pr, mr, det = evaluate(model, train_data, val_data)
            print(f"[ {ep:3}/{epochs}] loss={loss.item():.4f} acc={acc:.3f}  primary_R@10={pr:.4f}  ({time.time()-t0:.0f}s)")
            print(f"         {det}")

            if pr > best_r:
                best_r = pr
                save_model(model, weights_path)
                print(f"  ✓ New best R@10 = {best_r:.4f}")
        else:
            print(f"[ {ep:3}/{epochs}] loss={loss.item():.4f} acc={acc:.3f}")

    print(f"\nTraining complete. Best cross-type R@10 = {best_r:.4f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training-data")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        eval_every=args.eval_every,
        resume=args.resume,
    )
