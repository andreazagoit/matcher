"""
Ranking evaluation: Recall@K and NDCG@K over held-out validation pairs.
"""

from __future__ import annotations
import math
import random
from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F

from ml.modeling.hgt import HetEncoder, encode_all


def _compute_ndcg(top_k_idxs: list[int], item_ids: list[str], holdout_ids: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, idx in enumerate(top_k_idxs)
        if item_ids[idx] in holdout_ids
    )
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(holdout_ids), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


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
    anchor_features = val_data["anchor_features"]
    val_pairs = val_data["val_pairs"]
    seen_train_by_anchor = {
        key: set(items)
        for key, items in (val_data.get("seen_train_by_anchor") or {}).items()
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
        return {"recall@k": 0.0, "ndcg@k": 0.0, "k": k, "n_anchors": 0}

    if allowed_item_types:
        item_features = {
            iid: (itype, ivec)
            for iid, (itype, ivec) in item_features.items()
            if itype in allowed_item_types
        }
    if not item_features:
        return {"recall@k": 0.0, "ndcg@k": 0.0, "k": k, "n_anchors": 0}

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
        recalls.append(len(top_k_ids & all_holdout_ids) / len(all_holdout_ids))
        ndcgs.append(_compute_ndcg(top_k_idxs, item_ids, all_holdout_ids, k))

        # Per-task metrics by (anchor_type -> target_type).
        for target_type, holdout_ids in holdouts_by_type.items():
            if not holdout_ids:
                continue
            per_task_recalls[(atype, target_type)].append(len(top_k_ids & holdout_ids) / len(holdout_ids))
            per_task_ndcgs[(atype, target_type)].append(_compute_ndcg(top_k_idxs, item_ids, holdout_ids, k))

    recall_at_k = (sum(recalls) / len(recalls)) if recalls else 0.0
    ndcg_at_k = (sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0

    per_task = {
        f"{src}->{dst}": {
            "recall@k": sum(vals_r) / len(vals_r),
            "ndcg@k": sum(per_task_ndcgs[(src, dst)]) / len(per_task_ndcgs[(src, dst)]),
            "n_anchors": len(vals_r),
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
    total_n = sum(v["n_anchors"] for v in per_task.values())
    micro_recall = (
        sum(v["recall@k"] * v["n_anchors"] for v in per_task.values()) / total_n
        if total_n > 0 else recall_at_k
    )
    micro_ndcg = (
        sum(v["ndcg@k"] * v["n_anchors"] for v in per_task.values()) / total_n
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
    }
