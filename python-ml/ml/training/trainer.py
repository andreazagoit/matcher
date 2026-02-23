"""
Training loop, hard-negative mining, and model persistence.
"""

from __future__ import annotations
import copy
import os
import random
import time
from collections import defaultdict
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.config import LEARNING_RATE, EPOCHS, BATCH_SIZE, MODEL_WEIGHTS_PATH, NUM_WORKERS
from ml.modeling.hgt import (
    HetEncoder,
    TripletDataset,
    _collate,
    contrastive_loss,
    encode_all,
    device,
    _AUTOCAST_ENABLED,
    _AUTOCAST_DTYPE,
)
from ml.evaluation.metrics import evaluate_recall_ndcg


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
    num_workers: int = NUM_WORKERS,
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
        num_workers:       DataLoader worker processes for parallel CPU data loading.
                           0 = single-threaded (safe everywhere).
                           2–4 recommended on M1/M2 Mac to overlap CPU prep with MPS training.
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

    # Linear warmup for the first N epochs, then cosine decay for the rest.
    # Warmup prevents large gradient steps on a random init; 5 epochs is enough
    # to stabilise before the cosine schedule starts shrinking the LR.
    _n_warmup = min(5, max(1, epochs // 10))
    _warmup   = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=_n_warmup,
    )
    _cosine   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - _n_warmup),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[_warmup, _cosine], milestones=[_n_warmup],
    )

    base_triplets    = triplets[:]
    current_triplets = base_triplets[:]
    dataset = TripletDataset(current_triplets)
    _loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    loader  = DataLoader(dataset, **_loader_kwargs)

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
            loader  = DataLoader(dataset, **_loader_kwargs)
            n_batches = len(loader)
            print(
                f"  ↺  +{len(hard_negs):,} hard negs  "
                f"→ {len(current_triplets):,} total triplets",
                flush=True,
            )

        # ── Training step ──────────────────────────────────────────────────
        model.train()
        total_loss = total_pos = total_neg = total_info_same = total_info_cross = 0.0

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
                # DropGraph: half of steps train without graph attention so the
                # inference path (graph-free encoder-only) gets equal gradient to
                # the graph path.  At 15% the cross-type tasks (user→event/space)
                # stayed at ~0 because the HGT layers bridged cross-type alignment
                # during training but the raw encoders were left in misaligned
                # subspaces.  50/50 gives both paths the same gradient share.
                _use_graph = random.random() > 0.50
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
                    anchor_types=anchor_types,
                    item_types=item_types,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss       += loss.item()
            total_pos        += components.get("pos",        0.0)
            total_neg        += components.get("neg",        0.0)
            total_info_same  += components.get("info_same",  0.0)
            total_info_cross += components.get("info_cross", 0.0)

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
            f"  nce↑ {total_info_same/n_batches:.4f}"
            f"  nce× {total_info_cross/n_batches:.4f}]"
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
            r_at_k = metrics.get("micro_recall@k", metrics.get("macro_recall@k", metrics.get("recall@k", 0.0)))
            n_at_k = metrics.get("micro_ndcg@k",   metrics.get("macro_ndcg@k",   metrics.get("ndcg@k",   0.0)))

            # Early stopping driven by cross-type user recall (the product-critical
            # tasks). Same-type tasks (event→event, space→space) saturate by epoch 2
            # and would dominate micro-average, hiding lack of cross-type progress.
            per_task_m = metrics.get("per_task") or {}
            _cross_recalls = [
                per_task_m[k]["recall@k"]
                for k in ("user->event", "user->space")
                if k in per_task_m
            ]
            primary_recall = (
                sum(_cross_recalls) / len(_cross_recalls)
                if _cross_recalls
                else r_at_k
            )

            is_best = primary_recall > best_recall
            star    = "  ★ best" if is_best else ""

            print(
                f"\r  ep {epoch+1:>4}/{epochs}"
                f"  loss: {avg_loss:.4f}{breakdown}"
                f"  {epoch_time:.1f}s{eta_str}"
                f"  Recall@10: {r_at_k:.4f}"
                f"  NDCG@10: {n_at_k:.4f}"
                f"  (n={metrics.get('n_anchors', metrics.get('n_users', '?'))}){star}",
                flush=True,
            )
            per_task = metrics.get("per_task") or {}
            if per_task:
                parts = []
                for task_name in sorted(per_task.keys()):
                    task = per_task[task_name]
                    parts.append(
                        f"{task_name} R@10:{task['recall@k']:.4f} "
                        f"N@10:{task['ndcg@k']:.4f} (n={task.get('n_users', task.get('n_anchors', '?'))})"
                    )
                print(f"       {' | '.join(parts)}", flush=True)

            if is_best:
                best_recall   = primary_recall
                best_weights  = copy.deepcopy(model.state_dict())
                no_improve_ep = 0
                if checkpoint_path:
                    torch.save(best_weights, checkpoint_path)
            else:
                no_improve_ep += eval_every
                if patience > 0 and no_improve_ep >= patience:
                    print(
                        f"\n  ✗  Early stopping at epoch {epoch+1}"
                        f"  (best cross-type Recall@10={best_recall:.4f},"
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
        print(f"\n  ✓  Restored best checkpoint  (cross-type Recall@10={best_recall:.4f})", flush=True)

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
