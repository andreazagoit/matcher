"""
Build a PyG HeteroData graph from training-data/*.json.

Graph topology:
  user     ──attends──────────► event
  user     ──joins────────────► space
  event    ──hosted_by────────► space
  user     ──similar_to───────► user      (co-attendance)
  user     ──likes_category───► category  (impressions: liked + viewed)
  event    ──tagged_with──────► category
  space    ──tagged_with_space► category
  + reverse edges for all of the above (bidirectional message passing)

Returns:
  {
    "train_data": HeteroData,
    "val_data":   {anchor_features, item_features, val_pairs, seen_train_by_anchor}
    "node_ids":   {"user": [...], "event": [...], "space": [...], "category": [...]}
  }
"""

from __future__ import annotations
import json
import os
import random
from collections import defaultdict
from typing import Optional

import torch
from torch_geometric.data import HeteroData

from hgt.config import TRAINING_DATA_DIR, CATEGORY_EMBED_DIM
from hgt.features import build_user_features, build_event_features, build_space_features
from hgt.utils import days_until


# ── JSON loaders ──────────────────────────────────────────────────────────────

def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Feature builders (JSON dict → float list) ─────────────────────────────────

def _user_vec(
    u: dict,
    categories_data: dict[str, list[float]],
    category_weights: dict[str, float],
) -> list[float]:
    # Use all categories where the user has a positive impression weight
    active_cats = [c for c in category_weights if c in categories_data and category_weights[c] > 0]

    if not active_cats:
        return build_user_features(
            birthdate=u.get("birthdate"),
            category_embeddings=[[0.0] * CATEGORY_EMBED_DIM],
            category_weights=[1.0],
            gender=u.get("gender"),
            relationship_intent=u.get("relationshipIntent") or [],
            smoking=u.get("smoking"),
            drinking=u.get("drinking"),
            activity_level=u.get("activityLevel"),
        )

    cat_embs = [categories_data[c] for c in active_cats]
    weights  = [category_weights[c] for c in active_cats]

    return build_user_features(
        birthdate=u.get("birthdate"),
        category_embeddings=cat_embs,
        category_weights=weights,
        gender=u.get("gender"),
        relationship_intent=u.get("relationshipIntent") or [],
        smoking=u.get("smoking"),
        drinking=u.get("drinking"),
        activity_level=u.get("activityLevel"),
    )


def _event_vec(e: dict, categories_data: dict[str, list[float]]) -> list[float]:
    starts_at = e.get("startsAt")
    e_cats = e.get("categories") or []
    cat_embs = [categories_data[c] for c in e_cats if c in categories_data]
    if not cat_embs:
        cat_embs = [[0.0] * CATEGORY_EMBED_DIM]

    return build_event_features(
        category_embeddings=cat_embs,
        starts_at=starts_at,
        avg_attendee_age=e.get("avgAttendeeAge"),
        attendee_count=int(e.get("attendeeCount") or 0),
        days_until_event=days_until(starts_at),
        max_attendees=e.get("maxAttendees"),
        is_paid=bool(e.get("isPaid")),
        price_cents=int(e.get("priceCents") or 0),
    )


def _space_vec(s: dict, categories_data: dict[str, list[float]]) -> list[float]:
    s_cats = s.get("categories") or []
    cat_embs = [categories_data[c] for c in s_cats if c in categories_data]
    if not cat_embs:
        cat_embs = [[0.0] * CATEGORY_EMBED_DIM]

    return build_space_features(
        category_embeddings=cat_embs,
        avg_member_age=s.get("avgMemberAge"),
        member_count=int(s.get("memberCount") or 0),
        event_count=int(s.get("eventCount") or 0),
    )


# ── Edge helpers ───────────────────────────────────────────────────────────────

def _to_edge_index(src: list[int], dst: list[int]) -> torch.Tensor:
    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def _reverse(edge: torch.Tensor) -> torch.Tensor:
    return edge.flip(0)


# ── User–user co-attendance edges ─────────────────────────────────────────────

def _user_user_edges(
    events_raw: list[dict],
    spaces_raw: list[dict],
    attendees_raw: list[dict],
    members_raw: list[dict],
    u_idx: dict[str, int],
    e_idx: dict[str, int],
    s_idx: dict[str, int],
    min_coattend: int = 2,
    top_k: int = 10,
) -> tuple[list[int], list[int], list[float]]:
    by_item: dict[str, set[str]] = defaultdict(set)
    for r in attendees_raw:
        uid = r.get("userId")
        eid = r.get("eventId")
        if uid in u_idx and eid in e_idx:
            by_item[eid].add(uid)

    for r in members_raw:
        uid = r.get("userId")
        sid = r.get("spaceId")
        if uid in u_idx and sid in s_idx:
            by_item[sid].add(uid)

    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    for users in by_item.values():
        ul = sorted(users)
        if len(ul) > 200:
            ul = random.sample(ul, 200)
        for i in range(len(ul)):
            for j in range(i + 1, len(ul)):
                pair_count[(ul[i], ul[j])] += 1

    user_neighbors: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (u1, u2), cnt in pair_count.items():
        if cnt >= min_coattend:
            user_neighbors[u1].append((u2, cnt))
            user_neighbors[u2].append((u1, cnt))

    bonds: set[tuple[str, str]] = set()
    for uid, peers in user_neighbors.items():
        peers.sort(key=lambda x: x[1], reverse=True)
        for peer_id, _ in peers[:top_k]:
            bonds.add(tuple(sorted((uid, peer_id))))  # type: ignore[arg-type]

    max_coattend = float(max(pair_count.values(), default=1))
    src, dst, weights = [], [], []
    for u1, u2 in bonds:
        cnt = pair_count.get((u1, u2)) or pair_count.get((u2, u1))
        src.append(u_idx[u1])
        dst.append(u_idx[u2])
        weights.append(float(cnt) / max_coattend)

    return src, dst, weights


def _shared_category_edges(
    entities: list[dict],
    e_idx: dict[str, int],
    top_k: int = 5,
) -> tuple[list[int], list[int], list[float]]:
    """Connect entities (events or spaces) based on shared categories."""
    cat_to_entities: dict[str, list[str]] = defaultdict(list)
    for ent in entities:
        for c in ent.get("categories", []):
            cat_to_entities[c].append(ent["id"])

    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    for ents in cat_to_entities.values():
        el = sorted(ents)
        for i in range(len(el)):
            for j in range(i + 1, len(el)):
                pair_count[(el[i], el[j])] += 1

    ent_neighbors: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (e1, e2), cnt in pair_count.items():
        if cnt >= 2:
            ent_neighbors[e1].append((e2, cnt))
            ent_neighbors[e2].append((e1, cnt))

    bonds: set[tuple[str, str]] = set()
    for eid, peers in ent_neighbors.items():
        peers.sort(key=lambda x: x[1], reverse=True)
        for peer_id, _ in peers[:top_k]:
            bonds.add(tuple(sorted((eid, peer_id))))  # type: ignore[arg-type]

    max_c = float(max(pair_count.values(), default=1))
    src, dst, weights = [], [], []
    for e1, e2 in bonds:
        cnt = pair_count.get((e1, e2)) or pair_count.get((e2, e1))
        src.append(e_idx[e1])
        dst.append(e_idx[e2])
        weights.append(float(cnt) / max_c)

    return src, dst, weights


def _category_category_edges(
    categories_raw: list[dict],
    c_idx: dict[str, int],
    top_k: int = 5,
) -> tuple[list[int], list[int], list[float]]:
    """Connect categories based on cosine similarity of their 64d OpenAI embeddings."""
    if not categories_raw:
        return [], [], []

    import torch.nn.functional as F
    ids = [c["id"] for c in categories_raw if c.get("embedding")]
    if not ids:
        return [], [], []

    embs = torch.tensor(
        [c["embedding"] for c in categories_raw if c.get("embedding")],
        dtype=torch.float32,
    )
    embs = F.normalize(embs, p=2, dim=1)
    sim_matrix = torch.mm(embs, embs.t())

    src, dst, weights = [], [], []
    for i in range(len(ids)):
        scores = sim_matrix[i].clone()
        scores[i] = -1.0
        vals, indices = torch.topk(scores, k=min(top_k, len(ids) - 1))
        for val, idx in zip(vals.tolist(), indices.tolist()):
            id1, id2 = ids[i], ids[idx]
            if id1 < id2:
                src.append(c_idx[id1])
                dst.append(c_idx[id2])
                weights.append(float(val))

    return src, dst, weights


# ── Main builder ──────────────────────────────────────────────────────────────

def build_graph_data(
    data_dir: str = TRAINING_DATA_DIR,
    val_ratio: float = 0.20,
    min_coattend: int = 4,
    top_k_similar: int = 10,
) -> dict:
    """
    Load JSON exports, build a HeteroData graph with train/val split on edges.

    Returns a dict with keys:
      train_data  — HeteroData
      val_data    — dict (held-out pairs for Recall@K evaluation)
      node_ids    — dict ({type: [str id, ...]}, same order as tensor rows)
    """
    users_raw     = _load(os.path.join(data_dir, "users.json"))
    events_raw    = _load(os.path.join(data_dir, "events.json"))
    spaces_raw    = _load(os.path.join(data_dir, "spaces.json"))
    attendees_raw = _load(os.path.join(data_dir, "event_attendees.json"))
    members_raw   = _load(os.path.join(data_dir, "members.json"))

    categories_raw = _load(os.path.join(data_dir, "categories.json"))
    cat_impressions_raw = _load(os.path.join(data_dir, "category_impressions.json"))

    # ── Node ID → integer index ────────────────────────────────────────────────
    user_ids     = [u["id"] for u in users_raw]
    event_ids    = [e["id"] for e in events_raw]
    space_ids    = [s["id"] for s in spaces_raw]
    category_ids = [c["id"] for c in categories_raw]

    categories_data = {c["id"]: c["embedding"] for c in categories_raw if c.get("embedding")}

    u_idx = {uid: i for i, uid in enumerate(user_ids)}
    e_idx = {eid: i for i, eid in enumerate(event_ids)}
    s_idx = {sid: i for i, sid in enumerate(space_ids)}
    c_idx = {cid: i for i, cid in enumerate(category_ids)}

    # ── Category Weights per user (from impressions + attendance signal) ───────
    # Base: explicit impressions (liked=1.0, viewed=0.3, already aggregated).
    # Boost: +0.5 for each event attended, +0.3 for each space joined.
    user_category_weights: dict[str, dict[str, float]] = {u["id"]: {} for u in users_raw}

    for imp in cat_impressions_raw:
        uid, cid, w = imp.get("userId"), imp.get("categoryId"), float(imp.get("weight", 0.0))
        if uid in user_category_weights:
            user_category_weights[uid][cid] = user_category_weights[uid].get(cid, 0.0) + w

    event_map = {e["id"]: e for e in events_raw}
    for att in attendees_raw:
        uid, eid = att.get("userId"), att.get("eventId")
        if uid in user_category_weights and eid in event_map:
            for c in event_map[eid].get("categories", []):
                user_category_weights[uid][c] = user_category_weights[uid].get(c, 0.0) + 0.5

    space_map = {s["id"]: s for s in spaces_raw}
    for mem in members_raw:
        uid, sid = mem.get("userId"), mem.get("spaceId")
        if uid in user_category_weights and sid in space_map:
            for c in space_map[sid].get("categories", []):
                user_category_weights[uid][c] = user_category_weights[uid].get(c, 0.0) + 0.3

    # ── Node feature tensors ──────────────────────────────────────────────────
    user_x     = torch.tensor([_user_vec(u, categories_data, user_category_weights[u["id"]]) for u in users_raw], dtype=torch.float32)
    event_x    = torch.tensor([_event_vec(e, categories_data) for e in events_raw], dtype=torch.float32)
    space_x    = torch.tensor([_space_vec(s, categories_data) for s in spaces_raw], dtype=torch.float32)

    # Category nodes: 64D dense embeddings directly from Postgres
    if category_ids and categories_data:
        cat_list = [categories_data.get(cid, [0.0] * CATEGORY_EMBED_DIM) for cid in category_ids]
        category_x = torch.tensor(cat_list, dtype=torch.float32)
    else:
        category_x = torch.empty((0, CATEGORY_EMBED_DIM), dtype=torch.float32)

    # ── Universal train / val split logic ─────────────────────────────────────
    val_all:   dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    train_all: dict[tuple[str, str, str, str], set[str]]  = defaultdict(set)

    def _split_and_fill(records, src_type, rel_type, dst_type_map, src_idx_map, dst_idx_map,
                        src_list, dst_list, w_list, rev_rel_type=None,
                        is_timestamped=True, allow_1_degree_holdout=False):
        by_src: dict[str, list] = defaultdict(list)
        for r in records:
            sid, did, w, ts = r
            if sid in src_idx_map and did in dst_idx_map:
                by_src[sid].append(r)

        for sid, recs in by_src.items():
            if is_timestamped:
                recs.sort(key=lambda x: x[3])
            else:
                random.shuffle(recs)

            if len(recs) == 1:
                n_v = 1 if allow_1_degree_holdout and random.random() < val_ratio else 0
            else:
                n_v = max(1, int(len(recs) * val_ratio))
                if len(recs) - n_v < 1:
                    n_v = len(recs) - 1

            t_recs = recs[:-n_v] if n_v > 0 else recs
            v_recs = recs[-n_v:] if n_v > 0 else []

            d_type = dst_type_map[recs[0][1]] if isinstance(dst_type_map, dict) else dst_type_map

            for src_id, dst_id, w, _ in t_recs:
                src_list.append(src_idx_map[src_id])
                dst_list.append(dst_idx_map[dst_id])
                w_list.append(w)
                train_all[(src_type, src_id, rel_type, d_type)].add(dst_id)

                if src_type == d_type == "user":
                    src_list.append(src_idx_map[dst_id])
                    dst_list.append(src_idx_map[src_id])
                    w_list.append(w)
                    train_all[(src_type, dst_id, rel_type, src_type)].add(src_id)

            for src_id, dst_id, w, _ in v_recs:
                val_all[(src_type, src_id, rel_type, d_type)].append(dst_id)
                if rev_rel_type:
                    val_all[(d_type, dst_id, rev_rel_type, src_type)].append(src_id)

    # 1. user → event (attends)
    inter_recs = [
        (r.get("userId"), r.get("eventId"), float(r.get("weight", 1.0)) * 2.0, r.get("created_at", ""))
        for r in attendees_raw
    ]
    attends_src, attends_dst, attends_w = [], [], []
    _split_and_fill(inter_recs, "user", "attends", "event", u_idx, e_idx, attends_src, attends_dst, attends_w, rev_rel_type="rev_attends")

    # 2. user → space (joins)
    inter_recs = [
        (r.get("userId"), r.get("spaceId"), float(r.get("weight", 1.0)) * 1.5, r.get("created_at", ""))
        for r in members_raw
    ]
    joins_src, joins_dst, joins_w = [], [], []
    _split_and_fill(inter_recs, "user", "joins", "space", u_idx, s_idx, joins_src, joins_dst, joins_w, rev_rel_type="rev_joins")

    # 3. event → space (hosted_by)
    hosted_recs = [(e["id"], e.get("spaceId"), 1.0, "") for e in events_raw if e.get("spaceId")]
    hosted_src, hosted_dst, hosted_w = [], [], []
    _split_and_fill(hosted_recs, "event", "hosted_by", "space", e_idx, s_idx, hosted_src, hosted_dst, hosted_w, rev_rel_type="rev_hosted_by", is_timestamped=False, allow_1_degree_holdout=True)

    # 4. user → user (similar_to via co-attendance)
    sim_raw_src, sim_raw_dst, sim_raw_w = _user_user_edges(events_raw, spaces_raw, attendees_raw, members_raw, u_idx, e_idx, s_idx, min_coattend, top_k_similar)
    rev_u_idx = {i: uid for uid, i in u_idx.items()}
    sim_recs = [(rev_u_idx[s], rev_u_idx[d], w, "") for s, d, w in zip(sim_raw_src, sim_raw_dst, sim_raw_w)]
    sim_src, sim_dst, sim_w = [], [], []
    _split_and_fill(sim_recs, "user", "similar_to", "user", u_idx, u_idx, sim_src, sim_dst, sim_w, rev_rel_type="similar_to", is_timestamped=False)

    # 5. user → category (likes_category, from impressions)
    uc_recs = [
        (imp.get("userId"), imp.get("categoryId"), float(imp.get("weight", 1.0)), imp.get("createdAt", ""))
        for imp in cat_impressions_raw
    ]
    uc_src, uc_dst, uc_w = [], [], []
    _split_and_fill(uc_recs, "user", "likes_category", "category", u_idx, c_idx, uc_src, uc_dst, uc_w, rev_rel_type="rev_likes_category", is_timestamped=True)

    # 6. event → category (tagged_with)
    ec_recs = [(e["id"], c, 1.0, "") for e in events_raw for c in e.get("categories", [])]
    ec_src, ec_dst, ec_w = [], [], []
    _split_and_fill(ec_recs, "event", "tagged_with", "category", e_idx, c_idx, ec_src, ec_dst, ec_w, rev_rel_type="rev_tagged_with_event", is_timestamped=False)

    # 7. space → category (tagged_with_space)
    sc_recs = [(s["id"], c, 1.0, "") for s in spaces_raw for c in s.get("categories", [])]
    sc_src, sc_dst, sc_w = [], [], []
    _split_and_fill(sc_recs, "space", "tagged_with_space", "category", s_idx, c_idx, sc_src, sc_dst, sc_w, rev_rel_type="rev_tagged_with_space", is_timestamped=False)

    # 8. Similarity discovery (validation only)
    ee_src, ee_dst, ee_w = _shared_category_edges(events_raw, e_idx, top_k=top_k_similar)
    for s, d, _ in zip(ee_src, ee_dst, ee_w):
        val_all[("event", event_ids[s], "similarity", "event")].append(event_ids[d])

    ss_src, ss_dst, ss_w = _shared_category_edges(spaces_raw, s_idx, top_k=top_k_similar)
    for s, d, _ in zip(ss_src, ss_dst, ss_w):
        val_all[("space", space_ids[s], "similarity", "space")].append(space_ids[d])

    cc_src, cc_dst, cc_w = _category_category_edges(categories_raw, c_idx, top_k=top_k_similar)
    for s, d, _ in zip(cc_src, cc_dst, cc_w):
        val_all[("category", category_ids[s], "similarity", "category")].append(category_ids[d])

    # ── Populate PyG HeteroData ───────────────────────────────────────────────
    data = HeteroData()

    data["user"].x     = user_x
    data["event"].x    = event_x
    data["space"].x    = space_x
    data["category"].x = category_x

    def _add_edge(cond, src_list, dst_list, src_type, rel, dst_type):
        if cond:
            ei = _to_edge_index(src_list, dst_list)
            data[src_type, rel, dst_type].edge_index = ei
            return ei
        return None

    ei = _add_edge(attends_src, attends_src, attends_dst, "user",  "attends",      "event")
    if ei is not None: data["event", "rev_attends", "user"].edge_index = _reverse(ei)

    ei = _add_edge(joins_src, joins_src, joins_dst, "user", "joins", "space")
    if ei is not None: data["space", "rev_joins", "user"].edge_index = _reverse(ei)

    ei = _add_edge(hosted_src, hosted_src, hosted_dst, "event", "hosted_by", "space")
    if ei is not None: data["space", "rev_hosted_by", "event"].edge_index = _reverse(ei)

    ei = _add_edge(uc_src, uc_src, uc_dst, "user", "likes_category", "category")
    if ei is not None: data["category", "rev_likes_category", "user"].edge_index = _reverse(ei)

    ei = _add_edge(ec_src, ec_src, ec_dst, "event", "tagged_with", "category")
    if ei is not None: data["category", "rev_tagged_with_event", "event"].edge_index = _reverse(ei)

    ei = _add_edge(sc_src, sc_src, sc_dst, "space", "tagged_with_space", "category")
    if ei is not None: data["category", "rev_tagged_with_space", "space"].edge_index = _reverse(ei)

    # ── Validation data ────────────────────────────────────────────────────────
    val_pairs_list = [
        (atype, aid, rel_type, itype, iid)
        for (atype, aid, rel_type, itype), ids in val_all.items()
        for iid in ids
    ]

    seen_train_flattened: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (atype, aid, rel_type, itype), ids in train_all.items():
        seen_train_flattened[(atype, aid)] |= ids

    val_data = _build_val_data(
        users_raw, events_raw, spaces_raw,
        seen_train_flattened, val_pairs_list, categories_data, user_category_weights,
    )

    node_ids = {
        "user":     user_ids,
        "event":    event_ids,
        "space":    space_ids,
        "category": category_ids,
    }

    print(
        f"Graph: {len(user_ids)} users, {len(event_ids)} events, "
        f"{len(space_ids)} spaces, {len(category_ids)} categories | "
        f"attends={len(attends_src)} joins={len(joins_src)} "
        f"hosted_by={len(hosted_src)} "
        f"likes_category={len(uc_src)} tagged={len(ec_src) + len(sc_src)}"
    )

    return {"train_data": data, "val_data": val_data, "node_ids": node_ids}


# ── Validation helper ──────────────────────────────────────────────────────────

def _build_val_data(
    users_raw: list[dict],
    events_raw: list[dict],
    spaces_raw: list[dict],
    seen_train: dict[tuple[str, str], set[str]],
    val_pairs: list[tuple[str, str, str, str, str]],
    categories_data: dict[str, list[float]],
    user_category_weights: dict[str, dict[str, float]],
) -> dict:
    anchor_features: dict[tuple, list[float]] = {}

    user_by_id  = {u["id"]: u for u in users_raw}
    event_by_id = {e["id"]: e for e in events_raw}
    space_by_id = {s["id"]: s for s in spaces_raw}

    all_anchors = {(atype, aid) for atype, aid, _, _, _ in val_pairs}

    for atype, aid in all_anchors:
        if atype == "user":
            u = user_by_id.get(aid)
            if u:
                anchor_features[(atype, aid)] = _user_vec(u, categories_data, user_category_weights.get(aid, {}))
        elif atype == "event":
            e = event_by_id.get(aid)
            if e:
                anchor_features[(atype, aid)] = _event_vec(e, categories_data)
        elif atype == "space":
            s = space_by_id.get(aid)
            if s:
                anchor_features[(atype, aid)] = _space_vec(s, categories_data)
        elif atype == "category":
            emb = categories_data.get(aid)
            if emb:
                anchor_features[(atype, aid)] = emb

    item_features: dict[str, tuple[str, list[float]]] = {}
    for e in events_raw:
        item_features[e["id"]] = ("event", _event_vec(e, categories_data))
    for s in spaces_raw:
        item_features[s["id"]] = ("space", _space_vec(s, categories_data))
    for u in users_raw:
        item_features[u["id"]] = ("user", _user_vec(u, categories_data, user_category_weights.get(u["id"], {})))

    return {
        "anchor_features":      anchor_features,
        "item_features":        item_features,
        "val_pairs":            val_pairs,
        "seen_train_by_anchor": seen_train,
    }
