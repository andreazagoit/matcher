"""
Build a PyG HeteroData graph from training-data/*.json.

Graph topology:
  user  ──attends──────► event
  user  ──joins────────► space
  event ──hosted_by────► space
  user  ──similar_to───► user   (co-attendance)
  user  ──likes────────► tag    (declared tags, binary)
  event ──tagged_with──► tag
  space ──tagged_with──► tag
  + reverse edges for all of the above (bidirectional message passing)

Returns:
  {
    "train_data": HeteroData,                  used for training / ml:sync
    "val_data":   {anchor_features, item_features, val_pairs, seen_train_by_anchor}
    "node_ids":   {"user": [...], "event": [...], "space": [...]}
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

from torch_geometric.data import HeteroData

from hgt.config import TRAINING_DATA_DIR
from hgt.features import build_user_features, build_event_features, build_space_features
from hgt.utils import days_until


# ── JSON loaders ──────────────────────────────────────────────────────────────

def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Feature builders (JSON dict → float list) ─────────────────────────────────

def _user_vec(u: dict, tags_data: dict[str, list[float]], tag_weights: dict[str, float]) -> list[float]:
    u_tags = u.get("tags") or []
    from hgt.config import TAG_EMBED_DIM
    
    # We include all tags that have an embedding AND a positive weight for this user.
    # This makes the profile "dynamic" by including tags from interactions.
    active_tags = [t for t in tag_weights if t in tags_data and tag_weights[t] > 0]
    
    if not active_tags:
        return build_user_features(
            birthdate=u.get("birthdate"),
            tag_embeddings=[[0.0] * TAG_EMBED_DIM],
            tag_weights=[1.0],
            num_tags=len(u_tags),
            gender=u.get("gender"),
            relationship_intent=u.get("relationshipIntent") or [],
            smoking=u.get("smoking"),
            drinking=u.get("drinking"),
            activity_level=u.get("activityLevel"),
        )
    
    tag_embs = [tags_data[t] for t in active_tags]
    weights = [tag_weights[t] for t in active_tags]
    
    return build_user_features(
        birthdate=u.get("birthdate"),
        tag_embeddings=tag_embs,
        tag_weights=weights,
        num_tags=len(u_tags),
        gender=u.get("gender"),
        relationship_intent=u.get("relationshipIntent") or [],
        smoking=u.get("smoking"),
        drinking=u.get("drinking"),
        activity_level=u.get("activityLevel"),
    )


def _event_vec(e: dict, tags_data: dict[str, list[float]]) -> list[float]:
    starts_at = e.get("startsAt")
    e_tags = e.get("tags") or []
    from hgt.config import TAG_EMBED_DIM
    tag_embs = [tags_data[t] for t in e_tags if t in tags_data]
    if not tag_embs:
        tag_embs = [[0.0] * TAG_EMBED_DIM]

    return build_event_features(
        tag_embeddings=tag_embs,
        num_tags=len(e_tags),
        starts_at=starts_at,
        avg_attendee_age=e.get("avgAttendeeAge"),
        attendee_count=int(e.get("attendeeCount") or 0),
        days_until_event=days_until(starts_at),
        max_attendees=e.get("maxAttendees"),
        is_paid=bool(e.get("isPaid")),
        price_cents=int(e.get("priceCents") or 0),
    )


def _space_vec(s: dict, tags_data: dict[str, list[float]]) -> list[float]:
    s_tags = s.get("tags") or []
    from hgt.config import TAG_EMBED_DIM
    tag_embs = [tags_data[t] for t in s_tags if t in tags_data]
    if not tag_embs:
        tag_embs = [[0.0] * TAG_EMBED_DIM]

    return build_space_features(
        tag_embeddings=tag_embs,
        num_tags=len(s_tags),
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
) -> tuple[list[int], list[int]]:
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

    neighbors: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (u1, u2), cnt in pair_count.items():
        if cnt >= min_coattend:
            neighbors[u1].append((u2, cnt))
            neighbors[u2].append((u1, cnt))

    src, dst, weights = [], [], []
    max_coattend = float(max(pair_count.values(), default=1))

    # To keep the graph sparse, we take top_k neighbors per user
    # but store them as unique undirected bonds (u1 < u2) for split safety
    user_neighbors = defaultdict(list)
    for (u1, u2), cnt in pair_count.items():
        if cnt >= min_coattend:
            user_neighbors[u1].append((u2, cnt))
            user_neighbors[u2].append((u1, cnt))

    bonds = set()
    for uid, peers in user_neighbors.items():
        peers.sort(key=lambda x: x[1], reverse=True)
        for peer_id, cnt in peers[:top_k]:
            bonds.add(tuple(sorted((uid, peer_id))))

    for u1, u2 in bonds:
        cnt = pair_count.get((u1, u2)) or pair_count.get((u2, u1))
        src.append(u_idx[u1])
        dst.append(u_idx[u2])
        weights.append(float(cnt) / max_coattend)
            
    return src, dst, weights


def _shared_tag_edges(entities: list[dict], e_idx: dict[str, int], top_k: int = 5) -> tuple[list[int], list[int], list[float]]:
    """Connect entities (events or spaces) based on shared tags."""
    tag_to_entities = defaultdict(list)
    for ent in entities:
        for t in ent.get("tags", []):
            tag_to_entities[t].append(ent["id"])
            
    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    for ents in tag_to_entities.values():
        el = sorted(ents)
        for i in range(len(el)):
            for j in range(i + 1, len(el)):
                pair_count[(el[i], el[j])] += 1
                
    src, dst, weights = [], [], []
    ent_neighbors = defaultdict(list)
    for (e1, e2), cnt in pair_count.items():
        if cnt >= 2: # At least 2 shared tags
            ent_neighbors[e1].append((e2, cnt))
            ent_neighbors[e2].append((e1, cnt))
            
    bonds = set()
    for eid, peers in ent_neighbors.items():
        peers.sort(key=lambda x: x[1], reverse=True)
        for peer_id, cnt in peers[:top_k]:
            bonds.add(tuple(sorted((eid, peer_id))))
            
    max_c = float(max(pair_count.values(), default=1))
    for e1, e2 in bonds:
        cnt = pair_count.get((e1, e2)) or pair_count.get((e2, e1))
        src.append(e_idx[e1])
        dst.append(e_idx[e2])
        weights.append(float(cnt) / max_c)
        
    return src, dst, weights


def _tag_tag_edges(tags_raw: list[dict], t_idx: dict[str, int], top_k: int = 5) -> tuple[list[int], list[int], list[float]]:
    """Connect tags based on cosine similarity of their 64d OpenAI embeddings."""
    if not tags_raw: return [], [], []
    
    import torch.nn.functional as F
    ids = [t["id"] for t in tags_raw if t.get("embedding")]
    if not ids: return [], [], []
    
    embs = torch.tensor([t["embedding"] for t in tags_raw if t.get("embedding")], dtype=torch.float32)
    embs = F.normalize(embs, p=2, dim=1)
    sim_matrix = torch.mm(embs, embs.t())
    
    src, dst, weights = [], [], []
    for i in range(len(ids)):
        scores = sim_matrix[i]
        scores[i] = -1.0 # Ignore self
        vals, indices = torch.topk(scores, k=min(top_k, len(ids)-1))
        for val, idx in zip(vals.tolist(), indices.tolist()):
            # Store as unique undirected bond u1 < u2
            id1, id2 = ids[i], ids[idx]
            if id1 < id2:
                src.append(t_idx[id1])
                dst.append(t_idx[id2])
                weights.append(float(val))
                
    return src, dst, weights


# ── Main builder ──────────────────────────────────────────────────────────────

def build_graph_data(
    data_dir: str = TRAINING_DATA_DIR,
    val_ratio: float = 0.20,
    min_coattend: int = 2,
    top_k_similar: int = 10,
) -> dict:
    """
    Load JSON exports, build a HeteroData graph with train/val split on edges.

    Returns a dict with keys:
      train_data  — HeteroData  (message-passing graph + train edges for loss)
      val_data    — dict        (held-out pairs for Recall@K evaluation)
      node_ids    — dict        ({type: [str id, ...]}, same order as tensor rows)
    """
    users_raw  = _load(os.path.join(data_dir, "users.json"))
    events_raw = _load(os.path.join(data_dir, "events.json"))
    spaces_raw = _load(os.path.join(data_dir, "spaces.json"))
    attendees_raw = _load(os.path.join(data_dir, "event_attendees.json"))
    members_raw = _load(os.path.join(data_dir, "members.json"))

    # ── Node ID → integer index ────────────────────────────────────────────────
    user_ids  = [u["id"] for u in users_raw]
    event_ids = [e["id"] for e in events_raw]
    space_ids = [s["id"] for s in spaces_raw]
    
    tags_raw = _load(os.path.join(data_dir, "tags.json"))
    
    # Optional logic for when TAGS is completely missing or old DB was cleared out.
    if not tags_raw:
        tag_ids = []
        tags_data = {}
    else:    
        tag_ids = [t["id"] for t in tags_raw]
        # embedding matrix lookup for O(1) matching in _user_vec etc.
        tags_data = {t["id"]: t["embedding"] for t in tags_raw if t.get("embedding")}

    u_idx = {uid: i for i, uid in enumerate(user_ids)}
    e_idx = {eid: i for i, eid in enumerate(event_ids)}
    s_idx = {sid: i for i, sid in enumerate(space_ids)}
    t_idx = {tid: i for i, tid in enumerate(tag_ids)}

    # ── Tag Weights Calculation (Dynamic Profile) ──────────────────────────────
    # Base weight 1.0 for declared tags, +0.5 for events, +0.3 for spaces
    user_weights: dict[str, dict[str, float]] = {}
    for u in users_raw:
        user_weights[u["id"]] = {t: 1.0 for t in u.get("tags", [])}

    event_map = {e["id"]: e for e in events_raw}
    for att in attendees_raw:
        uid, eid = att.get("userId"), att.get("eventId")
        if uid in user_weights and eid in event_map:
            for t in event_map[eid].get("tags", []):
                user_weights[uid][t] = user_weights[uid].get(t, 0.0) + 0.5

    space_map = {s["id"]: s for s in spaces_raw}
    for mem in members_raw:
        uid, sid = mem.get("userId"), mem.get("spaceId")
        if uid in user_weights and sid in space_map:
            for t in space_map[sid].get("tags", []):
                user_weights[uid][t] = user_weights[uid].get(t, 0.0) + 0.3

    # ── Node feature tensors ──────────────────────────────────────────────────
    user_x  = torch.tensor([_user_vec(u, tags_data, user_weights[u["id"]])  for u in users_raw],  dtype=torch.float32)
    event_x = torch.tensor([_event_vec(e, tags_data) for e in events_raw], dtype=torch.float32)
    space_x = torch.tensor([_space_vec(s, tags_data) for s in spaces_raw], dtype=torch.float32)
    
    # Tag nodes: 64D dense embeddings directly from Postgres via tags.json
    from hgt.config import TAG_EMBED_DIM
    if tag_ids and len(tags_data) > 0:
        t_list = []
        for tid in tag_ids:
            t_list.append(tags_data.get(tid, [0.0] * TAG_EMBED_DIM))
        tag_x = torch.tensor(t_list, dtype=torch.float32)
    elif tag_ids:
        # We have tags but no embeddings at all (e.g. from synthetic generator)
        tag_x = torch.zeros((len(tag_ids), TAG_EMBED_DIM), dtype=torch.float32)
    else:
        # Fallback to completely empty if missing tags.json entirely
        tag_x = torch.empty((0, TAG_EMBED_DIM), dtype=torch.float32)

    u_idx = {uid: i for i, uid in enumerate(user_ids)}
    e_idx = {eid: i for i, eid in enumerate(event_ids)}
    s_idx = {sid: i for i, sid in enumerate(space_ids)}
    t_idx = {tid: i for i, tid in enumerate(tag_ids)}

    # ── Universal train / val split logic ─────────────────────────────────────
    # We want to monitor all edge types in _BPR_EDGES.
    
    # Structure: (src_type, src_id, dst_type) -> list[dst_id]
    val_all: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    # Structure: (src_type, src_id, dst_type) -> set[dst_id]
    train_all: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    def _split_and_fill(records, src_type, dst_type_map, src_idx_map, dst_idx_map, 
                         src_list, dst_list, w_list, 
                         is_timestamped=True, default_weight=1.0, allow_1_degree_holdout=False):
        # group by src_id
        by_src = defaultdict(list)
        for r in records:
            sid, did, w, ts = r
            if sid in src_idx_map and did in dst_idx_map:
                by_src[sid].append(r)
        
        for sid, recs in by_src.items():
            if is_timestamped:
                recs.sort(key=lambda x: x[3]) # chronological
            else:
                random.shuffle(recs)
            
            # If a source node only has 1 record for this relation, we usually never hold it out
            # to protect its embedding. However, for 1-to-1 relationships like `event->space`
            # we MUST hold some out (`allow_1_degree_holdout`), or we never evaluate them.
            if len(recs) == 1:
                n_v = 1 if allow_1_degree_holdout and random.random() < val_ratio else 0
            else:
                n_v = max(1, int(len(recs) * val_ratio))
                # Ensure we ALWAYS leave at least 1 record in training
                if len(recs) - n_v < 1:
                    n_v = len(recs) - 1
            
            t_recs = recs[:-n_v] if n_v > 0 else recs
            v_recs = recs[-n_v:] if n_v > 0 else []
            
            # Metadata for dst_type
            d_type = dst_type_map[recs[0][1]] if isinstance(dst_type_map, dict) else dst_type_map

            for src_id, dst_id, w, _ in t_recs:
                src_list.append(src_idx_map[src_id])
                dst_list.append(dst_idx_map[dst_id])
                w_list.append(w)
                train_all[(src_type, src_id, d_type)].add(dst_id)
                
                # If symmetric user relationship, ensure reverse is also in train/seen
                if src_type == d_type and src_type == "user":
                    src_list.append(src_idx_map[dst_id])
                    dst_list.append(src_idx_map[src_id])
                    w_list.append(w)
                    train_all[(src_type, dst_id, src_type)].add(src_id)
            
            for src_id, dst_id, w, _ in v_recs:
                val_all[(src_type, src_id, d_type)].append(dst_id)
                # Add reverse validation for high-granularity tracking
                val_all[(d_type, dst_id, src_type)].append(src_id)

    # 1. User interactions (attends, joins)
    # Mapping table for interaction records
    inter_recs = []
    for r in attendees_raw:
        uid, eid = r.get("userId"), r.get("eventId")
        w, ts = float(r.get("weight", 1.0)) * 2.0, r.get("created_at", "")
        inter_recs.append((uid, eid, w, ts))
    
    attends_src, attends_dst, attends_w = [], [], []
    _split_and_fill(inter_recs, "user", "event", u_idx, e_idx, attends_src, attends_dst, attends_w)

    inter_recs = []
    for r in members_raw:
        uid, sid = r.get("userId"), r.get("spaceId")
        w, ts = float(r.get("weight", 1.0)) * 1.5, r.get("created_at", "")
        inter_recs.append((uid, sid, w, ts))
    
    joins_src, joins_dst, joins_w = [], [], []
    _split_and_fill(inter_recs, "user", "space", u_idx, s_idx, joins_src, joins_dst, joins_w)

    # 2. hosted_by (event → space) (Many-to-1)
    # Every event has exactly 1 space. We MUST allow 1-degree holdouts to test event->space and space->event!
    hosted_recs = []
    for e in events_raw:
        eid, sid = e["id"], e.get("spaceId")
        if sid: hosted_recs.append((eid, sid, 1.0, ""))
    
    hosted_src, hosted_dst, hosted_w = [], [], []
    _split_and_fill(hosted_recs, "event", "space", e_idx, s_idx, hosted_src, hosted_dst, hosted_w, is_timestamped=False, allow_1_degree_holdout=True)

    # 3. similar_to (user → user)
    # (Pre-calculated by _user_user_edges)
    sim_raw_src, sim_raw_dst, sim_raw_w = _user_user_edges(events_raw, spaces_raw, attendees_raw, members_raw, u_idx, e_idx, s_idx, min_coattend, top_k_similar)
    # Convert indexes back to IDs to use our generic splitter
    rev_u_idx = {i: uid for uid, i in u_idx.items()}
    sim_recs = [(rev_u_idx[s], rev_u_idx[d], w, "") for s, d, w in zip(sim_raw_src, sim_raw_dst, sim_raw_w)]
    
    sim_src, sim_dst, sim_w = [], [], []
    _split_and_fill(sim_recs, "user", "user", u_idx, u_idx, sim_src, sim_dst, sim_w, is_timestamped=False)

    # 4. Tags (likes, tagged_with, tagged_with_space)
    ut_recs = []
    for u in users_raw:
        uid = u["id"]
        for t in u.get("tags", []): ut_recs.append((uid, t, 1.0, ""))
    ut_src, ut_dst, ut_w = [], [], []
    _split_and_fill(ut_recs, "user", "tag", u_idx, t_idx, ut_src, ut_dst, ut_w, is_timestamped=False)

    et_recs = []
    for e in events_raw:
        eid = e["id"]
        for t in e.get("tags", []): et_recs.append((eid, t, 1.0, ""))
    et_src, et_dst, et_w = [], [], []
    _split_and_fill(et_recs, "event", "tag", e_idx, t_idx, et_src, et_dst, et_w, is_timestamped=False)

    st_recs = []
    for s in spaces_raw:
        sid = s["id"]
        for t in s.get("tags", []): st_recs.append((sid, t, 1.0, ""))
    st_src, st_dst, st_w = [], [], []
    _split_and_fill(st_recs, "space", "tag", s_idx, t_idx, st_src, st_dst, st_w, is_timestamped=False)

    # 5. Same-type similarity (event-event, space-space, tag-tag) -> VALIDATION ONLY (Discovery)
    # We don't add these to the training graph, but we test if the model can "discover" them.
    ee_raw_src, ee_raw_dst, ee_raw_w = _shared_tag_edges(events_raw, e_idx, top_k=top_k_similar)
    for s, d, w in zip(ee_raw_src, ee_raw_dst, ee_raw_w):
        val_all[("event", event_ids[s], "event")].append(event_ids[d])

    ss_raw_src, ss_raw_dst, ss_raw_w = _shared_tag_edges(spaces_raw, s_idx, top_k=top_k_similar)
    for s, d, w in zip(ss_raw_src, ss_raw_dst, ss_raw_w):
        val_all[("space", space_ids[s], "space")].append(space_ids[d])

    tt_raw_src, tt_raw_dst, tt_raw_w = _tag_tag_edges(tags_raw, t_idx, top_k=top_k_similar)
    for s, d, w in zip(tt_raw_src, tt_raw_dst, tt_raw_w):
        val_all[("tag", tag_ids[s], "tag")].append(tag_ids[d])

    # ── Populate PyG HeteroData ───────────────────────────────────────────────
    data = HeteroData()

    data["user"].x  = user_x
    data["event"].x = event_x
    data["space"].x = space_x
    data["tag"].x   = tag_x

    if attends_src:
        ei = _to_edge_index(attends_src, attends_dst)
        w_tensor = torch.tensor(attends_w, dtype=torch.float32)
        # Normalize attends_w to [0, 1]
        ew = w_tensor / w_tensor.max() if w_tensor.numel() > 0 else w_tensor
        data["user", "attends", "event"].edge_index = ei
        data["user", "attends", "event"].edge_weight = ew
        data["event", "rev_attends", "user"].edge_index = _reverse(ei)
        data["event", "rev_attends", "user"].edge_weight = ew

    if joins_src:
        ei = _to_edge_index(joins_src, joins_dst)
        w_tensor = torch.tensor(joins_w, dtype=torch.float32)
        # Normalize joins_w to [0, 1]
        ew = w_tensor / w_tensor.max() if w_tensor.numel() > 0 else w_tensor
        data["user", "joins", "space"].edge_index = ei
        data["user", "joins", "space"].edge_weight = ew
        data["space", "rev_joins", "user"].edge_index = _reverse(ei)
        data["space", "rev_joins", "user"].edge_weight = ew

    if hosted_src:
        ei = _to_edge_index(hosted_src, hosted_dst)
        ew = torch.ones(ei.size(1), dtype=torch.float32)
        data["event", "hosted_by", "space"].edge_index = ei
        data["event", "hosted_by", "space"].edge_weight = ew
        data["space", "rev_hosted_by", "event"].edge_index = _reverse(ei)
        data["space", "rev_hosted_by", "event"].edge_weight = ew

    if sim_src:
        ei = _to_edge_index(sim_src, sim_dst)
        w_tensor = torch.tensor(sim_w, dtype=torch.float32)
        # Fallback normalization just in case
        ew = w_tensor / w_tensor.max() if w_tensor.numel() > 0 else w_tensor
        data["user", "similar_to", "user"].edge_index = ei
        data["user", "similar_to", "user"].edge_weight = ew

    if ut_src:
        ei = _to_edge_index(ut_src, ut_dst)
        ew = torch.ones(ei.size(1), dtype=torch.float32)
        data["user", "likes",           "tag"].edge_index = ei
        data["user", "likes",           "tag"].edge_weight = ew
        data["tag",  "rev_likes",       "user"].edge_index = _reverse(ei)
        data["tag",  "rev_likes",       "user"].edge_weight = ew

    if et_src:
        ei = _to_edge_index(et_src, et_dst)
        ew = torch.ones(ei.size(1), dtype=torch.float32)
        data["event", "tagged_with",           "tag"].edge_index = ei
        data["event", "tagged_with",           "tag"].edge_weight = ew
        data["tag",   "rev_tagged_with_event", "event"].edge_index = _reverse(ei)
        data["tag",   "rev_tagged_with_event", "event"].edge_weight = ew

    if st_src:
        ei = _to_edge_index(st_src, st_dst)
        ew = torch.ones(ei.size(1), dtype=torch.float32)
        data["space", "tagged_with_space",     "tag"].edge_index = ei
        data["space", "tagged_with_space",     "tag"].edge_weight = ew
        data["tag",   "rev_tagged_with_space", "space"].edge_index = _reverse(ei)
        data["tag",   "rev_tagged_with_space", "space"].edge_weight = ew

    # ── Validation data ────────────────────────────────────────────────────────
    # Convert our grouped dicts into the list format train.py expects
    val_pairs_list = []
    for (atype, aid, itype), ids in val_all.items():
        for iid in ids:
            val_pairs_list.append((atype, aid, itype, iid))

    # train_all is (atype, aid, itype) -> set(iid)
    # We need to flatten it for evaluate() to { (atype, aid): set(iid) }
    seen_train_flattened = defaultdict(set)
    for (atype, aid, itype), ids in train_all.items():
        seen_train_flattened[(atype, aid)] |= ids

    val_data = _build_val_data(
        users_raw, events_raw, spaces_raw, 
        seen_train_flattened, val_pairs_list, tags_data, user_weights
    )

    node_ids = {
        "user": user_ids, 
        "event": event_ids, 
        "space": space_ids,
        "tag": tag_ids
    }

    print(
        f"Graph: {len(user_ids)} users, {len(event_ids)} events, "
        f"{len(space_ids)} spaces, {len(tags_data)} tags | "
        f"attends={len(attends_src)} joins={len(joins_src)} "
        f"hosted_by={len(hosted_src)} similar_to={len(sim_src)} "
        f"likes={len(ut_src)} tagged={len(et_src)+len(st_src)} similarity={len(sim_src)}"
    )

    return {"train_data": data, "val_data": val_data, "node_ids": node_ids}


# ── Validation helper ──────────────────────────────────────────────────────────

def _build_val_data(
    users_raw: list[dict],
    events_raw: list[dict],
    spaces_raw: list[dict],
    seen_train: dict[tuple[str, str], set[str]],
    val_pairs: list[tuple[str, str, str, str]],
    tags_data: dict[str, list[float]],
    user_weights: dict[str, dict[str, float]],
) -> dict:
    """
    Returns a dict usable by evaluate_recall_ndcg in train.py:
      anchor_features       — {(atype, aid): feature_vec}
      item_features         — {iid: (itype, vec)}
      val_pairs             — [(atype, aid, itype, iid), ...]
      seen_train_by_anchor  — {(atype, aid): set(iid)}
    """
    anchor_features: dict[tuple, list[float]] = {}
    
    user_by_id = {u["id"]: u for u in users_raw}
    event_by_id = {e["id"]: e for e in events_raw}
    space_by_id = {s["id"]: s for s in spaces_raw}

    # Extract unique anchors from val pairs
    all_anchors = {(atype, aid) for atype, aid, itype, iid in val_pairs}

    for atype, aid in all_anchors:
        if atype == "user":
            u = user_by_id.get(aid)
            if u: anchor_features[(atype, aid)] = _user_vec(u, tags_data, user_weights[aid])
        elif atype == "event":
            e = event_by_id.get(aid)
            if e: anchor_features[(atype, aid)] = _event_vec(e, tags_data)
        elif atype == "space":
            s = space_by_id.get(aid)
            if s: anchor_features[(atype, aid)] = _space_vec(s, tags_data)
        elif atype == "tag":
            emb = tags_data.get(aid)
            if emb: anchor_features[(atype, aid)] = emb

    item_features: dict[str, tuple[str, list[float]]] = {}
    for e in events_raw:
        item_features[e["id"]] = ("event", _event_vec(e, tags_data))
    for s in spaces_raw:
        item_features[s["id"]] = ("space", _space_vec(s, tags_data))
    for u in users_raw:
        item_features[u["id"]] = ("user", _user_vec(u, tags_data, user_weights[u["id"]]))
    
    # We don't really have "features" for tags in the same way (they are the targets)
    # but we might need them if they were anchors. For now, they are only targets.

    return {
        "anchor_features":      anchor_features,
        "item_features":        item_features,
        "val_pairs":            val_pairs,
        "seen_train_by_anchor": seen_train,
    }
