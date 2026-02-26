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

from hgt.config import TRAINING_DATA_DIR, NUM_TAGS, TAG_TO_IDX
from hgt.features import build_user_features, build_event_features, build_space_features
from hgt.utils import days_until


# ── JSON loaders ──────────────────────────────────────────────────────────────

def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Feature builders (JSON dict → float list) ─────────────────────────────────

def _user_vec(u: dict) -> list[float]:
    return build_user_features(
        birthdate=u.get("birthdate"),
        tags=u.get("tags") or [],
        gender=u.get("gender"),
        relationship_intent=u.get("relationshipIntent") or [],
        smoking=u.get("smoking"),
        drinking=u.get("drinking"),
        activity_level=u.get("activityLevel"),
    )


def _event_vec(e: dict) -> list[float]:
    starts_at = e.get("startsAt")
    return build_event_features(
        tags=e.get("tags") or [],
        starts_at=starts_at,
        avg_attendee_age=e.get("avgAttendeeAge"),
        attendee_count=int(e.get("attendeeCount") or 0),
        days_until_event=days_until(starts_at),
        max_attendees=e.get("maxAttendees"),
        is_paid=bool(e.get("isPaid")),
        price_cents=int(e.get("priceCents") or 0),
    )


def _space_vec(s: dict) -> list[float]:
    return build_space_features(
        tags=s.get("tags") or [],
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

    src, dst = [], []
    for uid, peers in neighbors.items():
        peers.sort(key=lambda x: x[1], reverse=True)
        for peer_id, _ in peers[:top_k]:
            src.append(u_idx[uid])
            dst.append(u_idx[peer_id])
    return src, dst


# ── Main builder ──────────────────────────────────────────────────────────────

def build_graph_data(
    data_dir: str = TRAINING_DATA_DIR,
    val_ratio: float = 0.15,
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

    u_idx = {uid: i for i, uid in enumerate(user_ids)}
    e_idx = {eid: i for i, eid in enumerate(event_ids)}
    s_idx = {sid: i for i, sid in enumerate(space_ids)}

    # ── Node feature tensors ──────────────────────────────────────────────────
    user_x  = torch.tensor([_user_vec(u)  for u in users_raw],  dtype=torch.float32)
    event_x = torch.tensor([_event_vec(e) for e in events_raw], dtype=torch.float32)
    space_x = torch.tensor([_space_vec(s) for s in spaces_raw], dtype=torch.float32)
    # Tag nodes: identity matrix — each tag is a unique one-hot vector
    tag_x   = torch.eye(NUM_TAGS, dtype=torch.float32)

    # ── Temporal train / val split on interactions ────────────────────────────
    by_user: dict[str, list] = defaultdict(list)
    
    for r in attendees_raw:
        uid = r.get("userId")
        eid = r.get("eventId")
        w = float(r.get("weight", 1.0))
        ts = r.get("created_at", "")
        if uid in u_idx and eid in e_idx:
            by_user[uid].append((eid, "event", w, ts))

    for r in members_raw:
        uid = r.get("userId")
        sid = r.get("spaceId")
        w = float(r.get("weight", 1.0))
        ts = r.get("created_at", "")
        if uid in u_idx and sid in s_idx:
            by_user[uid].append((sid, "space", w, ts))

    attends_src, attends_dst, attends_w = [], [], []
    joins_src,   joins_dst,   joins_w   = [], [], []
    val_pos:   dict[str, dict[str, list[str]]] = {}
    train_pos: dict[str, dict[str, set[str]]]  = {}

    for uid, records in by_user.items():
        records.sort(key=lambda r: r[3])           # chronological
        n_val = max(1, int(len(records) * val_ratio))
        train_recs = records[:-n_val]
        val_recs   = records[-n_val:]

        train_pos.setdefault(uid, {"event": set(), "space": set()})
        for iid, itype, w, _ in train_recs:
            if itype == "event":
                attends_src.append(u_idx[uid])
                attends_dst.append(e_idx[iid])
                attends_w.append(w)
            else:
                joins_src.append(u_idx[uid])
                joins_dst.append(s_idx[iid])
                joins_w.append(w)
            train_pos[uid][itype].add(iid)

        for iid, itype, w, _ in val_recs:
            val_pos.setdefault(uid, {"event": [], "space": []})
            val_pos[uid][itype].append(iid)

    # ── hosted_by edges (event → space) ───────────────────────────────────────
    hosted_src, hosted_dst = [], []
    for e in events_raw:
        sid = e.get("spaceId")
        if e["id"] in e_idx and sid and sid in s_idx:
            hosted_src.append(e_idx[e["id"]])
            hosted_dst.append(s_idx[sid])

    # ── user similar_to user (co-attendance graph) ────────────────────────────
    sim_src, sim_dst = _user_user_edges(events_raw, spaces_raw, attendees_raw, members_raw, u_idx, e_idx, s_idx, min_coattend, top_k_similar)

    # ── Tag edges ─────────────────────────────────────────────────────────────
    # user → tag (binary: 1.0 per declared tag)
    ut_src, ut_dst = [], []
    for u in users_raw:
        uid = u["id"]
        if uid not in u_idx:
            continue
        for tag in (u.get("tags") or []):
            t = TAG_TO_IDX.get(tag)
            if t is not None:
                ut_src.append(u_idx[uid]); ut_dst.append(t)

    # event → tag
    et_src, et_dst = [], []
    for e in events_raw:
        eid = e["id"]
        if eid not in e_idx:
            continue
        for tag in (e.get("tags") or []):
            t = TAG_TO_IDX.get(tag)
            if t is not None:
                et_src.append(e_idx[eid]); et_dst.append(t)

    # space → tag
    st_src, st_dst = [], []
    for s in spaces_raw:
        sid = s["id"]
        if sid not in s_idx:
            continue
        for tag in (s.get("tags") or []):
            t = TAG_TO_IDX.get(tag)
            if t is not None:
                st_src.append(s_idx[sid]); st_dst.append(t)

    # ── Assemble HeteroData ───────────────────────────────────────────────────
    data = HeteroData()
    data["user"].x  = user_x
    data["event"].x = event_x
    data["space"].x = space_x
    data["tag"].x   = tag_x

    if attends_src:
        ei = _to_edge_index(attends_src, attends_dst)
        data["user",  "attends",     "event"].edge_index  = ei
        data["user",  "attends",     "event"].edge_weight = torch.tensor(attends_w, dtype=torch.float32)
        data["event", "rev_attends", "user"].edge_index   = _reverse(ei)

    if joins_src:
        ei = _to_edge_index(joins_src, joins_dst)
        data["user",  "joins",    "space"].edge_index  = ei
        data["user",  "joins",    "space"].edge_weight = torch.tensor(joins_w, dtype=torch.float32)
        data["space", "rev_joins","user"].edge_index   = _reverse(ei)

    if hosted_src:
        ei = _to_edge_index(hosted_src, hosted_dst)
        data["event", "hosted_by",     "space"].edge_index = ei
        data["space", "rev_hosted_by", "event"].edge_index = _reverse(ei)

    if sim_src:
        data["user", "similar_to", "user"].edge_index = _to_edge_index(sim_src, sim_dst)

    if ut_src:
        ei = _to_edge_index(ut_src, ut_dst)
        data["user", "likes",     "tag"].edge_index = ei
        data["tag",  "rev_likes", "user"].edge_index = _reverse(ei)

    if et_src:
        ei = _to_edge_index(et_src, et_dst)
        data["event", "tagged_with",           "tag"].edge_index = ei
        data["tag",   "rev_tagged_with_event", "event"].edge_index = _reverse(ei)

    if st_src:
        ei = _to_edge_index(st_src, st_dst)
        data["space", "tagged_with_space",     "tag"].edge_index = ei
        data["tag",   "rev_tagged_with_space", "space"].edge_index = _reverse(ei)

    # ── Validation data ────────────────────────────────────────────────────────
    val_data = _build_val_data(
        val_pos, train_pos,
        users_raw, events_raw, spaces_raw,
    )

    node_ids = {"user": user_ids, "event": event_ids, "space": space_ids}

    print(
        f"Graph: {len(user_ids)} users, {len(event_ids)} events, "
        f"{len(space_ids)} spaces, {NUM_TAGS} tags | "
        f"attends={len(attends_src)} joins={len(joins_src)} "
        f"hosted_by={len(hosted_src)} similar_to={len(sim_src)} "
        f"likes={len(ut_src)} tagged_with={len(et_src)+len(st_src)}"
    )

    return {"train_data": data, "val_data": val_data, "node_ids": node_ids}


# ── Validation helper ──────────────────────────────────────────────────────────

def _build_val_data(
    val_pos:   dict[str, dict[str, list[str]]],
    train_pos: dict[str, dict[str, set[str]]],
    users_raw:  list[dict],
    events_raw: list[dict],
    spaces_raw: list[dict],
) -> dict:
    """
    Returns a dict usable by evaluate_recall_ndcg in train.py:
      anchor_features       — {(atype, aid): feature_vec}
      item_features         — {iid: (itype, vec)}
      val_pairs             — [(atype, aid, itype, iid), ...]
      seen_train_by_anchor  — {(atype, aid): set(iid)}
    """
    anchor_features: dict[tuple, list[float]] = {}
    val_pairs: list[tuple] = []
    seen_train: dict[tuple, set[str]] = {}

    user_by_id = {u["id"]: u for u in users_raw}

    for uid, by_type in val_pos.items():
        if not any(by_type.values()):
            continue
        u = user_by_id.get(uid)
        if not u:
            continue
        key = ("user", uid)
        anchor_features[key] = _user_vec(u)
        seen_train[key] = set()
        for itype, ids in (train_pos.get(uid) or {}).items():
            seen_train[key] |= ids
        for itype, iids in by_type.items():
            for iid in iids:
                val_pairs.append(("user", uid, itype, iid))

    item_features: dict[str, tuple[str, list[float]]] = {}
    for e in events_raw:
        item_features[e["id"]] = ("event", _event_vec(e))
    for s in spaces_raw:
        item_features[s["id"]] = ("space", _space_vec(s))
    for u in users_raw:
        item_features[u["id"]] = ("user", _user_vec(u))

    return {
        "anchor_features":      anchor_features,
        "item_features":        item_features,
        "val_pairs":            val_pairs,
        "seen_train_by_anchor": seen_train,
    }
