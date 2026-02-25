"""
Batch-embed all entities from the DB using the full heterogeneous graph.

Reads users / events / spaces + their interaction edges from PostgreSQL,
builds a HeteroData graph, runs HGTConv forward_graph, and upserts every
embedding back into the `embeddings` table.

Usage:
  python -m ml.sync
  python -m ml.sync --model-path python-ml/hgt_weights.pt
"""

from __future__ import annotations
import argparse
import sys
import time
from datetime import date
from collections import defaultdict

import psycopg2
import psycopg2.extras
import torch
from torch_geometric.data import HeteroData

from hgt.config import DATABASE_URL, MODEL_WEIGHTS_PATH, NUM_TAGS, TAG_TO_IDX, TAG_VOCAB
from hgt.features import build_user_features, build_event_features, build_space_features
from hgt.model import load_model, device
from hgt.utils import days_until


# ── DB connection ─────────────────────────────────────────────────────────────

def _connect():
    return psycopg2.connect(DATABASE_URL)


# ── DB readers ────────────────────────────────────────────────────────────────

def _read_users(cur) -> tuple[list[str], list[list[float]], dict[str, list[str]]]:
    cur.execute("""
        SELECT
            u.id::text,
            u.birthdate::text,
            u.gender,
            u.relationship_intent,
            u.smoking,
            u.drinking,
            u.activity_level,
            u.tags,
            (COUNT(DISTINCT ea.event_id) + COUNT(DISTINCT m.space_id))::int AS interaction_count
        FROM users u
        LEFT JOIN event_attendees ea
               ON ea.user_id = u.id AND ea.status = 'attended'
        LEFT JOIN members m
               ON m.user_id = u.id AND m.status = 'active'
                 u.smoking, u.drinking, u.activity_level, u.tags
        ORDER BY u.id
    """)
    rows = cur.fetchall()

    ids, vecs = [], []
    tags_by_user: dict[str, list[str]] = {}
    for u in rows:
        uid  = u[0]
        tags = list(u[7]) if u[7] else []
        ids.append(uid)
        tags_by_user[uid] = tags
        vecs.append(build_user_features(
            birthdate=u[1],
            tags=tags,
            gender=u[2],
            relationship_intent=list(u[3]) if u[3] else [],
            smoking=u[4],
            drinking=u[5],
            activity_level=u[6],
            interaction_count=int(u[8] or 0),
        ))
    return ids, vecs, tags_by_user


def _read_events(cur) -> tuple[list[str], list[list[float]], list[str], dict[str, list[str]]]:
    cur.execute("""
        SELECT
            e.id::text,
            e.tags,
            e.starts_at,
            e.max_attendees,
            e.price,
            e.space_id::text,
            COUNT(ea_real.user_id)::int                           AS attended_count,
            AVG(date_part('year', age(u_real.birthdate)))::float  AS avg_age_attended,
            COUNT(ea_going.user_id)::int                          AS going_count,
            AVG(date_part('year', age(u_going.birthdate)))::float AS avg_age_going
        FROM events e
        LEFT JOIN event_attendees ea_real
               ON ea_real.event_id = e.id AND ea_real.status = 'attended'
        LEFT JOIN users u_real
               ON u_real.id = ea_real.user_id AND u_real.birthdate IS NOT NULL
        LEFT JOIN event_attendees ea_going
               ON ea_going.event_id = e.id AND ea_going.status = 'going'
        LEFT JOIN users u_going
               ON u_going.id = ea_going.user_id AND u_going.birthdate IS NOT NULL
        GROUP BY e.id, e.tags, e.starts_at, e.max_attendees, e.price, e.space_id
        ORDER BY e.id
    """)
    ids, vecs, space_ids = [], [], []
    tags_by_event: dict[str, list[str]] = {}
    for e in cur.fetchall():
        starts_at      = e[2]
        is_completed   = bool(starts_at and starts_at.date() < date.today())
        attendee_count = int(e[6] if is_completed else e[8]) or 0
        avg_age        = (float(e[7]) if is_completed and e[7] else (float(e[9]) if e[9] else None))
        eid  = e[0]
        etags = list(e[1]) if e[1] else []
        ids.append(eid)
        space_ids.append(e[5])
        tags_by_event[eid] = etags
        vecs.append(build_event_features(
            tags=etags,
            starts_at=starts_at.isoformat() if starts_at else None,
            avg_attendee_age=avg_age,
            attendee_count=attendee_count,
            days_until_event=days_until(starts_at),
            max_attendees=int(e[3]) if e[3] else None,
            is_paid=bool(e[4] and int(e[4]) > 0),
            price_cents=int(e[4]) if e[4] else 0,
        ))
    return ids, vecs, space_ids, tags_by_event


def _read_spaces(cur) -> tuple[list[str], list[list[float]], dict[str, list[str]]]:
    cur.execute("""
        SELECT
            s.id::text,
            s.tags,
            COUNT(DISTINCT m.user_id)::int                  AS member_count,
            AVG(date_part('year', age(u.birthdate)))::float AS avg_member_age,
            COUNT(DISTINCT e.id)::int                       AS event_count
        FROM spaces s
        LEFT JOIN members m ON m.space_id = s.id AND m.status = 'active'
        LEFT JOIN users u   ON u.id = m.user_id AND u.birthdate IS NOT NULL
        LEFT JOIN events e  ON e.space_id = s.id AND e.starts_at IS NOT NULL
        WHERE s.is_active = true
        GROUP BY s.id, s.tags
        ORDER BY s.id
    """)
    ids, vecs = [], []
    tags_by_space: dict[str, list[str]] = {}
    for s in cur.fetchall():
        sid   = s[0]
        stags = list(s[1]) if s[1] else []
        ids.append(sid)
        tags_by_space[sid] = stags
        vecs.append(build_space_features(
            tags=stags,
            avg_member_age=float(s[3]) if s[3] else None,
            member_count=int(s[2] or 0),
            event_count=int(s[4] or 0),
        ))
    return ids, vecs, tags_by_space


def _read_interactions(cur) -> list[tuple[str, str, str, float]]:
    """Returns [(user_id, item_id, item_type, weight), ...] for graph edges."""
    cur.execute("""
        SELECT user_id::text, event_id::text, 'event', 1.0
        FROM event_attendees
        WHERE status IN ('attended', 'going')
        UNION ALL
        SELECT user_id::text, space_id::text, 'space', 0.9
        FROM members
        WHERE status = 'active'
    """)
    return [(r[0], r[1], r[2], float(r[3])) for r in cur.fetchall()]


# ── DB writer ─────────────────────────────────────────────────────────────────

def _upsert_embeddings(cur, records: list[tuple[str, str, list[float]]]) -> None:
    psycopg2.extras.execute_batch(
        cur,
        """
        INSERT INTO embeddings (entity_id, entity_type, embedding, updated_at)
        VALUES (%s::text, %s, %s::vector, NOW())
        ON CONFLICT (entity_id, entity_type)
        DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = NOW()
        """,
        [
            (eid, etype, "[" + ",".join(f"{v:.6f}" for v in emb) + "]")
            for eid, etype, emb in records
        ],
        page_size=500,
    )


# ── Graph builder from DB data ────────────────────────────────────────────────

def _build_db_graph(
    user_ids: list[str],   user_vecs: list[list[float]],  user_tags: dict[str, list[str]],
    event_ids: list[str],  event_vecs: list[list[float]], event_space_ids: list[str],
    event_tags: dict[str, list[str]],
    space_ids: list[str],  space_vecs: list[list[float]], space_tags: dict[str, list[str]],
    interactions: list[tuple[str, str, str, float]],
) -> HeteroData:
    u_idx = {uid: i for i, uid in enumerate(user_ids)}
    e_idx = {eid: i for i, eid in enumerate(event_ids)}
    s_idx = {sid: i for i, sid in enumerate(space_ids)}

    data = HeteroData()
    data["user"].x  = torch.tensor(user_vecs,  dtype=torch.float32)
    data["event"].x = torch.tensor(event_vecs, dtype=torch.float32)
    data["space"].x = torch.tensor(space_vecs, dtype=torch.float32)
    data["tag"].x   = torch.eye(NUM_TAGS, dtype=torch.float32)

    # user–event and user–space interaction edges
    a_src, a_dst, a_w = [], [], []
    j_src, j_dst, j_w = [], [], []
    for uid, iid, itype, w in interactions:
        if uid not in u_idx:
            continue
        if itype == "event" and iid in e_idx:
            a_src.append(u_idx[uid]); a_dst.append(e_idx[iid]); a_w.append(w)
        elif itype == "space" and iid in s_idx:
            j_src.append(u_idx[uid]); j_dst.append(s_idx[iid]); j_w.append(w)

    def _ei(s, d):
        return torch.tensor([s, d], dtype=torch.long)

    if a_src:
        ei = _ei(a_src, a_dst)
        data["user",  "attends",     "event"].edge_index  = ei
        data["user",  "attends",     "event"].edge_weight = torch.tensor(a_w, dtype=torch.float32)
        data["event", "rev_attends", "user"].edge_index   = ei.flip(0)

    if j_src:
        ei = _ei(j_src, j_dst)
        data["user",  "joins",    "space"].edge_index  = ei
        data["user",  "joins",    "space"].edge_weight = torch.tensor(j_w, dtype=torch.float32)
        data["space", "rev_joins","user"].edge_index   = ei.flip(0)

    # event–space hosted_by edges
    h_src, h_dst = [], []
    for eid, sid in zip(event_ids, event_space_ids):
        if sid and sid in s_idx:
            h_src.append(e_idx[eid]); h_dst.append(s_idx[sid])
    if h_src:
        ei = _ei(h_src, h_dst)
        data["event", "hosted_by",     "space"].edge_index = ei
        data["space", "rev_hosted_by", "event"].edge_index = ei.flip(0)

    # tag edges
    ut_src, ut_dst = [], []
    for uid, tags in user_tags.items():
        if uid not in u_idx:
            continue
        for tag in tags:
            t = TAG_TO_IDX.get(tag)
            if t is not None:
                ut_src.append(u_idx[uid]); ut_dst.append(t)
    if ut_src:
        ei = _ei(ut_src, ut_dst)
        data["user", "likes",     "tag"].edge_index = ei
        data["tag",  "rev_likes", "user"].edge_index = ei.flip(0)

    et_src, et_dst = [], []
    for eid, tags in event_tags.items():
        if eid not in e_idx:
            continue
        for tag in tags:
            t = TAG_TO_IDX.get(tag)
            if t is not None:
                et_src.append(e_idx[eid]); et_dst.append(t)
    if et_src:
        ei = _ei(et_src, et_dst)
        data["event", "tagged_with",           "tag"].edge_index = ei
        data["tag",   "rev_tagged_with_event", "event"].edge_index = ei.flip(0)

    st_src, st_dst = [], []
    for sid, tags in space_tags.items():
        if sid not in s_idx:
            continue
        for tag in tags:
            t = TAG_TO_IDX.get(tag)
            if t is not None:
                st_src.append(s_idx[sid]); st_dst.append(t)
    if st_src:
        ei = _ei(st_src, st_dst)
        data["space", "tagged_with_space",     "tag"].edge_index = ei
        data["tag",   "rev_tagged_with_space", "space"].edge_index = ei.flip(0)

    return data


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed all DB entities using the trained HGT graph model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=MODEL_WEIGHTS_PATH)
    args = parser.parse_args()

    model = load_model(args.model_path)
    if model is None:
        print(
            f"Error: no trained model at {args.model_path}.\n"
            "Run 'npm run ml:train' first.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Model loaded from {args.model_path}")

    t0 = time.time()
    print("Reading entities and interactions from DB …", flush=True)
    conn = _connect()
    try:
        cur = conn.cursor()
        user_ids,  user_vecs,  user_tags              = _read_users(cur)
        event_ids, event_vecs, event_space_ids, \
            event_tags                                = _read_events(cur)
        space_ids, space_vecs, space_tags             = _read_spaces(cur)
        interactions                                  = _read_interactions(cur)
    finally:
        conn.close()

    print(
        f"  {len(user_ids):>6,} users  {len(event_ids):>6,} events  "
        f"{len(space_ids):>6,} spaces  {len(interactions):>8,} interactions  "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )

    print("Building graph …", flush=True)
    data = _build_db_graph(
        user_ids, user_vecs, user_tags,
        event_ids, event_vecs, event_space_ids, event_tags,
        space_ids, space_vecs, space_tags,
        interactions,
    )

    print("Running forward_graph …", flush=True)
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        emb = model.forward_graph(
            {t: data[t].x.to(device) for t in data.node_types},
            {et: data[et].edge_index.to(device) for et in data.edge_types},
        )
    print(f"  Done in {time.time()-t1:.1f}s", flush=True)

    print("Upserting embeddings to DB …", flush=True)
    records: list[tuple[str, str, list[float]]] = []

    # User / event / space embeddings (UUID entity_id)
    for ids, node_type in [(user_ids, "user"), (event_ids, "event"), (space_ids, "space")]:
        emb_mat = emb[node_type].cpu()
        for i, eid in enumerate(ids):
            records.append((eid, node_type, emb_mat[i].tolist()))

    # Tag embeddings — entity_id is the tag string (e.g. "music", "travel")
    # These capture the learned semantic position of each tag in the shared space,
    # enabling "recommended tags for user" via cosine similarity.
    tag_emb_mat = emb["tag"].cpu()
    for idx, tag_name in enumerate(TAG_VOCAB):
        records.append((tag_name, "tag", tag_emb_mat[idx].tolist()))

    t2 = time.time()
    conn = _connect()
    try:
        cur = conn.cursor()
        _upsert_embeddings(cur, records)
        conn.commit()
    finally:
        conn.close()

    n_tags = len(TAG_VOCAB)
    print(
        f"  Upserted {len(records):,} embeddings "
        f"({len(records) - n_tags} entities + {n_tags} tags) "
        f"in {time.time()-t2:.1f}s\n"
        f"✓ Done. Total: {time.time()-t0:.1f}s"
    )


if __name__ == "__main__":
    main()
