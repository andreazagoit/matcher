"""
Batch-embed all entities from the DB using the Multi-Tower architecture.

Reads users / events / spaces from PostgreSQL and passes their feature
vectors through each entity's dedicated MLP.

Usage:
  npm run mt:sync
"""

from __future__ import annotations
import argparse
import sys
import time
from datetime import date

import psycopg2
import psycopg2.extras
import torch

from mt.config import DATABASE_URL, MODEL_WEIGHTS_PATH, NUM_TAGS, TAG_VOCAB
from hgt.features import build_user_features, build_event_features, build_space_features
from mt.model import load_model, device
from hgt.utils import days_until
from hgt.utils import days_until


# ── DB connection ─────────────────────────────────────────────────────────────

def _connect():
    return psycopg2.connect(DATABASE_URL)


# ── DB readers ────────────────────────────────────────────────────────────────

def _read_users(cur) -> tuple[list[str], list[list[float]]]:
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
        GROUP BY u.id, u.birthdate, u.gender, u.relationship_intent,
                 u.smoking, u.drinking, u.activity_level, u.tags
    """)
    rows = cur.fetchall()

    ids, vecs = [], []
    for u in rows:
        ids.append(u[0])
        vecs.append(build_user_features(
            birthdate=u[1],
            tags=list(u[7]) if u[7] else [],
            gender=u[2],
            relationship_intent=list(u[3]) if u[3] else [],
            smoking=u[4],
            drinking=u[5],
            activity_level=u[6],
            interaction_count=int(u[8] or 0),
        ))
    return ids, vecs


def _read_events(cur) -> tuple[list[str], list[list[float]]]:
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
    """)
    ids, vecs = [], []
    for e in cur.fetchall():
        starts_at      = e[2]
        is_completed   = bool(starts_at and starts_at.date() < date.today())
        attendee_count = int(e[6] if is_completed else e[8]) or 0
        avg_age        = (float(e[7]) if is_completed and e[7] else (float(e[9]) if e[9] else None))
        
        ids.append(e[0])
        vecs.append(build_event_features(
            tags=list(e[1]) if e[1] else [],
            starts_at=starts_at.isoformat() if starts_at else None,
            avg_attendee_age=avg_age,
            attendee_count=attendee_count,
            days_until_event=days_until(starts_at),
            max_attendees=int(e[3]) if e[3] else None,
            is_paid=bool(e[4] and int(e[4]) > 0),
            price_cents=int(e[4]) if e[4] else 0,
        ))
    return ids, vecs


def _read_spaces(cur) -> tuple[list[str], list[list[float]]]:
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
    """)
    ids, vecs = [], []
    for s in cur.fetchall():
        ids.append(s[0])
        vecs.append(build_space_features(
            tags=list(s[1]) if s[1] else [],
            avg_member_age=float(s[3]) if s[3] else None,
            member_count=int(s[2] or 0),
            event_count=int(s[4] or 0),
        ))
    return ids, vecs


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed all DB entities using the trained Multi-Tower model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=MODEL_WEIGHTS_PATH)
    args = parser.parse_args()

    model = load_model(args.model_path)
    if model is None:
        print(
            f"Error: no trained model at {args.model_path}.\n"
            "Run 'npm run mt:train' first.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Model loaded from {args.model_path}")

    t0 = time.time()
    print("Reading node features from DB …", flush=True)
    conn = _connect()
    try:
        cur = conn.cursor()
        user_ids,  user_vecs  = _read_users(cur)
        event_ids, event_vecs = _read_events(cur)
        space_ids, space_vecs = _read_spaces(cur)
    finally:
        conn.close()

    print(
        f"  {len(user_ids):>6,} users  {len(event_ids):>6,} events  "
        f"{len(space_ids):>6,} spaces "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )

    print("Running Multi-Tower independent forward passes …", flush=True)
    t1 = time.time()
    model.eval()

    # Pre-build tensors
    x_dict = {
        "user": torch.tensor(user_vecs, dtype=torch.float32).to(device) if user_vecs else torch.empty((0, 0), device=device),
        "event": torch.tensor(event_vecs, dtype=torch.float32).to(device) if event_vecs else torch.empty((0, 0), device=device),
        "space": torch.tensor(space_vecs, dtype=torch.float32).to(device) if space_vecs else torch.empty((0, 0), device=device),
        "tag": torch.eye(NUM_TAGS, dtype=torch.float32).to(device),
    }

    with torch.no_grad():
        emb = model.forward_all(x_dict)
    print(f"  Done in {time.time()-t1:.1f}s", flush=True)

    print("Upserting embeddings to DB …", flush=True)
    records: list[tuple[str, str, list[float]]] = []

    for ids, node_type in [(user_ids, "user"), (event_ids, "event"), (space_ids, "space")]:
        if len(ids) == 0:
            continue
        emb_mat = emb[node_type].cpu()
        for i, eid in enumerate(ids):
            records.append((eid, node_type, emb_mat[i].tolist()))

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
