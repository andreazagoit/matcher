#!/usr/bin/env python3
"""
Batch embed all entities and write directly to the DB.

Reads users/events/spaces from PostgreSQL, generates embeddings
with the trained model, and upserts them into the `embeddings` table.

No intermediate files — single step from DB to DB.

Usage:
  npm run ml:embed-all
  python embed_all.py
  python embed_all.py --concurrency 256
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import date, datetime
from typing import Optional

import psycopg2
import psycopg2.extras

from config import DATABASE_URL, MODEL_WEIGHTS_PATH, TRAINING_DATA_DIR
from features import build_user_features, build_event_features, build_space_features
from model import load_model, encode_all


# ─── DB helpers ────────────────────────────────────────────────────────────────

def _connect():
    return psycopg2.connect(DATABASE_URL)


def _days_until(starts_at) -> Optional[int]:
    if not starts_at:
        return None
    try:
        dt = starts_at if isinstance(starts_at, datetime) else datetime.fromisoformat(str(starts_at))
        return (dt.date() - date.today()).days
    except (ValueError, TypeError):
        return None


# ─── DB readers ────────────────────────────────────────────────────────────────

def read_users(cur) -> dict[str, tuple[str, list[float]]]:
    cur.execute("""
        SELECT
            u.id::text,
            u.birthdate::text,
            u.gender,
            u.relationship_intent,
            u.smoking,
            u.drinking,
            u.activity_level,
            (COUNT(DISTINCT ea.event_id) + COUNT(DISTINCT m.space_id))::int AS interaction_count
        FROM users u
        LEFT JOIN event_attendees ea
               ON ea.user_id = u.id AND ea.status = 'attended'
        LEFT JOIN members m
               ON m.user_id = u.id AND m.status = 'active'
        GROUP BY u.id, u.birthdate, u.gender, u.relationship_intent,
                 u.smoking, u.drinking, u.activity_level
    """)
    users = cur.fetchall()

    cur.execute("SELECT user_id::text, tag, weight::float FROM user_interests")
    tags_by_user: dict[str, dict[str, float]] = {}
    for row in cur.fetchall():
        tags_by_user.setdefault(row[0], {})[row[1]] = float(row[2])

    result: dict[str, tuple[str, list[float]]] = {}
    for u in users:
        uid = u[0]
        result[uid] = (
            "user",
            build_user_features(
                birthdate=u[1],
                tag_weights=tags_by_user.get(uid, {}),
                gender=u[2],
                relationship_intent=list(u[3]) if u[3] else [],
                smoking=u[4],
                drinking=u[5],
                activity_level=u[6],
                interaction_count=int(u[7] or 0),
            ),
        )
    return result


def read_events(cur) -> dict[str, tuple[str, list[float]]]:
    cur.execute("""
        SELECT
            e.id::text,
            e.tags,
            e.starts_at,
            e.max_attendees,
            e.price,
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
        GROUP BY e.id, e.tags, e.starts_at, e.max_attendees, e.price
    """)
    result: dict[str, tuple[str, list[float]]] = {}
    for e in cur.fetchall():
        starts_at = e[2]
        is_completed = bool(starts_at and starts_at.date() < date.today())
        attendee_count = int(e[5] if is_completed else e[7]) or 0
        avg_age = float(e[6]) if is_completed and e[6] else (float(e[8]) if e[8] else None)
        result[e[0]] = (
            "event",
            build_event_features(
                tags=list(e[1]) if e[1] else [],
                starts_at=e[2].isoformat() if e[2] else None,
                avg_attendee_age=avg_age,
                attendee_count=attendee_count,
                days_until_event=_days_until(e[2]),
                max_attendees=int(e[3]) if e[3] else None,
                is_paid=bool(e[4] and int(e[4]) > 0),
                price_cents=int(e[4]) if e[4] else 0,
            ),
        )
    return result


def read_spaces(cur) -> dict[str, tuple[str, list[float]]]:
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
        LEFT JOIN events e  ON e.space_id = s.id
                  AND e.starts_at IS NOT NULL
        WHERE s.is_active = true
        GROUP BY s.id, s.tags
    """)
    result: dict[str, tuple[str, list[float]]] = {}
    for s in cur.fetchall():
        result[s[0]] = (
            "space",
            build_space_features(
                tags=list(s[1]) if s[1] else [],
                avg_member_age=float(s[3]) if s[3] else None,
                member_count=int(s[2] or 0),
                event_count=int(s[4] or 0),
            ),
        )
    return result


# ─── DB writer ─────────────────────────────────────────────────────────────────

def upsert_embeddings(cur, records: list[tuple[str, str, list[float]]]) -> None:
    """
    Upsert (entity_id, entity_type, embedding) rows into the embeddings table.
    embedding is passed as a pgvector literal string '[f1,f2,...]'.
    """
    psycopg2.extras.execute_batch(
        cur,
        """
        INSERT INTO embeddings (entity_id, entity_type, embedding, updated_at)
        VALUES (%s::uuid, %s, %s::vector, NOW())
        ON CONFLICT (entity_id, entity_type)
        DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = NOW()
        """,
        [
            (eid, etype, "[" + ",".join(f"{v:.6f}" for v in emb) + "]")
            for eid, etype, emb in records
        ],
        page_size=500,
    )


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read all entities from DB, generate embeddings, write back to DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=MODEL_WEIGHTS_PATH)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Encoding batch size")
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    model = load_model(args.model_path)
    if model is None:
        print(
            f"Error: no trained model at {args.model_path}.\n"
            "Run 'npm run ml:train' first.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Model loaded from {args.model_path}")

    # ── Read from DB ───────────────────────────────────────────────────────────
    print("\nReading entities from DB...", flush=True)
    t0 = time.time()
    conn = _connect()
    try:
        cur = conn.cursor()

        users  = read_users(cur)
        events = read_events(cur)
        spaces = read_spaces(cur)

        all_features: dict[str, tuple[str, list[float]]] = {}
        for _src_name, _src in [("users", users), ("events", events), ("spaces", spaces)]:
            for _eid, _val in _src.items():
                if _eid in all_features:
                    print(
                        f"Warning: ID collision '{_eid}' "
                        f"({all_features[_eid][0]} vs {_val[0]}) — skipping {_src_name} entry.",
                        file=sys.stderr,
                    )
                    continue
                all_features[_eid] = _val
        print(
            f"  {len(users):>6,} users  "
            f"{len(events):>6,} events  "
            f"{len(spaces):>6,} spaces  "
            f"→  {len(all_features):,} total  ({time.time()-t0:.1f}s)",
            flush=True,
        )

        # ── Encode in batch ────────────────────────────────────────────────────
        print(f"\nEncoding (batch_size={args.batch_size})...", flush=True)
        t1 = time.time()
        embeddings = encode_all(model, all_features, batch_size=args.batch_size)
        print(f"  Done in {time.time()-t1:.1f}s", flush=True)

        # ── Upsert to DB ───────────────────────────────────────────────────────
        print("\nUpserting to DB...", flush=True)
        t2 = time.time()
        records = [
            (eid, all_features[eid][0], emb.cpu().tolist())
            for eid, emb in embeddings.items()
        ]
        upsert_embeddings(cur, records)
        conn.commit()
        cur.close()
    finally:
        conn.close()

    elapsed = time.time() - t2
    print(f"  Upserted {len(records):,} embeddings in {elapsed:.1f}s", flush=True)
    print(f"\n✓ Done. Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
