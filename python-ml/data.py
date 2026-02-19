"""
Database access and training data loading.

Reads interactions from PostgreSQL and builds (anchor, positive, negative) triplets
for contrastive training.
"""

from __future__ import annotations
import random
from typing import Optional
import psycopg2
import psycopg2.extras
from config import DATABASE_URL
from features import (
    build_user_features,
    build_event_features,
    build_space_features,
)


# ─── Connection ────────────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


# ─── Raw entity loaders ────────────────────────────────────────────────────────

def load_all_users(conn) -> dict[str, list[float]]:
    """Returns {user_id: feature_vector} for all users."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT u.id, u.birthdate,
                   COUNT(DISTINCT ea.event_id) + COUNT(DISTINCT m.space_id) AS interaction_count
            FROM users u
            LEFT JOIN event_attendees ea ON ea.user_id = u.id AND ea.status IN ('going', 'attended')
            LEFT JOIN members m ON m.user_id = u.id AND m.status = 'active'
            GROUP BY u.id, u.birthdate
        """)
        users = cur.fetchall()

        cur.execute("SELECT user_id, tag, weight FROM user_interests")
        interest_rows = cur.fetchall()

    interests_by_user: dict[str, dict[str, float]] = {}
    for row in interest_rows:
        uid = str(row["user_id"])
        interests_by_user.setdefault(uid, {})[row["tag"]] = float(row["weight"])

    result = {}
    for u in users:
        uid = str(u["id"])
        result[uid] = build_user_features(
            birthdate=u["birthdate"],
            tag_weights=interests_by_user.get(uid, {}),
            interaction_count=int(u["interaction_count"] or 0),
        )
    return result


def load_all_events(conn) -> dict[str, list[float]]:
    """Returns {event_id: feature_vector} for all published events."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT e.id, e.tags,
                   AVG(date_part('year', age(u.birthdate))) AS avg_age,
                   COUNT(ea.user_id) AS attendee_count
            FROM events e
            LEFT JOIN event_attendees ea ON ea.event_id = e.id AND ea.status IN ('going', 'attended')
            LEFT JOIN users u ON u.id = ea.user_id AND u.birthdate IS NOT NULL
            WHERE e.status IN ('published', 'completed')
            GROUP BY e.id, e.tags
        """)
        rows = cur.fetchall()

    result = {}
    for r in rows:
        result[str(r["id"])] = build_event_features(
            tags=r["tags"] or [],
            avg_attendee_age=float(r["avg_age"]) if r["avg_age"] else None,
            attendee_count=int(r["attendee_count"] or 0),
        )
    return result


def load_all_spaces(conn) -> dict[str, list[float]]:
    """Returns {space_id: feature_vector} for all active spaces."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.id, s.tags,
                   AVG(date_part('year', age(u.birthdate))) AS avg_age,
                   COUNT(m.user_id) AS member_count
            FROM spaces s
            LEFT JOIN members m ON m.space_id = s.id AND m.status = 'active'
            LEFT JOIN users u ON u.id = m.user_id AND u.birthdate IS NOT NULL
            WHERE s.is_active = true
            GROUP BY s.id, s.tags
        """)
        rows = cur.fetchall()

    result = {}
    for r in rows:
        result[str(r["id"])] = build_space_features(
            tags=r["tags"] or [],
            avg_member_age=float(r["avg_age"]) if r["avg_age"] else None,
            member_count=int(r["member_count"] or 0),
        )
    return result


# ─── Interactions loader ───────────────────────────────────────────────────────

def load_positive_interactions(conn) -> list[tuple[str, str, str]]:
    """
    Returns list of (user_id, item_id, item_type) for all positive interactions.
    Sources:
      - event_attendees (going/attended)     → (user, event)
      - members (active)                     → (user, space)
      - conversations (active)               → (user, other_user) bidirectional
      - impressions (clicked/joined/messaged) → (user, item)
    """
    interactions: list[tuple[str, str, str]] = []

    with conn.cursor() as cur:
        cur.execute("""
            SELECT user_id::text, event_id::text
            FROM event_attendees
            WHERE status IN ('going', 'attended')
        """)
        for r in cur.fetchall():
            interactions.append((r["user_id"], r["event_id"], "event"))

        cur.execute("""
            SELECT user_id::text, space_id::text
            FROM members
            WHERE status = 'active'
        """)
        for r in cur.fetchall():
            interactions.append((r["user_id"], r["space_id"], "space"))

        cur.execute("""
            SELECT initiator_id::text, recipient_id::text
            FROM conversations
            WHERE status = 'active'
        """)
        for r in cur.fetchall():
            interactions.append((r["initiator_id"], r["recipient_id"], "user"))
            interactions.append((r["recipient_id"], r["initiator_id"], "user"))

        # impressions table (may not exist yet - graceful fallback)
        try:
            cur.execute("""
                SELECT user_id::text, item_id::text, item_type
                FROM impressions
                WHERE action IN ('clicked', 'joined', 'messaged')
            """)
            for r in cur.fetchall():
                interactions.append((r["user_id"], r["item_id"], r["item_type"]))
        except Exception:
            pass

    return interactions


# ─── Training dataset builder ──────────────────────────────────────────────────

def build_training_triplets(
    negative_samples: int = 5,
) -> list[tuple[list[float], list[float], int]]:
    """
    Returns list of (anchor_features, item_features, label) where label is 1 or 0.
    Each positive interaction generates `negative_samples` negatives via random sampling.
    """
    conn = get_connection()
    try:
        users = load_all_users(conn)
        events = load_all_events(conn)
        spaces = load_all_spaces(conn)
        interactions = load_positive_interactions(conn)
    finally:
        conn.close()

    all_items: dict[str, tuple[str, list[float]]] = {}
    for eid, fvec in events.items():
        all_items[eid] = ("event", fvec)
    for sid, fvec in spaces.items():
        all_items[sid] = ("space", fvec)
    for uid, fvec in users.items():
        all_items[uid] = ("user", fvec)

    all_item_ids = list(all_items.keys())

    # Build set of positive pairs for fast lookup
    positive_set: set[tuple[str, str]] = {(uid, iid) for uid, iid, _ in interactions}

    triplets: list[tuple[list[float], list[float], int]] = []

    for user_id, item_id, item_type in interactions:
        user_vec = users.get(user_id)
        if user_vec is None:
            continue

        if item_type == "event":
            item_vec = events.get(item_id)
        elif item_type == "space":
            item_vec = spaces.get(item_id)
        elif item_type == "user":
            item_vec = users.get(item_id)
        else:
            item_vec = None

        if item_vec is None:
            continue

        # Positive pair
        triplets.append((user_vec, item_vec, 1))

        # Negative samples
        sampled = 0
        attempts = 0
        while sampled < negative_samples and attempts < negative_samples * 10:
            attempts += 1
            neg_id = random.choice(all_item_ids)
            if (user_id, neg_id) in positive_set or neg_id == user_id:
                continue
            _, neg_vec = all_items[neg_id]
            triplets.append((user_vec, neg_vec, 0))
            sampled += 1

    random.shuffle(triplets)
    return triplets


# ─── Entity feature loader (for inference) ─────────────────────────────────────

def load_entity_features(entity_id: str, entity_type: str) -> Optional[list]:
    """Load feature vector for a single entity from DB (used at inference time)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if entity_type == "user":
                cur.execute("SELECT birthdate FROM users WHERE id = %s", (entity_id,))
                row = cur.fetchone()
                if not row:
                    return None
                cur.execute(
                    "SELECT tag, weight FROM user_interests WHERE user_id = %s",
                    (entity_id,),
                )
                interests = {r["tag"]: float(r["weight"]) for r in cur.fetchall()}
                cur.execute("""
                    SELECT COUNT(*)::int AS cnt
                    FROM event_attendees WHERE user_id = %s AND status IN ('going','attended')
                """, (entity_id,))
                interaction_count = cur.fetchone()["cnt"] or 0
                return build_user_features(row["birthdate"], interests, interaction_count)

            elif entity_type == "event":
                cur.execute("SELECT tags FROM events WHERE id = %s", (entity_id,))
                row = cur.fetchone()
                if not row:
                    return None
                cur.execute("""
                    SELECT AVG(date_part('year', age(u.birthdate))) AS avg_age,
                           COUNT(ea.user_id)::int AS cnt
                    FROM event_attendees ea
                    JOIN users u ON u.id = ea.user_id
                    WHERE ea.event_id = %s AND ea.status IN ('going','attended')
                """, (entity_id,))
                stats = cur.fetchone()
                return build_event_features(
                    tags=row["tags"] or [],
                    avg_attendee_age=float(stats["avg_age"]) if stats["avg_age"] else None,
                    attendee_count=stats["cnt"] or 0,
                )

            elif entity_type == "space":
                cur.execute("SELECT tags FROM spaces WHERE id = %s", (entity_id,))
                row = cur.fetchone()
                if not row:
                    return None
                cur.execute("""
                    SELECT AVG(date_part('year', age(u.birthdate))) AS avg_age,
                           COUNT(m.user_id)::int AS cnt
                    FROM members m
                    JOIN users u ON u.id = m.user_id
                    WHERE m.space_id = %s AND m.status = 'active'
                """, (entity_id,))
                stats = cur.fetchone()
                return build_space_features(
                    tags=row["tags"] or [],
                    avg_member_age=float(stats["avg_age"]) if stats["avg_age"] else None,
                    member_count=stats["cnt"] or 0,
                )
    finally:
        conn.close()
    return None


def load_all_entity_features(conn) -> dict[str, tuple[str, list[float]]]:
    """Load all entities as {entity_id: (entity_type, features)} for bulk similarity search."""
    users = load_all_users(conn)
    events = load_all_events(conn)
    spaces = load_all_spaces(conn)

    result: dict[str, tuple[str, list[float]]] = {}
    for uid, fvec in users.items():
        result[uid] = ("user", fvec)
    for eid, fvec in events.items():
        result[eid] = ("event", fvec)
    for sid, fvec in spaces.items():
        result[sid] = ("space", fvec)
    return result
