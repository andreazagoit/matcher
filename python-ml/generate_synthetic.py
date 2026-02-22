#!/usr/bin/env python3
"""
Synthetic training data generator — realistic edition.

Produces JSON files in training-data/ that mirror 'npm run ml:export'.

─── Default volumes ──────────────────────────────────────────────────────────
Change the four constants below to adjust dataset size.
"""

from __future__ import annotations

# ┌─────────────────────────────────────────────────────────────────────────────
# │ CONFIGURE HERE
N_USERS        = 10_000
N_EVENTS       = 1_000
N_SPACES       = 500
N_INTERACTIONS = 100_000   # positive pairs (split ~60% event, 25% space, 15% user)
# └─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import math
import os
import random
import uuid
from collections import defaultdict
from datetime import date, timedelta
from typing import Optional


# ─── Tag vocabulary (must match config.py) ────────────────────────────────────

TAG_VOCAB = [
    "trekking", "camping", "climbing", "cycling", "beach", "mountains", "gardening",
    "cinema", "theater", "live_music", "museums", "reading", "photography", "art",
    "cooking", "restaurants", "wine", "craft_beer", "street_food", "coffee",
    "running", "gym", "yoga", "swimming", "football", "tennis", "padel", "basketball",
    "music", "drawing", "writing", "diy", "gaming", "coding",
    "travel", "volunteering", "languages", "pets", "parties", "board_games",
]

TAG_CLUSTERS: list[list[str]] = [
    ["trekking", "camping", "climbing", "cycling", "beach", "mountains", "gardening"],
    ["cinema", "theater", "live_music", "museums", "reading", "photography", "art"],
    ["cooking", "restaurants", "wine", "craft_beer", "street_food", "coffee"],
    ["running", "gym", "yoga", "swimming", "football", "tennis", "padel", "basketball"],
    ["music", "drawing", "writing", "diy", "gaming", "coding"],
    ["travel", "volunteering", "languages", "pets", "parties", "board_games"],
]

# Fast lookup: tag → cluster index
TAG_TO_CLUSTER: dict[str, int] = {
    tag: ci for ci, cluster in enumerate(TAG_CLUSTERS) for tag in cluster
}

# ─── Personas ─────────────────────────────────────────────────────────────────
#
# Each persona represents an archetypal user type.
# Rules:
#  - age_range:     realistic age bracket for this persona
#  - cluster:       primary tag cluster (60-70% of their tags come from here)
#  - activity/smoking/drinking: persona-specific lifestyle distributions
#  - rel_intent_w:  weights over [serious_relationship, casual_dating, friendship, chat]
#  - gender_w:      weights over [man, woman, non_binary]

PERSONAS = [
    {
        "name":         "outdoor_adventurer",   # cluster 0
        "cluster":      0,
        "age_range":    (22, 32),
        "gender_w":     [0.40, 0.50, 0.10],
        "rel_intent_w": [0.25, 0.40, 0.25, 0.10],  # casual > serious
        "smoking_w":    [0.88, 0.10, 0.02],          # almost never smokes
        "drinking_w":   [0.25, 0.60, 0.15],
        "activity_choices": ["active", "very_active"],
        "activity_w":   [0.35, 0.65],
    },
    {
        "name":         "culture_lover",        # cluster 1
        "cluster":      1,
        "age_range":    (26, 42),
        "gender_w":     [0.35, 0.55, 0.10],
        "rel_intent_w": [0.40, 0.25, 0.25, 0.10],  # serious > casual
        "smoking_w":    [0.55, 0.32, 0.13],
        "drinking_w":   [0.15, 0.52, 0.33],          # drinks more (aperitivo culture)
        "activity_choices": ["light", "moderate"],
        "activity_w":   [0.40, 0.60],
    },
    {
        "name":         "foodie",               # cluster 2
        "cluster":      2,
        "age_range":    (24, 40),
        "gender_w":     [0.40, 0.50, 0.10],
        "rel_intent_w": [0.35, 0.30, 0.25, 0.10],
        "smoking_w":    [0.60, 0.28, 0.12],
        "drinking_w":   [0.08, 0.47, 0.45],          # drinks the most (wine, beer)
        "activity_choices": ["light", "moderate"],
        "activity_w":   [0.50, 0.50],
    },
    {
        "name":         "sports_enthusiast",    # cluster 3
        "cluster":      3,
        "age_range":    (18, 30),
        "gender_w":     [0.62, 0.33, 0.05],
        "rel_intent_w": [0.20, 0.45, 0.25, 0.10],  # casual > serious (young)
        "smoking_w":    [0.93, 0.05, 0.02],          # almost never smokes
        "drinking_w":   [0.32, 0.55, 0.13],
        "activity_choices": ["active", "very_active"],
        "activity_w":   [0.28, 0.72],
    },
    {
        "name":         "creative",             # cluster 4
        "cluster":      4,
        "age_range":    (20, 36),
        "gender_w":     [0.36, 0.46, 0.18],          # most non_binary
        "rel_intent_w": [0.20, 0.35, 0.32, 0.13],
        "smoking_w":    [0.42, 0.38, 0.20],           # smokes the most
        "drinking_w":   [0.15, 0.52, 0.33],
        "activity_choices": ["sedentary", "light", "moderate"],
        "activity_w":   [0.20, 0.48, 0.32],
    },
    {
        "name":         "social_butterfly",     # cluster 5
        "cluster":      5,
        "age_range":    (20, 34),
        "gender_w":     [0.38, 0.52, 0.10],
        "rel_intent_w": [0.18, 0.35, 0.32, 0.15],   # friendship + casual
        "smoking_w":    [0.48, 0.35, 0.17],
        "drinking_w":   [0.05, 0.43, 0.52],           # drinks the most
        "activity_choices": ["moderate", "active"],
        "activity_w":   [0.55, 0.45],
    },
]

REL_INTENTS = ["serious_relationship", "casual_dating", "friendship", "chat"]
GENDERS     = ["man", "woman", "non_binary"]


# ─── Helpers ───────────────────────────────────────────────────────────────────

def uid() -> str:
    return str(uuid.uuid4())

def wc(choices: list, weights: list):
    return random.choices(choices, weights=weights, k=1)[0]

def rand_date(start: date, end: date) -> date:
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta))

def skewed_age(lo: int, hi: int) -> int:
    """
    Beta-distributed age skewed toward the lower third of [lo, hi].
    Models the reality that dating app users peak in their mid-20s.
    """
    raw = random.betavariate(2.0, 3.5)   # peaks ~36% of range
    return lo + round(raw * (hi - lo))

def power_law_popularity(n: int, exponent: float = 1.8) -> list[float]:
    """
    Returns n popularity scores following a power law (Zipf-like).
    A few items are very popular, most are niche.
    """
    ranks  = list(range(1, n + 1))
    random.shuffle(ranks)
    scores = [1.0 / (r ** exponent) for r in ranks]
    max_s  = max(scores)
    return [s / max_s for s in scores]


# ─── Tag sampling ──────────────────────────────────────────────────────────────

def sample_user_tags(persona: dict) -> dict[str, float]:
    """
    Sample tags for a user biased toward their persona's cluster.
    ~65% from primary cluster, ~35% from other clusters.
    Core interests (1-2) get high weights; secondary interests get lower.
    """
    n_total  = random.randint(4, 9)
    n_prim   = max(2, round(n_total * 0.65))
    prim     = TAG_CLUSTERS[persona["cluster"]]
    sec_pool = [t for t in TAG_VOCAB if t not in prim]

    chosen = set(random.sample(prim, min(n_prim, len(prim))))
    n_sec  = n_total - len(chosen)
    if n_sec > 0:
        chosen |= set(random.sample(sec_pool, min(n_sec, len(sec_pool))))

    # 1-2 core interests with high weight, rest secondary
    tags   = list(chosen)
    result = {}
    n_core = min(2, len(tags))
    for i, tag in enumerate(tags):
        if i < n_core:
            result[tag] = round(random.uniform(0.70, 1.00), 2)
        else:
            result[tag] = round(random.uniform(0.25, 0.65), 2)
    return result


def sample_entity_tags(preferred_cluster: int = -1) -> list[str]:
    """
    Sample 1-4 tags for an event or space.
    If preferred_cluster >= 0, bias toward that cluster.
    """
    n = random.randint(1, 4)
    if preferred_cluster >= 0 and random.random() < 0.70:
        clus = TAG_CLUSTERS[preferred_cluster]
    else:
        clus = random.choice(TAG_CLUSTERS)
    tags = set(random.sample(clus, min(max(1, round(n * 0.75)), len(clus))))
    rest = [t for t in TAG_VOCAB if t not in clus]
    if len(tags) < n:
        tags |= set(random.sample(rest, min(n - len(tags), len(rest))))
    return list(tags)


# ─── Entity generators ─────────────────────────────────────────────────────────

def gen_user(persona_idx: int) -> dict:
    p     = PERSONAS[persona_idx]
    lo, hi = p["age_range"]
    today  = date.today()
    age    = skewed_age(lo, hi)
    bd     = today - timedelta(days=round(age * 365.25))

    n_ri  = random.randint(1, 3)
    ri    = random.choices(REL_INTENTS, weights=p["rel_intent_w"], k=n_ri)
    ri    = list(dict.fromkeys(ri))  # deduplicate keeping order

    return {
        "id":                  uid(),
        "persona":             p["name"],          # informational only, not in real export
        "persona_idx":         persona_idx,        # used internally for interaction generation
        "birthdate":           bd.isoformat(),
        "gender":              wc(GENDERS, p["gender_w"]),
        "relationship_intent": ri,
        "smoking":             wc(SMOKING,   p["smoking_w"])           if random.random() > 0.10 else None,
        "drinking":            wc(DRINKING,  p["drinking_w"])          if random.random() > 0.10 else None,
        "activity_level":      wc(p["activity_choices"], p["activity_w"]) if random.random() > 0.10 else None,
        "interaction_count":   0,
        "conversation_count":  0,
        "tag_weights":         sample_user_tags(p),
    }


SMOKING  = ["never", "sometimes", "regularly"]
DRINKING = ["never", "sometimes", "regularly"]


def gen_event(space_id: str, status: str, preferred_cluster: int, popularity: float) -> dict:
    today = date.today()
    if status == "completed":
        starts      = rand_date(today - timedelta(days=365), today - timedelta(days=1))
        # Popular events fill up; niche ones have fewer attendees
        base_count  = round(10 + popularity * 70)
        a_count     = round(random.gauss(base_count, base_count * 0.25))
        a_count     = max(2, a_count)
        avg_age     = round(random.gauss(28 + preferred_cluster * 2, 5), 1)
    else:
        starts      = rand_date(today + timedelta(days=1), today + timedelta(days=180))
        a_count     = round(random.gauss(popularity * 25, 5))
        a_count     = max(0, a_count)
        avg_age     = round(random.gauss(28 + preferred_cluster * 2, 5), 1) if a_count > 0 else None

    max_att = random.choice([None, None, 20, 30, 50, 100, 200])
    price   = 0 if random.random() < 0.65 else random.choice([5, 10, 15, 20, 30])

    return {
        "id":               uid(),
        "space_id":         space_id,
        "tags":             sample_entity_tags(preferred_cluster),
        "starts_at":        f"{starts} 20:00:00",
        "max_attendees":    max_att,
        "is_paid":          price > 0,
        "status":           status,
        "attendee_count":   a_count,
        "avg_attendee_age": avg_age,
        # internal — not in real export
        "preferred_cluster": preferred_cluster,
        "popularity":        round(popularity, 3),
    }


def gen_space(preferred_cluster: int, popularity: float) -> dict:
    return {
        "id":             uid(),
        "tags":           sample_entity_tags(preferred_cluster),
        "member_count":   0,
        "avg_member_age": None,
        "event_count":    0,
        # internal
        "preferred_cluster": preferred_cluster,
        "popularity":        round(popularity, 3),
    }


# ─── Interaction generation ────────────────────────────────────────────────────

def compute_item_persona_affinity(tags: list[str]) -> list[float]:
    """
    Returns a 6-dim vector: fraction of tags belonging to each cluster.
    Used to score how well an event/space matches each persona.
    """
    counts = [0] * 6
    for tag in tags:
        ci = TAG_TO_CLUSTER.get(tag)
        if ci is not None:
            counts[ci] += 1
    total = sum(counts) or 1
    return [c / total for c in counts]


def assign_interactions(
    users:          list[dict],
    events:         list[dict],
    spaces:         list[dict],
    n_interactions: int,
) -> list[dict]:
    """
    Generates ~n_interactions positive pairs.

    Rules:
    1. PERSONA COHERENCE: users sample events/spaces primarily from their
       persona's pool (items with high affinity for their cluster).
       This creates the collaborative filtering signal HGT needs.
    2. POPULARITY BOOST: high-popularity items have higher base probability.
    3. CONVERSATION CLUSTERING: users prefer same-persona partners (70%).
    4. SERENDIPITY: 15% of each user's interactions are cross-persona (exploration).
    """
    n_users = len(users)
    n_ev    = max(1, round(n_interactions * 0.60 / n_users))
    n_sp    = max(1, round(n_interactions * 0.25 / n_users))
    n_co    = max(1, round(n_interactions * 0.15 / n_users))

    # Pre-compute persona affinity for each event and space
    for e in events:
        e["_affinity"] = compute_item_persona_affinity(e["tags"])
    for s in spaces:
        s["_affinity"] = compute_item_persona_affinity(s["tags"])

    # Build per-persona sorted pools (sorted by persona affinity desc)
    n_personas = len(PERSONAS)
    persona_event_pools: list[list[str]] = []
    persona_space_pools: list[list[str]] = []

    for p_idx in range(n_personas):
        # Sort events by affinity[p_idx] desc, then by popularity desc
        ev_sorted = sorted(
            events,
            key=lambda e: (e["_affinity"][p_idx] * 0.7 + e["popularity"] * 0.3),
            reverse=True,
        )
        persona_event_pools.append([e["id"] for e in ev_sorted])

        sp_sorted = sorted(
            spaces,
            key=lambda s: (s["_affinity"][p_idx] * 0.7 + s["popularity"] * 0.3),
            reverse=True,
        )
        persona_space_pools.append([s["id"] for s in sp_sorted])

    # Group users by persona for conversation clustering
    users_by_persona: list[list[str]] = [[] for _ in range(n_personas)]
    for u in users:
        users_by_persona[u["persona_idx"]].append(u["id"])

    all_event_ids = [e["id"] for e in events]
    all_space_ids = [s["id"] for s in spaces]

    interactions: list[dict]   = []
    positive_pairs: set[tuple] = set()
    user_event_cnt: dict[str, int] = defaultdict(int)
    user_conv_cnt:  dict[str, int] = defaultdict(int)
    space_member_data: dict[str, list[dict]] = defaultdict(list)

    today            = date.today()
    event_status_map = {e["id"]: e["status"] for e in events}

    def _recency(d: date) -> float:
        """Exponential decay: exp(-days_since / 180). Half-life ≈ 6 months."""
        return math.exp(-(today - d).days / 180.0)

    def _add(uid: str, iid: str, itype: str) -> bool:
        if (uid, iid) not in positive_pairs:
            positive_pairs.add((uid, iid))

            # Random date in the past year (skewed toward more recent)
            days_back  = round(random.betavariate(1.5, 3.0) * 364) + 1
            created_at = today - timedelta(days=days_back)

            if itype == "event":
                ev_status = event_status_map.get(iid, "published")
                type_w    = 1.0 if ev_status == "completed" else 0.7
            elif itype == "space":
                type_w = 0.9
            else:   # user / conversation
                type_w = 0.85

            interactions.append({
                "user_id":    uid,
                "item_id":    iid,
                "item_type":  itype,
                "weight":     round(type_w * _recency(created_at), 4),
                "created_at": created_at.isoformat(),
            })
            return True
        return False

    for user in users:
        uid   = user["id"]
        p_idx = user["persona_idx"]

        # ── Events ──────────────────────────────────────────────────────────
        pool  = persona_event_pools[p_idx]
        # 85% from top-30% (persona match), 15% random (serendipity)
        top   = pool[: max(1, len(pool) // 3)]
        rest  = pool[len(pool) // 3 :]

        n_top  = max(0, round(n_ev * 0.85))
        n_rest = n_ev - n_top
        sampled_ev  = random.sample(top,  min(n_top,  len(top)))
        sampled_ev += random.sample(rest, min(n_rest, len(rest)))
        if len(sampled_ev) < n_ev:
            extra = [e for e in all_event_ids if e not in sampled_ev]
            sampled_ev += random.sample(extra, min(n_ev - len(sampled_ev), len(extra)))

        for eid in sampled_ev:
            if _add(uid, eid, "event"):
                user_event_cnt[uid] += 1

        # ── Spaces ──────────────────────────────────────────────────────────
        pool  = persona_space_pools[p_idx]
        top   = pool[: max(1, len(pool) // 3)]
        rest  = pool[len(pool) // 3 :]

        n_top  = max(0, round(n_sp * 0.85))
        n_rest = n_sp - n_top
        sampled_sp  = random.sample(top,  min(n_top,  len(top)))
        sampled_sp += random.sample(rest, min(n_rest, len(rest)))

        for sid in sampled_sp:
            if _add(uid, sid, "space"):
                space_member_data[sid].append(user)

        # ── Conversations ───────────────────────────────────────────────────
        same = [u for u in users_by_persona[p_idx] if u != uid]
        diff = [u["id"] for u in users if u["persona_idx"] != p_idx]

        n_same = round(n_co * 0.70)
        n_diff = n_co - n_same
        partners  = random.sample(same, min(n_same, len(same)))
        partners += random.sample(diff, min(n_diff, len(diff)))

        for pid in partners:
            if _add(uid, pid, "user"):
                user_conv_cnt[uid] += 1
                user_conv_cnt[pid] += 1

    # ── Update denormalized stats ────────────────────────────────────────────
    for user in users:
        user["interaction_count"]  = user_event_cnt[user["id"]]
        user["conversation_count"] = user_conv_cnt[user["id"]]

    for space in spaces:
        members = space_member_data[space["id"]]
        space["member_count"] = len(members)
        ages = []
        for m in members:
            if m.get("birthdate"):
                bd = date.fromisoformat(m["birthdate"])
                ages.append((date.today() - bd).days / 365.25)
        space["avg_member_age"] = round(sum(ages) / len(ages), 1) if ages else None

    return interactions


# ─── Strip internal fields before writing ──────────────────────────────────────

def _clean_user(u: dict) -> dict:
    return {k: v for k, v in u.items() if k not in ("persona", "persona_idx")}

def _clean_event(e: dict) -> dict:
    return {k: v for k, v in e.items() if k not in ("preferred_cluster", "popularity", "_affinity")}

def _clean_space(s: dict) -> dict:
    return {k: v for k, v in s.items() if k not in ("preferred_cluster", "popularity", "_affinity")}


# ─── Main ──────────────────────────────────────────────────────────────────────

def generate(
    n_users:        int = N_USERS,
    n_events:       int = N_EVENTS,
    n_spaces:       int = N_SPACES,
    n_interactions: int = N_INTERACTIONS,
    out_dir:        str = os.path.join(os.path.dirname(__file__), "training-data"),
    seed:           int = 42,
) -> None:
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating synthetic training data  (seed={seed})")
    print(f"  users={n_users:,}  events={n_events:,}  spaces={n_spaces:,}  interactions≈{n_interactions:,}\n")

    # ── Spaces ─────────────────────────────────────────────────────────────
    # Clusters distributed evenly; popularity follows power law
    space_popularities = power_law_popularity(n_spaces)
    spaces = []
    for i, pop in enumerate(space_popularities):
        cluster = i % len(TAG_CLUSTERS)
        spaces.append(gen_space(cluster, pop))

    # ── Events ─────────────────────────────────────────────────────────────
    # ~60% completed (historical), ~40% published (upcoming)
    event_popularities = power_law_popularity(n_events)
    events = []
    for i, pop in enumerate(event_popularities):
        cluster = i % len(TAG_CLUSTERS)
        status  = "completed" if i < round(n_events * 0.6) else "published"
        events.append(gen_event(random.choice(spaces)["id"], status, cluster, pop))

    event_counts: dict[str, int] = defaultdict(int)
    for e in events:
        event_counts[e["space_id"]] += 1
    for space in spaces:
        space["event_count"] = event_counts[space["id"]]

    # ── Users ──────────────────────────────────────────────────────────────
    # Personas are evenly distributed (each cluster gets ~1/6 of users)
    print("  Generating users...")
    users = []
    for i in range(n_users):
        p_idx = i % len(PERSONAS)
        users.append(gen_user(p_idx))
    random.shuffle(users)  # don't leak ordering into training

    # ── Interactions ───────────────────────────────────────────────────────
    print("  Generating interactions...")
    interactions = assign_interactions(users, events, spaces, n_interactions)

    # ── Write ──────────────────────────────────────────────────────────────
    def _write(name: str, data: list) -> None:
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ {name:<25} {len(data):>8,} records")

    print()
    _write("users.json",        [_clean_user(u) for u in users])
    _write("events.json",       [_clean_event(e) for e in events])
    _write("spaces.json",       [_clean_space(s) for s in spaces])
    _write("interactions.json", interactions)

    n_pos = len(interactions)
    print(f"\n  positive pairs : {n_pos:,}  (target {n_interactions:,})")
    print(f"  avg per user   : {n_pos / n_users:.1f}")

    # ── Quick sanity report ────────────────────────────────────────────────
    by_type = defaultdict(int)
    for ix in interactions:
        by_type[ix["item_type"]] += 1
    print(f"\n  interaction breakdown:")
    for t, c in sorted(by_type.items()):
        print(f"    {t:<8}  {c:>8,}  ({100*c/n_pos:.1f}%)")

    avg_w = sum(ix["weight"] for ix in interactions) / max(1, n_pos)
    print(f"\n  avg weight     : {avg_w:.3f}  (1.0 = very recent attended, ~0.1 = older interaction)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ML training data.")
    parser.add_argument("--users",        type=int, default=N_USERS)
    parser.add_argument("--events",       type=int, default=N_EVENTS)
    parser.add_argument("--spaces",       type=int, default=N_SPACES)
    parser.add_argument("--interactions", type=int, default=N_INTERACTIONS)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "training-data"))
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    generate(
        n_users=args.users,
        n_events=args.events,
        n_spaces=args.spaces,
        n_interactions=args.interactions,
        out_dir=args.out_dir,
        seed=args.seed,
    )
