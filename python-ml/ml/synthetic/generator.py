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
N_INTERACTIONS = 100_000   # positive pairs (split ~70% event, 30% space)
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

from ml.config import TAG_VOCAB, TRAINING_DATA_DIR

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
SMOKING     = ["never", "sometimes", "regularly"]
DRINKING    = ["never", "sometimes", "regularly"]

SPACE_ARCHETYPES: list[dict] = [
    {"name": "nerd_hub", "cluster": 4, "tag_pool": ["gaming", "coding", "board_games", "reading", "coffee"]},
    {"name": "board_games_society", "cluster": 5, "tag_pool": ["board_games", "parties", "coffee", "reading", "gaming"]},
    {"name": "indie_gaming_club", "cluster": 4, "tag_pool": ["gaming", "coding", "music", "parties", "coffee"]},
    {"name": "tech_founders_circle", "cluster": 4, "tag_pool": ["coding", "languages", "coffee", "travel", "reading"]},
    {"name": "ai_builders_lab", "cluster": 4, "tag_pool": ["coding", "reading", "diy", "coffee", "gaming"]},
    {"name": "cinephile_collective", "cluster": 1, "tag_pool": ["cinema", "theater", "reading", "coffee", "museums"]},
    {"name": "book_cafe_club", "cluster": 1, "tag_pool": ["reading", "writing", "coffee", "theater", "museums"]},
    {"name": "street_photo_crew", "cluster": 1, "tag_pool": ["photography", "art", "travel", "coffee", "museums"]},
    {"name": "modern_art_collective", "cluster": 1, "tag_pool": ["art", "museums", "drawing", "photography", "theater"]},
    {"name": "live_music_tribe", "cluster": 4, "tag_pool": ["music", "live_music", "parties", "coffee", "art"]},
    {"name": "foodie_circle", "cluster": 2, "tag_pool": ["restaurants", "street_food", "cooking", "coffee", "travel"]},
    {"name": "wine_tasting_society", "cluster": 2, "tag_pool": ["wine", "restaurants", "craft_beer", "street_food", "coffee"]},
    {"name": "coffee_explorers", "cluster": 2, "tag_pool": ["coffee", "street_food", "restaurants", "reading", "travel"]},
    {"name": "urban_runners", "cluster": 3, "tag_pool": ["running", "gym", "cycling", "yoga", "swimming"]},
    {"name": "yoga_wellness", "cluster": 3, "tag_pool": ["yoga", "running", "swimming", "gym", "meditation"]},
    {"name": "mountain_hikers", "cluster": 0, "tag_pool": ["trekking", "mountains", "camping", "travel", "climbing"]},
    {"name": "climbing_crew", "cluster": 0, "tag_pool": ["climbing", "mountains", "trekking", "camping", "gym"]},
    {"name": "travel_backpackers", "cluster": 5, "tag_pool": ["travel", "languages", "beach", "mountains", "photography"]},
    {"name": "pet_lovers_club", "cluster": 5, "tag_pool": ["pets", "volunteering", "travel", "coffee", "parties"]},
    {"name": "language_exchange_lounge", "cluster": 5, "tag_pool": ["languages", "travel", "coffee", "parties", "reading"]},
]

EVENT_ARCHETYPES: list[dict] = [
    {"name": "board_game_night", "cluster": 5, "tag_pool": ["board_games", "parties", "coffee"], "price_choices": [0, 500, 1000], "hour_choices": [19, 20, 21], "max_choices": [12, 20, 30]},
    {"name": "esports_tournament", "cluster": 4, "tag_pool": ["gaming", "coding", "parties"], "price_choices": [0, 1000, 1500], "hour_choices": [18, 19, 20], "max_choices": [20, 40, 80]},
    {"name": "lan_party", "cluster": 4, "tag_pool": ["gaming", "coding", "music"], "price_choices": [0, 800, 1200], "hour_choices": [19, 20], "max_choices": [16, 24, 40]},
    {"name": "hackathon", "cluster": 4, "tag_pool": ["coding", "diy", "coffee"], "price_choices": [0, 2000, 3000], "hour_choices": [9, 10], "max_choices": [30, 60, 120]},
    {"name": "ai_workshop", "cluster": 4, "tag_pool": ["coding", "reading", "coffee"], "price_choices": [0, 1500, 2500], "hour_choices": [18, 19], "max_choices": [20, 40, 60]},
    {"name": "startup_pitch_night", "cluster": 4, "tag_pool": ["coding", "languages", "coffee"], "price_choices": [0, 1000, 2000], "hour_choices": [18, 19, 20], "max_choices": [20, 40, 80]},
    {"name": "movie_screening", "cluster": 1, "tag_pool": ["cinema", "theater", "coffee"], "price_choices": [0, 700, 1200], "hour_choices": [20, 21], "max_choices": [20, 40, 80]},
    {"name": "book_discussion", "cluster": 1, "tag_pool": ["reading", "writing", "coffee"], "price_choices": [0, 500, 1000], "hour_choices": [18, 19, 20], "max_choices": [10, 16, 24]},
    {"name": "photo_walk", "cluster": 1, "tag_pool": ["photography", "art", "travel"], "price_choices": [0, 1000], "hour_choices": [9, 10, 16], "max_choices": [12, 20, 30]},
    {"name": "art_workshop", "cluster": 1, "tag_pool": ["art", "drawing", "museums"], "price_choices": [1000, 2000, 3000], "hour_choices": [17, 18, 19], "max_choices": [10, 16, 24]},
    {"name": "open_mic_live", "cluster": 4, "tag_pool": ["music", "live_music", "parties"], "price_choices": [0, 1000, 1500], "hour_choices": [20, 21], "max_choices": [20, 40, 70]},
    {"name": "cooking_class", "cluster": 2, "tag_pool": ["cooking", "restaurants", "street_food"], "price_choices": [1500, 2500, 3500], "hour_choices": [11, 18, 19], "max_choices": [8, 12, 20]},
    {"name": "wine_tasting", "cluster": 2, "tag_pool": ["wine", "restaurants", "craft_beer"], "price_choices": [2000, 3000, 5000], "hour_choices": [19, 20], "max_choices": [12, 20, 30]},
    {"name": "coffee_cupping", "cluster": 2, "tag_pool": ["coffee", "street_food", "reading"], "price_choices": [0, 1000, 1500], "hour_choices": [10, 11, 16], "max_choices": [10, 16, 24]},
    {"name": "running_session", "cluster": 3, "tag_pool": ["running", "cycling", "gym"], "price_choices": [0, 500, 1000], "hour_choices": [7, 8, 18], "max_choices": [12, 25, 40]},
    {"name": "yoga_session", "cluster": 3, "tag_pool": ["yoga", "swimming", "running"], "price_choices": [0, 1000, 1500], "hour_choices": [7, 8, 19], "max_choices": [10, 20, 30]},
    {"name": "mountain_trek", "cluster": 0, "tag_pool": ["trekking", "mountains", "camping"], "price_choices": [0, 1500, 2500], "hour_choices": [7, 8, 9], "max_choices": [10, 20, 30]},
    {"name": "climbing_session", "cluster": 0, "tag_pool": ["climbing", "mountains", "gym"], "price_choices": [1000, 2000, 3000], "hour_choices": [17, 18, 19], "max_choices": [8, 16, 24]},
    {"name": "city_trip", "cluster": 5, "tag_pool": ["travel", "languages", "photography"], "price_choices": [0, 2000, 4000], "hour_choices": [8, 9, 10], "max_choices": [12, 20, 35]},
    {"name": "language_meetup", "cluster": 5, "tag_pool": ["languages", "travel", "coffee"], "price_choices": [0, 500, 1000], "hour_choices": [18, 19, 20], "max_choices": [12, 24, 40]},
]

SPACE_EVENT_COMPATIBILITY: dict[str, list[str]] = {
    "nerd_hub": ["board_game_night", "esports_tournament", "lan_party", "hackathon", "ai_workshop"],
    "board_games_society": ["board_game_night", "language_meetup", "coffee_cupping"],
    "indie_gaming_club": ["esports_tournament", "lan_party", "open_mic_live"],
    "tech_founders_circle": ["hackathon", "ai_workshop", "startup_pitch_night", "language_meetup"],
    "ai_builders_lab": ["hackathon", "ai_workshop", "startup_pitch_night"],
    "cinephile_collective": ["movie_screening", "book_discussion", "photo_walk"],
    "book_cafe_club": ["book_discussion", "coffee_cupping", "language_meetup"],
    "street_photo_crew": ["photo_walk", "city_trip", "art_workshop"],
    "modern_art_collective": ["art_workshop", "photo_walk", "open_mic_live"],
    "live_music_tribe": ["open_mic_live", "movie_screening", "board_game_night"],
    "foodie_circle": ["cooking_class", "wine_tasting", "coffee_cupping"],
    "wine_tasting_society": ["wine_tasting", "cooking_class", "open_mic_live"],
    "coffee_explorers": ["coffee_cupping", "book_discussion", "language_meetup"],
    "urban_runners": ["running_session", "city_trip", "yoga_session"],
    "yoga_wellness": ["yoga_session", "running_session", "coffee_cupping"],
    "mountain_hikers": ["mountain_trek", "city_trip", "climbing_session"],
    "climbing_crew": ["climbing_session", "mountain_trek", "running_session"],
    "travel_backpackers": ["city_trip", "language_meetup", "photo_walk"],
    "pet_lovers_club": ["city_trip", "coffee_cupping", "board_game_night"],
    "language_exchange_lounge": ["language_meetup", "city_trip", "book_discussion"],
}

PERSONA_SPACE_PREFS: dict[str, list[str]] = {
    "outdoor_adventurer": [
        "mountain_hikers", "climbing_crew", "travel_backpackers", "urban_runners",
        "yoga_wellness", "pet_lovers_club", "language_exchange_lounge", "coffee_explorers",
    ],
    "culture_lover": [
        "cinephile_collective", "book_cafe_club", "modern_art_collective", "street_photo_crew",
        "live_music_tribe", "language_exchange_lounge", "coffee_explorers", "travel_backpackers",
    ],
    "foodie": [
        "foodie_circle", "wine_tasting_society", "coffee_explorers", "language_exchange_lounge",
        "book_cafe_club", "live_music_tribe", "travel_backpackers", "pet_lovers_club",
    ],
    "sports_enthusiast": [
        "urban_runners", "yoga_wellness", "mountain_hikers", "climbing_crew",
        "travel_backpackers", "nerd_hub", "language_exchange_lounge", "foodie_circle",
    ],
    "creative": [
        "modern_art_collective", "live_music_tribe", "street_photo_crew", "book_cafe_club",
        "cinephile_collective", "language_exchange_lounge", "indie_gaming_club", "coffee_explorers",
    ],
    "social_butterfly": [
        "language_exchange_lounge", "pet_lovers_club", "travel_backpackers", "board_games_society",
        "foodie_circle", "live_music_tribe", "coffee_explorers", "urban_runners",
    ],
}


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


def sample_time_slot(status: str) -> tuple[int, int]:
    """
    Sample a realistic start time by time-slot (morning/afternoon/evening),
    including minute granularity.
    """
    if status == "published":
        slot = random.choices(
            ["morning", "afternoon", "evening"],
            weights=[0.18, 0.32, 0.50],
            k=1,
        )[0]
    else:
        slot = random.choices(
            ["morning", "afternoon", "evening"],
            weights=[0.22, 0.34, 0.44],
            k=1,
        )[0]

    if slot == "morning":
        hour = random.randint(7, 11)
    elif slot == "afternoon":
        hour = random.randint(12, 17)
    else:
        hour = random.randint(18, 22)

    return hour, 0  # round to nearest hour for cleaner time bands


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
    Sample dense user interests:
      - many low-weight tags from broad exploration (page visits)
      - few high-weight core tags aligned with persona
    """
    prim = TAG_CLUSTERS[persona["cluster"]]
    sec_pool = [t for t in TAG_VOCAB if t not in prim]

    n_total = random.randint(14, 24)
    n_core = random.randint(2, 4)
    n_mid = random.randint(4, 7)
    n_low = max(0, n_total - n_core - n_mid)

    core_tags = random.sample(prim, min(n_core, len(prim)))
    remaining_prim = [t for t in prim if t not in core_tags]
    mid_from_prim = random.sample(remaining_prim, min(len(remaining_prim), max(1, round(n_mid * 0.6))))
    n_mid_sec = max(0, n_mid - len(mid_from_prim))
    mid_from_sec = random.sample(sec_pool, min(n_mid_sec, len(sec_pool)))

    low_pool = [t for t in TAG_VOCAB if t not in set(core_tags + mid_from_prim + mid_from_sec)]
    low_tags = random.sample(low_pool, min(n_low, len(low_pool)))

    result: dict[str, float] = {}
    for tag in core_tags:
        result[tag] = round(random.uniform(0.75, 1.00), 2)
    for tag in (mid_from_prim + mid_from_sec):
        result[tag] = round(random.uniform(0.28, 0.65), 2)
    for tag in low_tags:
        result[tag] = round(random.uniform(0.03, 0.20), 2)
    return result


def bump_user_tag_weights(
    user: dict,
    item_tags: list[str],
    strength: float,
    decay_existing: float = 0.0,
) -> None:
    """
    Simulates incremental preference updates after page visits/interactions.
    Users accumulate many low signals and a few strong ones.
    """
    if not item_tags:
        return
    tw = user["tag_weights"]
    if decay_existing > 0:
        for tag, w in list(tw.items()):
            tw[tag] = max(0.0, round(w * (1.0 - decay_existing), 4))
    for tag in item_tags:
        current = float(tw.get(tag, 0.0))
        inc = strength * (1.0 - min(1.0, current))
        tw[tag] = round(min(1.25, current + inc), 4)


def _sample_tag_count(min_n: int, max_n: int, weights: list[float]) -> int:
    values = list(range(min_n, max_n + 1))
    return random.choices(values, weights=weights, k=1)[0]


def sample_space_tags(archetype: dict) -> list[str]:
    n = _sample_tag_count(1, 5, [0.10, 0.20, 0.40, 0.20, 0.10])
    pool = [t for t in archetype["tag_pool"] if t in TAG_VOCAB]
    if len(pool) < n:
        fallback = [t for t in TAG_VOCAB if t not in pool]
        pool = pool + random.sample(fallback, min(n - len(pool), len(fallback)))
    return random.sample(pool, min(n, len(pool)))


def sample_event_tags(space_tags: list[str], event_archetype: dict) -> list[str]:
    n = _sample_tag_count(1, 4, [0.15, 0.35, 0.35, 0.15])
    n_from_space = max(1, min(len(space_tags), round(n * random.uniform(0.7, 1.0))))
    chosen = set(random.sample(space_tags, n_from_space)) if space_tags else set()

    event_pool = [t for t in event_archetype["tag_pool"] if t in TAG_VOCAB and t not in chosen]
    if len(chosen) < n and event_pool:
        chosen |= set(random.sample(event_pool, min(n - len(chosen), len(event_pool))))

    if len(chosen) < n:
        fallback = [t for t in TAG_VOCAB if t not in chosen]
        chosen |= set(random.sample(fallback, min(n - len(chosen), len(fallback))))

    return list(chosen)


def sample_entity_tags(preferred_cluster: int = -1) -> list[str]:
    if preferred_cluster >= 0:
        pool = list(TAG_CLUSTERS[preferred_cluster])
    else:
        pool = list(random.choice(TAG_CLUSTERS))
    n = _sample_tag_count(1, 4, [0.15, 0.35, 0.35, 0.15])
    if len(pool) < n:
        rest = [t for t in TAG_VOCAB if t not in pool]
        pool += random.sample(rest, min(n - len(pool), len(rest)))
    return random.sample(pool, min(n, len(pool)))


def _tag_score(weight: float | None) -> float:
    if weight is None:
        return -1.0
    if weight <= 0.10:
        return -0.35
    if weight <= 0.40:
        return 0.15
    return 1.0


def _implicit_tag_preference_factor(user_tag_weights: dict[str, float], item_tags: list[str]) -> float:
    if not item_tags:
        return 0.75
    scores = [_tag_score(user_tag_weights.get(tag)) for tag in item_tags]
    mean_score = sum(scores) / len(scores)
    norm = (mean_score + 1.0) / 2.0
    return 0.55 + (norm ** 0.85) * 0.50


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
        "smoking":             wc(SMOKING,   p["smoking_w"])              if random.random() > 0.10 else None,
        "drinking":            wc(DRINKING,  p["drinking_w"])             if random.random() > 0.10 else None,
        "activity_level":      wc(p["activity_choices"], p["activity_w"]) if random.random() > 0.10 else None,
        "interaction_count":   0,
        "tag_weights":         sample_user_tags(p),
    }


def gen_event(space_id: str, status: str, preferred_cluster: int, popularity: float) -> dict:
    today = date.today()
    if status == "completed":
        starts      = rand_date(today - timedelta(days=365), today - timedelta(days=1))
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
    start_hour, start_minute = sample_time_slot(status)

    return {
        "id":               uid(),
        "space_id":         space_id,
        "tags":             sample_entity_tags(preferred_cluster),
        "starts_at":        f"{starts} {start_hour:02d}:{start_minute:02d}:00",
        "max_attendees":    max_att,
        "is_paid":          price > 0,
        "price_cents":      price * 100,
        "attendee_count":   a_count,
        "avg_attendee_age": avg_age,
        "preferred_cluster": preferred_cluster,
        "popularity":        round(popularity, 3),
    }


def gen_space(archetype: dict, popularity: float) -> dict:
    return {
        "id":             uid(),
        "tags":           sample_space_tags(archetype),
        "member_count":   0,
        "avg_member_age": None,
        "event_count":    0,
        "preferred_cluster": archetype["cluster"],
        "archetype":      archetype["name"],
        "popularity":        round(popularity, 3),
    }


# ─── Interaction generation ────────────────────────────────────────────────────

def compute_item_persona_affinity(tags: list[str]) -> list[float]:
    """Returns a 6-dim vector: fraction of tags belonging to each cluster."""
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
    2. POPULARITY BOOST: high-popularity items have higher base probability.
    3. SERENDIPITY: 15% of each user's interactions are cross-persona (exploration).
    """
    n_users = len(users)
    n_ev    = max(1, round(n_interactions * 0.70 / n_users))
    n_sp    = max(1, round(n_interactions * 0.30 / n_users))

    for e in events:
        e["_affinity"] = compute_item_persona_affinity(e["tags"])
    for s in spaces:
        s["_affinity"] = compute_item_persona_affinity(s["tags"])

    n_personas = len(PERSONAS)
    persona_event_pools: list[list[str]] = []
    persona_space_pools: list[list[str]] = []

    for p_idx in range(n_personas):
        ev_sorted = sorted(
            events,
            key=lambda e: (e["_affinity"][p_idx] * 0.7 + e["popularity"] * 0.3),
            reverse=True,
        )
        persona_event_pools.append([e["id"] for e in ev_sorted])

        sp_sorted = sorted(
            spaces,
            key=lambda s: (
                s["_affinity"][p_idx] * 0.6
                + s["popularity"] * 0.25
                + (0.15 if s.get("archetype") in PERSONA_SPACE_PREFS.get(PERSONAS[p_idx]["name"], []) else 0.0)
            ),
            reverse=True,
        )
        persona_space_pools.append([s["id"] for s in sp_sorted])

    all_event_ids = [e["id"] for e in events]

    interactions: list[dict]   = []
    positive_pairs: set[tuple] = set()
    user_event_cnt: dict[str, int] = defaultdict(int)
    user_space_cnt: dict[str, int] = defaultdict(int)
    space_member_data: dict[str, list[dict]] = defaultdict(list)
    event_by_id = {e["id"]: e for e in events}
    space_by_id = {s["id"]: s for s in spaces}
    user_by_id  = {u["id"]: u for u in users}
    today = date.today()

    def _recency(d: date) -> float:
        return math.exp(-(today - d).days / 180.0)

    def _add(user_id: str, iid: str, itype: str) -> bool:
        if (user_id, iid) not in positive_pairs:
            positive_pairs.add((user_id, iid))
            days_back  = round(random.betavariate(1.5, 3.0) * 364) + 1
            created_at = today - timedelta(days=days_back)
            user  = user_by_id[user_id]
            p_idx = user["persona_idx"]
            pref_factor = 1.0
            if itype == "event":
                event = event_by_id[iid]
                try:
                    event_day = date.fromisoformat(str(event["starts_at"]).split(" ")[0])
                except (TypeError, ValueError):
                    event_day = today
                type_w = 1.0 if event_day < today else 0.7
                pref_score = float(event.get("_affinity", [0.0] * len(PERSONAS))[p_idx])
                persona_factor = 0.35 + 0.65 * pref_score
                tag_factor = _implicit_tag_preference_factor(user["tag_weights"], event["tags"])
                pref_factor = persona_factor * tag_factor
                bump_user_tag_weights(user, event["tags"], strength=0.02 + 0.08 * pref_factor)
            else:
                space = space_by_id[iid]
                type_w = 0.9
                pref_score = float(space.get("_affinity", [0.0] * len(PERSONAS))[p_idx])
                persona_factor = 0.35 + 0.65 * pref_score
                tag_factor = _implicit_tag_preference_factor(user["tag_weights"], space["tags"])
                pref_factor = persona_factor * tag_factor
                bump_user_tag_weights(user, space["tags"], strength=0.015 + 0.06 * pref_factor)

            interactions.append({
                "user_id":    user_id,
                "item_id":    iid,
                "item_type":  itype,
                "weight":     round(type_w * pref_factor * _recency(created_at), 4),
                "created_at": created_at.isoformat(),
            })
            return True
        return False

    for user in users:
        user_id = user["id"]
        p_idx   = user["persona_idx"]

        # ── Events ──────────────────────────────────────────────────────────
        pool  = persona_event_pools[p_idx]
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
            if _add(user_id, eid, "event"):
                user_event_cnt[user_id] += 1

        # ── Spaces ──────────────────────────────────────────────────────────
        pool  = persona_space_pools[p_idx]
        top   = pool[: max(1, len(pool) // 3)]
        rest  = pool[len(pool) // 3 :]

        n_top  = max(0, round(n_sp * 0.85))
        n_rest = n_sp - n_top
        sampled_sp  = random.sample(top,  min(n_top,  len(top)))
        sampled_sp += random.sample(rest, min(n_rest, len(rest)))

        for sid in sampled_sp:
            if _add(user_id, sid, "space"):
                space_member_data[sid].append(user)
                user_space_cnt[user_id] += 1

    for user in users:
        user["interaction_count"] = user_event_cnt[user["id"]] + user_space_cnt[user["id"]]

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
    return {k: v for k, v in s.items() if k not in ("preferred_cluster", "archetype", "popularity", "_affinity")}


# ─── Entry point ───────────────────────────────────────────────────────────────

def generate(
    n_users:        int = N_USERS,
    n_events:       int = N_EVENTS,
    n_spaces:       int = N_SPACES,
    n_interactions: int = N_INTERACTIONS,
    out_dir:        str = TRAINING_DATA_DIR,
    seed:           int = 42,
) -> None:
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating synthetic training data  (seed={seed})")
    print(f"  users={n_users:,}  events={n_events:,}  spaces={n_spaces:,}  interactions≈{n_interactions:,}\n")

    space_popularities = power_law_popularity(n_spaces)
    spaces = []
    for i, pop in enumerate(space_popularities):
        cluster = i % len(TAG_CLUSTERS)
        cluster_candidates = [a for a in SPACE_ARCHETYPES if a["cluster"] == cluster]
        if cluster_candidates and random.random() < 0.75:
            archetype = random.choice(cluster_candidates)
        else:
            archetype = random.choice(SPACE_ARCHETYPES)
        spaces.append(gen_space(archetype, pop))

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

    print("  Generating users...")
    users = []
    for i in range(n_users):
        p_idx = i % len(PERSONAS)
        users.append(gen_user(p_idx))
    random.shuffle(users)

    print("  Generating interactions...")
    interactions = assign_interactions(users, events, spaces, n_interactions)

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

    by_type = defaultdict(int)
    for ix in interactions:
        by_type[ix["item_type"]] += 1
    print(f"\n  interaction breakdown:")
    for t, c in sorted(by_type.items()):
        print(f"    {t:<8}  {c:>8,}  ({100*c/n_pos:.1f}%)")

    avg_w = sum(ix["weight"] for ix in interactions) / max(1, n_pos)
    print(f"\n  avg weight     : {avg_w:.3f}  (1.0 = very recent attended, ~0.1 = older interaction)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ML training data.")
    parser.add_argument("--users",        type=int, default=N_USERS)
    parser.add_argument("--events",       type=int, default=N_EVENTS)
    parser.add_argument("--spaces",       type=int, default=N_SPACES)
    parser.add_argument("--interactions", type=int, default=N_INTERACTIONS)
    parser.add_argument("--out-dir",      default=TRAINING_DATA_DIR)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    generate(
        n_users=args.users,
        n_events=args.events,
        n_spaces=args.spaces,
        n_interactions=args.interactions,
        out_dir=args.out_dir,
        seed=args.seed,
    )
