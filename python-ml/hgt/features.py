"""
Feature engineering: transforms raw entity data into typed float vectors.

Each entity type has its own feature space (different dims, different semantics):

  User  → 60-dim  (tags, profile, interactions)
  Event → 51-dim  (tags, attendees, timing, price)
  Space → 43-dim  (tags, avg_member_age, member_count, event_count)

See ml.config for exact layout documentation.
"""

from __future__ import annotations
from typing import Optional

from hgt.config import (
    TAG_TO_IDX, NUM_TAGS,
    GENDER_TO_IDX, REL_INTENT_TO_IDX, SMOKING_TO_IDX, DRINKING_TO_IDX, ACTIVITY_TO_IDX,
    GENDER_VOCAB, REL_INTENT_VOCAB, SMOKING_VOCAB, DRINKING_VOCAB, ACTIVITY_VOCAB,
    USER_DIM, EVENT_DIM, SPACE_DIM,
)
from hgt.utils import (
    normalize_age, normalize_count, normalize_days, normalize_price_cents,
    time_cyclical_features, calculate_age, days_until,
)


# ─── Vocabulary helpers ─────────────────────────────────────────────────────────

def _onehot(vocab_to_idx: dict[str, int], vocab_size: int, value: Optional[str]) -> list[float]:
    v = [0.0] * vocab_size
    if value and value in vocab_to_idx:
        v[vocab_to_idx[value]] = 1.0
    return v


def _multihot(vocab_to_idx: dict[str, int], vocab_size: int, values: list[str]) -> list[float]:
    v = [0.0] * vocab_size
    for val in (values or []):
        if val in vocab_to_idx:
            v[vocab_to_idx[val]] = 1.0
    return v


# ─── User ──────────────────────────────────────────────────────────────────────

def build_user_features(
    birthdate,
    tags: list[str],
    gender: Optional[str] = None,
    relationship_intent: list[str] | None = None,
    smoking: Optional[str] = None,
    drinking: Optional[str] = None,
    activity_level: Optional[str] = None,
    interaction_count: int = 0,
) -> list[float]:
    """
    Builds a 60-dim user feature vector.

    Args:
        birthdate:            users.birthdate (date, str "YYYY-MM-DD", or None)
        tags:                 users.tags (list of declared tag strings)
        gender:               users.gender enum value or None
        relationship_intent:  users.relationship_intent array or None
        smoking:              users.smoking enum value or None
        drinking:             users.drinking enum value or None
        activity_level:       users.activity_level enum value or None
        interaction_count:    events attended + spaces joined
    """
    # [0:40] tags multi-hot
    tags_vec = [0.0] * NUM_TAGS
    for tag in (tags or []):
        idx = TAG_TO_IDX.get(tag)
        if idx is not None:
            tags_vec[idx] = 1.0

    # [40] age
    age_vec = [normalize_age(calculate_age(birthdate))]

    # [41:44] gender one-hot
    gender_vec = _onehot(GENDER_TO_IDX, len(GENDER_VOCAB), gender)

    # [44:48] relationship_intent multi-hot
    rel_vec = _multihot(REL_INTENT_TO_IDX, len(REL_INTENT_VOCAB), relationship_intent or [])

    # [48:51] smoking one-hot
    smoking_vec = _onehot(SMOKING_TO_IDX, len(SMOKING_VOCAB), smoking)

    # [51:54] drinking one-hot
    drinking_vec = _onehot(DRINKING_TO_IDX, len(DRINKING_VOCAB), drinking)

    # [54:59] activity one-hot
    activity_vec = _onehot(ACTIVITY_TO_IDX, len(ACTIVITY_VOCAB), activity_level)

    # [59] interaction count (events attended + spaces joined) — sqrt-normalized, scale=10
    interaction_vec = [normalize_count(interaction_count, scale=10.0)]

    vec = tags_vec + age_vec + gender_vec + rel_vec + smoking_vec + drinking_vec + activity_vec + interaction_vec
    assert len(vec) == USER_DIM, f"Expected {USER_DIM}, got {len(vec)}"
    return vec


# ─── Event ─────────────────────────────────────────────────────────────────────

def build_event_features(
    tags: list[str],
    avg_attendee_age: Optional[float],
    attendee_count: int = 0,
    days_until_event: Optional[int] = None,
    starts_at=None,
    max_attendees: Optional[int] = None,
    is_paid: bool = False,
    price_cents: Optional[int] = None,
) -> list[float]:
    """
    Builds a 51-dim event feature vector.

    For past events, attendee_count should be the number of users with
    status='attended' (real participation). For upcoming events, use status='going'.

    Args:
        tags:              events.tags
        avg_attendee_age:  AVG(age) of real attendees (past) or registered (upcoming)
        attendee_count:    actual participants (past) or registered (upcoming)
        days_until_event:  days from today to starts_at; negative = past event
        starts_at:         event start timestamp used for hour/day cyclical encoding
        max_attendees:     events.max_attendees (None = no cap)
        is_paid:           True if events.price > 0
        price_cents:       events.price in cents
    """
    # [0:40] tags multi-hot
    tags_vec = [0.0] * NUM_TAGS
    for tag in (tags or []):
        idx = TAG_TO_IDX.get(tag)
        if idx is not None:
            tags_vec[idx] = 1.0

    # [40] avg attendee age
    age_vec = [normalize_age(avg_attendee_age)]

    # [41] attendee count — sqrt-normalized, scale=20
    count_vec = [normalize_count(attendee_count, scale=20.0)]

    if days_until_event is None:
        days_until_event = days_until(starts_at)

    # [42] days until event — negative=past, 0=today, 0.5≈6 months, 1=1 year+
    days_vec = [normalize_days(days_until_event)]

    # [43] capacity fill rate — how full the event is (demand signal)
    #   0.5 = unknown/unlimited capacity (neutral)
    if max_attendees and max_attendees > 0:
        fill_rate = min(1.0, attendee_count / max_attendees)
    else:
        fill_rate = 0.5
    fill_vec = [fill_rate]

    # [44] is_paid — paid events attract different audiences than free ones
    inferred_paid = is_paid or (price_cents is not None and price_cents > 0)
    paid_vec = [1.0 if inferred_paid else 0.0]

    # [45] numeric price signal (log-normalized)
    price_vec = [normalize_price_cents(price_cents)]

    # [46:50] cyclical calendar/time + weekend
    time_vec = time_cyclical_features(starts_at)

    vec = tags_vec + age_vec + count_vec + days_vec + fill_vec + paid_vec + price_vec + time_vec
    assert len(vec) == EVENT_DIM, f"Expected {EVENT_DIM}, got {len(vec)}"
    return vec


# ─── Space ─────────────────────────────────────────────────────────────────────

def build_space_features(
    tags: list[str],
    avg_member_age: Optional[float],
    member_count: int = 0,
    event_count: int = 0,
) -> list[float]:
    """
    Builds a 43-dim space feature vector.

    Args:
        tags:           spaces.tags
        avg_member_age: AVG(age) of active members, or None
        member_count:   number of active members
        event_count:    number of events organized by this space
    """
    # [0:40] tags multi-hot
    tags_vec = [0.0] * NUM_TAGS
    for tag in (tags or []):
        idx = TAG_TO_IDX.get(tag)
        if idx is not None:
            tags_vec[idx] = 1.0

    # [40] avg member age
    age_vec = [normalize_age(avg_member_age)]

    # [41] member count — sqrt-normalized, scale=30
    count_vec = [normalize_count(member_count, scale=30.0)]

    # [42] event count — how active the space is, sqrt-normalized, scale=10
    event_vec = [normalize_count(event_count, scale=10.0)]

    vec = tags_vec + age_vec + count_vec + event_vec
    assert len(vec) == SPACE_DIM, f"Expected {SPACE_DIM}, got {len(vec)}"
    return vec


# ─── Cold start: Jaccard-based tag similarity ──────────────────────────────────

def jaccard_tag_similarity(tags_a: list[str], tags_b: list[str]) -> float:
    """
    Jaccard similarity between two tag lists.
    Used as fallback when ML model is not trained yet.
    """
    if not tags_a or not tags_b:
        return 0.0
    set_a, set_b = set(tags_a), set(tags_b)
    return len(set_a & set_b) / len(set_a | set_b)
