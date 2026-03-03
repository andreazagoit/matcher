"""
Feature engineering: transforms raw entity data into typed float vectors.

Each entity type has its own feature space (different dims, different semantics):

  User     → 18-dim  (demographic / behavioural profile only)
  Event    → 75-dim  (categories pool, attendees, timing, price)
  Space    → 67-dim  (categories pool, avg_member_age, member_count, event_count)
  Category → 64-dim  (64d text-embedding-3-small directly)

User category preferences are NOT stored in the user node vector.
They are represented exclusively by user→likes_category edges in the graph,
which is where the HGT convolution layers pick them up.

See hgt.config for exact layout documentation.
"""

from __future__ import annotations
from typing import Optional

from hgt.config import (
    CATEGORY_EMBED_DIM,
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


def _pool_category_embeddings(
    category_embeddings: list[list[float]] | None,
    weights: list[float] | None = None
) -> list[float]:
    """
    Returns the weighted mean vector of the provided 64-dim category embeddings, or zeros.
    If weights are provided, they must match the length of category_embeddings.
    """
    if not category_embeddings:
        return [0.0] * CATEGORY_EMBED_DIM

    num_cats = len(category_embeddings)
    dim = len(category_embeddings[0])

    if weights is None:
        weights = [1.0] * num_cats

    if len(weights) != num_cats:
        weights = [1.0] * num_cats

    weighted_sum = [0.0] * dim
    total_weight = sum(weights) or 1.0

    for i, emb in enumerate(category_embeddings):
        w = weights[i]
        for j, val in enumerate(emb):
            weighted_sum[j] += val * w

    return [val / total_weight for val in weighted_sum]


# ─── User ──────────────────────────────────────────────────────────────────────

def build_user_features(
    birthdate,
    gender: Optional[str] = None,
    relationship_intent: list[str] | None = None,
    smoking: Optional[str] = None,
    drinking: Optional[str] = None,
    activity_level: Optional[str] = None,
) -> list[float]:
    """
    Builds a user feature vector from demographic and behavioural profile data.

    Category preferences are intentionally excluded — they are represented
    entirely by user→likes_category edges in the graph, where the HGT
    convolution layers aggregate them via message passing.
    """
    # [0] age
    age_vec = [normalize_age(calculate_age(birthdate))]

    # [1:4] gender one-hot
    gender_vec = _onehot(GENDER_TO_IDX, len(GENDER_VOCAB), gender)

    # [4:8] relationship_intent multi-hot
    rel_vec = [0.0] * len(REL_INTENT_VOCAB)
    for val in (relationship_intent or []):
        if val in REL_INTENT_TO_IDX:
            rel_vec[REL_INTENT_TO_IDX[val]] = 1.0

    # [8:11] smoking one-hot
    smoking_vec = _onehot(SMOKING_TO_IDX, len(SMOKING_VOCAB), smoking)

    # [11:14] drinking one-hot
    drinking_vec = _onehot(DRINKING_TO_IDX, len(DRINKING_VOCAB), drinking)

    # [14:19] activity one-hot
    activity_vec = _onehot(ACTIVITY_TO_IDX, len(ACTIVITY_VOCAB), activity_level)

    vec = age_vec + gender_vec + rel_vec + smoking_vec + drinking_vec + activity_vec
    assert len(vec) == USER_DIM, f"Expected {USER_DIM}, got {len(vec)}"
    return vec


# ─── Event ─────────────────────────────────────────────────────────────────────

def build_event_features(
    category_embeddings: list[list[float]] | None,
    starts_at,
    avg_attendee_age: Optional[float] = None,
    attendee_count: int = 0,
    days_until_event: Optional[int] = None,
    max_attendees: Optional[int] = None,
    is_paid: bool = False,
    price_cents: Optional[int] = None,
) -> list[float]:
    """
    Builds an event feature vector using pooled category embeddings.
    """
    # [0:CATEGORY_EMBED_DIM] categories mean pool
    categories_vec = _pool_category_embeddings(category_embeddings)

    # [+1] avg attendee age
    avg_age_vec = [normalize_age(avg_attendee_age)]

    # [+1] attendee count
    attendees_vec = [normalize_count(attendee_count, scale=10.0)]

    # [+1] days until event
    days_vec = [0.0]
    if days_until_event is not None:
        days_vec = [normalize_days(days_until_event)]
    elif starts_at:
        days_vec = [normalize_days(days_until(starts_at))]

    # [+1] is paid
    is_paid_vec = [1.0 if is_paid else 0.0]

    # [+1] price cents (log normalized)
    price_vec = [normalize_price_cents(price_cents)]

    # [+2] start hour cyclical
    # [+2] start weekday cyclical
    # [+1] is weekend
    cyclical_features = time_cyclical_features(starts_at)
    start_hour_cycles    = cyclical_features[0:2]
    start_weekday_cycles = cyclical_features[2:4]
    is_weekend_vec       = [cyclical_features[4]]

    # [+1] capacity fill rate
    fill_rate_vec = [0.0]
    if max_attendees and max_attendees > 0:
        fill_rate_vec = [min(1.0, float(attendee_count) / float(max_attendees))]

    vec = (categories_vec + avg_age_vec + attendees_vec + days_vec + fill_rate_vec +
           is_paid_vec + price_vec + start_hour_cycles + start_weekday_cycles +
           is_weekend_vec)

    assert len(vec) == EVENT_DIM, f"Expected {EVENT_DIM}, got {len(vec)}"
    return vec


# ─── Space ─────────────────────────────────────────────────────────────────────

def build_space_features(
    category_embeddings: list[list[float]] | None,
    avg_member_age: Optional[float] = None,
    member_count: int = 0,
    event_count: int = 0,
) -> list[float]:
    """
    Builds a space feature vector using pooled category embeddings.
    """
    # [0:CATEGORY_EMBED_DIM] categories mean pool
    categories_vec = _pool_category_embeddings(category_embeddings)

    # [+1] avg member age
    avg_age_vec = [normalize_age(avg_member_age)]

    # [+1] member count
    members_vec = [normalize_count(member_count, scale=31.62)]  # sqrt(1000)

    # [+1] event count
    events_vec = [normalize_count(event_count, scale=10.0)]

    vec = categories_vec + avg_age_vec + members_vec + events_vec
    assert len(vec) == SPACE_DIM, f"Expected {SPACE_DIM}, got {len(vec)}"
    return vec
