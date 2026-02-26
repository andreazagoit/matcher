"""
Feature engineering: transforms raw entity data into typed float vectors.

Each entity type has its own feature space (different dims, different semantics):

  User  → 83-dim  (tags, profile, interactions)
  Event → 75-dim  (tags, attendees, timing, price)
  Space → 67-dim  (tags, avg_member_age, member_count, event_count)

See ml.config for exact layout documentation.
"""

from __future__ import annotations
from typing import Optional

from hgt.config import (
    TAG_EMBED_DIM,
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


def _pool_tag_embeddings(
    tag_embeddings: list[list[float]] | None, 
    weights: list[float] | None = None
) -> list[float]:
    """
    Returns the weighted mean vector of the provided 64-dim tag embeddings, or zeros.
    If weights are provided, they must match the length of tag_embeddings.
    """
    if not tag_embeddings:
        return [0.0] * TAG_EMBED_DIM
    
    num_tags = len(tag_embeddings)
    dim = len(tag_embeddings[0])
    
    # Default to equal weights if not provided
    if weights is None:
        weights = [1.0] * num_tags
    
    if len(weights) != num_tags:
        # Fallback to simple mean if mismatch
        weights = [1.0] * num_tags

    weighted_sum = [0.0] * dim
    total_weight = sum(weights) or 1.0
    
    for i, emb in enumerate(tag_embeddings):
        w = weights[i]
        for j, val in enumerate(emb):
            weighted_sum[j] += val * w
            
    # Real weighted mean (normalized by sum of weights)
    return [val / total_weight for val in weighted_sum]


# ─── User ──────────────────────────────────────────────────────────────────────

def build_user_features(
    birthdate,
    tag_embeddings: list[list[float]] | None = None,
    tag_weights: list[float] | None = None,
    num_tags: int = 0,
    gender: Optional[str] = None,
    relationship_intent: list[str] | None = None,
    smoking: Optional[str] = None,
    drinking: Optional[str] = None,
    activity_level: Optional[str] = None,
) -> list[float]:
    """
    Builds a user feature vector using pooled dense tags.
    """
    # [0:TAG_EMBED_DIM] tags weighted pool
    tags_vec = _pool_tag_embeddings(tag_embeddings, weights=tag_weights)

    # [+1] age
    age_vec = [normalize_age(calculate_age(birthdate))]

    # [+3] gender one-hot
    gender_vec = _onehot(GENDER_TO_IDX, len(GENDER_VOCAB), gender)

    # [+4] relationship_intent multi-hot
    rel_vec = [0.0] * len(REL_INTENT_VOCAB)
    for val in (relationship_intent or []):
        if val in REL_INTENT_TO_IDX:
            rel_vec[REL_INTENT_TO_IDX[val]] = 1.0

    # [+3] smoking one-hot
    smoking_vec = _onehot(SMOKING_TO_IDX, len(SMOKING_VOCAB), smoking)

    # [+3] drinking one-hot
    drinking_vec = _onehot(DRINKING_TO_IDX, len(DRINKING_VOCAB), drinking)

    # [+5] activity one-hot
    activity_vec = _onehot(ACTIVITY_TO_IDX, len(ACTIVITY_VOCAB), activity_level)

    # [+1] number of tags declared (sqrt normalized, max ~20)
    num_tags_vec = [normalize_count(num_tags, scale=4.47)] # sqrt(20) ~ 4.47

    vec = tags_vec + age_vec + gender_vec + rel_vec + smoking_vec + drinking_vec + activity_vec + num_tags_vec
    assert len(vec) == USER_DIM, f"Expected {USER_DIM}, got {len(vec)}"
    return vec


# ─── Event ─────────────────────────────────────────────────────────────────────

def build_event_features(
    tag_embeddings: list[list[float]] | None,
    num_tags: int,
    starts_at,
    avg_attendee_age: Optional[float] = None,
    attendee_count: int = 0,
    days_until_event: Optional[int] = None,
    max_attendees: Optional[int] = None,
    is_paid: bool = False,
    price_cents: Optional[int] = None,
) -> list[float]:
    """
    Builds an event feature vector using pooled dense tags.
    """
    # [0:TAG_EMBED_DIM] tags mean pool
    tags_vec = _pool_tag_embeddings(tag_embeddings)

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
    start_hour_cycles = cyclical_features[0:2]
    start_weekday_cycles = cyclical_features[2:4]
    is_weekend_vec = [cyclical_features[4]]

    # [+1] capacity fill rate
    fill_rate_vec = [0.0]
    if max_attendees and max_attendees > 0:
        fill_rate = min(1.0, float(attendee_count) / float(max_attendees))
        fill_rate_vec = [fill_rate]

    # [+1] number of tags declared
    num_tags_vec = [normalize_count(num_tags, scale=4.47)]

    vec = tags_vec + avg_age_vec + attendees_vec + days_vec + fill_rate_vec + \
          is_paid_vec + price_vec + start_hour_cycles + start_weekday_cycles + \
          is_weekend_vec + num_tags_vec

    assert len(vec) == EVENT_DIM, f"Expected {EVENT_DIM}, got {len(vec)}"
    return vec


# ─── Space ─────────────────────────────────────────────────────────────────────

def build_space_features(
    tag_embeddings: list[list[float]] | None,
    num_tags: int,
    avg_member_age: Optional[float] = None,
    member_count: int = 0,
    event_count: int = 0,
) -> list[float]:
    """
    Builds a space feature vector using pooled dense tags.
    """
    # [0:TAG_EMBED_DIM] tags mean pool
    tags_vec = _pool_tag_embeddings(tag_embeddings)

    # [+1] avg member age
    avg_age_vec = [normalize_age(avg_member_age)]

    # [+1] member count
    members_vec = [normalize_count(member_count, scale=31.62)] # sqrt(1000) ~ 31.62

    # [+1] event count
    events_vec = [normalize_count(event_count, scale=10.0)] # sqrt(100) = 10.0

    # [+1] number of tags declared
    # Wait, the config says only 3 features after TAG_EMBED_DIM!
    # config.py: SPACE_DIM = TAG_EMBED_DIM + 1 + 1 + 1 (tags + age + member_count + event_count)
    # The original config had NUM_TAGS + 3. Let's strictly follow SPACE_DIM layout.
    
    vec = tags_vec + avg_age_vec + members_vec + events_vec
    assert len(vec) == SPACE_DIM, f"Expected {SPACE_DIM}, got {len(vec)}"
    return vec
