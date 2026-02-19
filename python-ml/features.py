"""
Feature engineering: transforms raw DB rows into a uniform 45-dim float vector.

Layout:
  [0:40]  tag weights (user: 0.0-1.0 from user_interests; event/space: 1.0 if tag present)
  [40]    age normalized to 0-1 (18→0, 65→1)
  [41:44] entity_type one-hot: [user, event, space]
  [44]    popularity normalized to 0-1
"""

from __future__ import annotations
from datetime import date
from typing import Optional
from config import (
    TAG_TO_IDX, NUM_TAGS, ENTITY_TYPE_TO_IDX, FEATURE_DIM,
    AGE_MIN, AGE_MAX,
)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def normalize_age(age: Optional[float]) -> float:
    if age is None:
        return 0.5
    return max(0.0, min(1.0, (age - AGE_MIN) / (AGE_MAX - AGE_MIN)))


def normalize_popularity(count: int | None, max_count: int = 500) -> float:
    if not count:
        return 0.0
    return min(1.0, count / max_count)


def calculate_age(birthdate) -> float | None:
    if birthdate is None:
        return None
    today = date.today()
    if isinstance(birthdate, str):
        from datetime import datetime
        birthdate = datetime.strptime(birthdate, "%Y-%m-%d").date()
    age = today.year - birthdate.year - (
        (today.month, today.day) < (birthdate.month, birthdate.day)
    )
    return float(age)


def _base_vector() -> list[float]:
    return [0.0] * FEATURE_DIM


def _set_entity_type(vec: list[float], entity_type: str) -> None:
    idx = ENTITY_TYPE_TO_IDX.get(entity_type)
    if idx is not None:
        vec[NUM_TAGS + 1 + idx] = 1.0


def _set_tags_from_weights(vec: list[float], tag_weights: dict[str, float]) -> None:
    """User: tag → weight (0.0-1.0) from user_interests."""
    for tag, weight in tag_weights.items():
        idx = TAG_TO_IDX.get(tag)
        if idx is not None:
            vec[idx] = float(weight)


def _set_tags_from_list(vec: list[float], tags: list[str]) -> None:
    """Event/Space: tag is either present (1.0) or absent (0.0)."""
    for tag in (tags or []):
        idx = TAG_TO_IDX.get(tag)
        if idx is not None:
            vec[idx] = 1.0


# ─── User ──────────────────────────────────────────────────────────────────────

def build_user_features(
    birthdate,
    tag_weights: "dict[str, float]",
    interaction_count: int = 0,
) -> "list[float]":
    """
    Args:
        birthdate:         users.birthdate (date or str or None)
        tag_weights:       {tag: weight} from user_interests
        interaction_count: total positive interactions (joins + attendances + accepted matches)
    """
    vec = _base_vector()
    _set_tags_from_weights(vec, tag_weights)
    vec[NUM_TAGS] = normalize_age(calculate_age(birthdate))
    _set_entity_type(vec, "user")
    vec[FEATURE_DIM - 1] = normalize_popularity(interaction_count)
    return vec


# ─── Event ─────────────────────────────────────────────────────────────────────

def build_event_features(
    tags: "list[str]",
    avg_attendee_age: Optional[float],
    attendee_count: int = 0,
) -> "list[float]":
    """
    Args:
        tags:              events.tags
        avg_attendee_age:  AVG(age) of attendees with status going/attended, or None
        attendee_count:    number of going/attended attendees
    """
    vec = _base_vector()
    _set_tags_from_list(vec, tags)
    vec[NUM_TAGS] = normalize_age(avg_attendee_age)
    _set_entity_type(vec, "event")
    vec[FEATURE_DIM - 1] = normalize_popularity(attendee_count)
    return vec


# ─── Space ─────────────────────────────────────────────────────────────────────

def build_space_features(
    tags: "list[str]",
    avg_member_age: Optional[float],
    member_count: int = 0,
) -> "list[float]":
    """
    Args:
        tags:           spaces.tags
        avg_member_age: AVG(age) of active members, or None
        member_count:   number of active members
    """
    vec = _base_vector()
    _set_tags_from_list(vec, tags)
    vec[NUM_TAGS] = normalize_age(avg_member_age)
    _set_entity_type(vec, "space")
    vec[FEATURE_DIM - 1] = normalize_popularity(member_count, max_count=1000)
    return vec


# ─── Cold start: Jaccard-based tag similarity ──────────────────────────────────

def jaccard_tag_similarity(
    tag_weights_a: "dict[str, float]",
    tags_b: "list[str]",
) -> float:
    """
    Weighted Jaccard between user tag weights and a flat tag list (event/space).
    Used as fallback when ML model is not trained yet.
    """
    if not tag_weights_a or not tags_b:
        return 0.0
    set_b = set(tags_b)
    intersection = sum(w for tag, w in tag_weights_a.items() if tag in set_b)
    union = sum(tag_weights_a.values()) + len(set_b - set(tag_weights_a.keys()))
    return intersection / union if union > 0 else 0.0
