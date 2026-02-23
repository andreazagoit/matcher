"""
Shared numeric and date-time utilities used across features, data, and embed_all.
"""

from __future__ import annotations
import math
from datetime import date, datetime
from typing import Optional

from ml.config import AGE_MIN, AGE_MAX


# ─── Normalization ──────────────────────────────────────────────────────────────

def normalize_age(age: Optional[float]) -> float:
    if age is None:
        return 0.5
    return max(0.0, min(1.0, (age - AGE_MIN) / (AGE_MAX - AGE_MIN)))


def normalize_count(count: int | None, scale: float = 1.0) -> float:
    """Square-root normalization — reduces the impact of outliers."""
    if not count:
        return 0.0
    return min(1.0, math.sqrt(count) / scale)


def normalize_days(days: int | None, max_days: int = 365) -> float:
    """
    Maps days relative to today into [0, 1]:
      -max_days (distant past) → 0.0
       0        (today)        → 0.5
      +max_days (far future)   → 1.0

    Using a symmetric scale preserves temporal information for past events
    (previously all negative values collapsed to 0.0).
    """
    if days is None:
        return 0.5
    return max(0.0, min(1.0, 0.5 + 0.5 * (days / max_days)))


def normalize_price_cents(price_cents: int | None, max_price_cents: int = 50_000) -> float:
    """
    Log-normalized event price in [0, 1].
    0 cents -> 0.0, 50000 cents or more -> 1.0.
    """
    if not price_cents or price_cents <= 0:
        return 0.0
    clipped = min(price_cents, max_price_cents)
    return math.log1p(clipped) / math.log1p(max_price_cents)


# ─── Date / time ────────────────────────────────────────────────────────────────

def parse_datetime(value: Optional[str | datetime]) -> Optional[datetime]:
    """Accept both "YYYY-MM-DD HH:MM:SS" and ISO "YYYY-MM-DDTHH:MM:SS"."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace(" ", "T"))
    except (ValueError, TypeError):
        return None


def time_cyclical_features(starts_at: Optional[str | datetime]) -> list[float]:
    """
    Returns [hour_sin, hour_cos, dow_sin, dow_cos, is_weekend].
    Uses neutral defaults when starts_at is missing/unparseable.
    """
    dt = parse_datetime(starts_at)
    if dt is None:
        return [0.0, 1.0, 0.0, 1.0, 0.0]

    hour = dt.hour + (dt.minute / 60.0)
    dow = dt.weekday()  # 0=Mon ... 6=Sun

    hour_angle = 2.0 * math.pi * (hour / 24.0)
    dow_angle = 2.0 * math.pi * (dow / 7.0)

    is_weekend = 1.0 if dow >= 5 else 0.0
    return [
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(dow_angle),
        math.cos(dow_angle),
        is_weekend,
    ]


def calculate_age(birthdate) -> float | None:
    if birthdate is None:
        return None
    today = date.today()
    if isinstance(birthdate, str):
        birthdate = datetime.strptime(birthdate, "%Y-%m-%d").date()
    age = today.year - birthdate.year - (
        (today.month, today.day) < (birthdate.month, birthdate.day)
    )
    return float(age)


def days_until(starts_at) -> Optional[int]:
    """Days from today to starts_at. Negative = past. None if missing/unparseable."""
    dt = parse_datetime(starts_at)
    if dt is None:
        return None
    return (dt.date() - date.today()).days
