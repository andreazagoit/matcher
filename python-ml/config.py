import os

# Load .env file if present (for local development)
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

# ─── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/matcher")

# ─── Model ─────────────────────────────────────────────────────────────────────
EMBED_DIM = 256
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 256
NEGATIVE_SAMPLES = 5       # negatives per positive interaction
DROPOUT = 0.2

# ─── Tag vocabulary (must match lib/models/tags/data.ts) ───────────────────────
TAG_VOCAB: list[str] = [
    # outdoor
    "trekking", "camping", "climbing", "cycling", "beach", "mountains", "gardening",
    # culture
    "cinema", "theater", "live_music", "museums", "reading", "photography", "art",
    # food
    "cooking", "restaurants", "wine", "craft_beer", "street_food", "coffee",
    # sports
    "running", "gym", "yoga", "swimming", "football", "tennis", "padel", "basketball",
    # creative
    "music", "drawing", "writing", "diy", "gaming", "coding",
    # social
    "travel", "volunteering", "languages", "pets", "parties", "board_games",
]

TAG_TO_IDX: dict[str, int] = {tag: i for i, tag in enumerate(TAG_VOCAB)}
NUM_TAGS = len(TAG_VOCAB)  # 40

# ─── Profile enum vocabularies ────────────────────────────────────────────────
# Must stay in sync with lib/models/users/schema.ts pgEnums

GENDER_VOCAB = ["man", "woman", "non_binary"]
REL_INTENT_VOCAB = ["serious_relationship", "casual_dating", "friendship", "chat"]
SMOKING_VOCAB = ["never", "sometimes", "regularly"]
DRINKING_VOCAB = ["never", "sometimes", "regularly"]
ACTIVITY_VOCAB = ["sedentary", "light", "moderate", "active", "very_active"]

GENDER_TO_IDX: dict[str, int]       = {v: i for i, v in enumerate(GENDER_VOCAB)}
REL_INTENT_TO_IDX: dict[str, int]   = {v: i for i, v in enumerate(REL_INTENT_VOCAB)}
SMOKING_TO_IDX: dict[str, int]      = {v: i for i, v in enumerate(SMOKING_VOCAB)}
DRINKING_TO_IDX: dict[str, int]     = {v: i for i, v in enumerate(DRINKING_VOCAB)}
ACTIVITY_TO_IDX: dict[str, int]     = {v: i for i, v in enumerate(ACTIVITY_VOCAB)}

# ─── Feature vector layouts ────────────────────────────────────────────────────
#
# User (USER_DIM = 60):
#   [0:40]  tag weights            (NUM_TAGS)
#   [40]    age norm               (1)
#   [41:44] gender one-hot         (3)
#   [44:48] rel_intent multi-hot   (4)
#   [48:51] smoking one-hot        (3)
#   [51:54] drinking one-hot       (3)
#   [54:59] activity one-hot       (5)
#   [59]    interaction count norm (1) — events attended + spaces joined
#
# Event (EVENT_DIM = 51):
#   [0:40]  tags multi-hot         (NUM_TAGS)
#   [40]    avg attendee age norm  (1)
#   [41]    attendee count norm    (1)
#   [42]    days until event norm  (1)
#   [43]    capacity fill rate     (1) — attendee_count / max_attendees, 0.5 if unbounded
#   [44]    is_paid                (1) — 1.0 if price > 0, else 0.0
#   [45]    price log norm         (1) — normalized log1p(price_cents)
#   [46:48] start hour cyclical    (2) — sin/cos(hour of day)
#   [48:50] start weekday cyclical (2) — sin/cos(day of week)
#   [50]    is_weekend             (1)
#
# Space (SPACE_DIM = 43):
#   [0:40]  tags multi-hot         (NUM_TAGS)
#   [40]    avg member age norm    (1)
#   [41]    member count norm      (1)
#   [42]    event count norm       (1) — events in this space

USER_DIM  = NUM_TAGS + 1 + len(GENDER_VOCAB) + len(REL_INTENT_VOCAB) + len(SMOKING_VOCAB) + len(DRINKING_VOCAB) + len(ACTIVITY_VOCAB) + 1  # 60
EVENT_DIM = NUM_TAGS + 1 + 1 + 1 + 1 + 1 + 1 + 5  # tags + age + count + days + fill + paid + price + time(5) = 51
SPACE_DIM = NUM_TAGS + 1 + 1 + 1                    # tags + age + member_count + event_count = 43

# Entity types
ENTITY_TYPES = ["user", "event", "space"]
ENTITY_TYPE_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(ENTITY_TYPES)}

# Age normalization range
AGE_MIN = 18.0
AGE_MAX = 65.0

# ─── Paths ─────────────────────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights.pt")
TRAINING_DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training-data")
