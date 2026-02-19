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
EMBED_DIM = 64
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

# ─── Entity types ──────────────────────────────────────────────────────────────
ENTITY_TYPES = ["user", "event", "space"]
ENTITY_TYPE_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(ENTITY_TYPES)}
NUM_ENTITY_TYPES = len(ENTITY_TYPES)  # 3

# ─── Feature vector layout ─────────────────────────────────────────────────────
# [0:40]   tag weights/flags (NUM_TAGS)
# [40]     age normalized 0-1
# [41:44]  entity_type one-hot (NUM_ENTITY_TYPES)
# [44]     popularity normalized 0-1
# total = 45
FEATURE_DIM = NUM_TAGS + 1 + NUM_ENTITY_TYPES + 1  # 45

# Age normalization range
AGE_MIN = 18.0
AGE_MAX = 65.0

# ─── Paths ─────────────────────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH = "model_weights.pt"
