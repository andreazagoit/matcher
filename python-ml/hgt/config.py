import os

# Root of the python-ml/ service directory (one level up from this file's ml/ package).
_SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env file if present (for local development)
_env_path = os.path.join(_SERVICE_DIR, ".env")
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
EMBED_DIM  = 256
HIDDEN_DIM = 256   # must be ≥ EMBED_DIM: encoder compresses → HGT operates → proj scales down
LEARNING_RATE = 5e-4
EPOCHS = 100
BATCH_SIZE = 256
NEGATIVE_SAMPLES = 5       # negatives per positive interaction
DROPOUT = 0.2
NUM_WORKERS = 0          # 0 = single-threaded (safest on MPS); set to 2-4 on CUDA

CATEGORY_EMBED_DIM = 64  # text-embedding-3-small output dimension

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
# Model Dimension Layouts
#
# User (USER_DIM = 18):
#   Category preferences are NOT encoded in the node vector — they live
#   entirely in the graph as user→likes_category edges. The node vector
#   contains only demographic / behavioural profile data.
#   [0]    age norm                        (1)
#   [1:4]  gender one-hot                  (3)
#   [4:8]  rel_intent multi-hot            (4)
#   [8:11] smoking one-hot                 (3)
#   [11:14] drinking one-hot               (3)
#   [14:19] activity one-hot               (5)
USER_DIM  = 1 + len(GENDER_VOCAB) + len(REL_INTENT_VOCAB) + len(SMOKING_VOCAB) + len(DRINKING_VOCAB) + len(ACTIVITY_VOCAB)

# Event (EVENT_DIM = CATEGORY_EMBED_DIM + 11 = 75):
#   [0:CATEGORY_EMBED_DIM] categories mean-pool    (64)
#   [CATEGORY_EMBED_DIM]   avg attendee age norm   (1)
#   [+1]                   attendee count norm     (1)
#   [+2]                   days until event norm   (1)
#   [+3]                   capacity fill rate      (1)
#   [+4]                   is_paid                 (1)
#   [+5]                   price log norm          (1)
#   [+6:+8]                start hour cyclical     (2)
#   [+8:+10]               start weekday cyclical  (2)
#   [+10]                  is_weekend              (1)
EVENT_DIM = CATEGORY_EMBED_DIM + 1 + 1 + 1 + 1 + 1 + 1 + 5

# Space (SPACE_DIM = CATEGORY_EMBED_DIM + 3 = 67):
#   [0:CATEGORY_EMBED_DIM] categories mean-pool    (64)
#   [CATEGORY_EMBED_DIM]   avg member age norm     (1)
#   [+1]                   member count norm       (1)
#   [+2]                   event count norm        (1)
SPACE_DIM = CATEGORY_EMBED_DIM + 1 + 1 + 1

# Category node: 64D dense embedding from text-embedding-3-small
CATEGORY_DIM = CATEGORY_EMBED_DIM

# Entity types
ENTITY_TYPES = ["user", "event", "space"]
ENTITY_TYPE_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(ENTITY_TYPES)}

# Age normalization range
AGE_MIN = 18.0
AGE_MAX = 65.0

# ─── Graph topology (HGTConv metadata) ────────────────────────────────────────
NODE_TYPES: list[str] = ["user", "event", "space", "category"]

EDGE_TYPES: list[tuple[str, str, str]] = [
    # behavioural edges
    ("user",     "attends",                   "event"),
    ("event",    "rev_attends",               "user"),
    ("user",     "joins",                     "space"),
    ("space",    "rev_joins",                 "user"),
    ("event",    "hosted_by",                 "space"),
    ("space",    "rev_hosted_by",             "event"),
    ("user",     "similar_to",               "user"),
    # category edges
    ("user",     "likes_category",            "category"),
    ("category", "rev_likes_category",        "user"),
    ("event",    "tagged_with",               "category"),
    ("category", "rev_tagged_with_event",     "event"),
    ("space",    "tagged_with_space",         "category"),
    ("category", "rev_tagged_with_space",     "space"),
    # structural similarity edges (bidirectional, used for message passing + BPR regularisation)
    ("event",    "similarity",               "event"),
    ("space",    "similarity",               "space"),
]

METADATA: tuple = (NODE_TYPES, EDGE_TYPES)

HGT_HEADS  = 4
HGT_LAYERS = 3

# ─── Paths ─────────────────────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH  = os.path.join(_SERVICE_DIR, "hgt_weights.pt")
TRAINING_DATA_DIR   = os.path.join(_SERVICE_DIR, "training-data")
