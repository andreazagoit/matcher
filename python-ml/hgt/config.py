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
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 256
NEGATIVE_SAMPLES = 5       # negatives per positive interaction
DROPOUT = 0.1
NUM_WORKERS = 0          # 0 = single-threaded (safest on MPS); set to 2-4 on CUDA

# ─── Tag vocabulary (must match lib/models/tags/data.ts) ───────────────────────
# Order must stay in sync: same categories, same per-category order.
TAG_VOCAB: list[str] = [
    # outdoor (20)
    "trekking", "camping", "climbing", "cycling", "beach", "mountains",
    "gardening", "surfing", "skiing", "kayaking", "fishing", "trail_running",
    "snowboarding", "skateboarding", "horse_riding", "sailing", "scuba_diving",
    "paragliding", "bouldering", "canyoning",
    # culture (20)
    "cinema", "theater", "live_music", "museums", "reading", "photography",
    "art", "opera", "ballet", "comedy_shows", "podcasts", "architecture",
    "vintage", "anime", "comics", "street_art", "literary_clubs", "calligraphy",
    "sculpture", "ceramics",
    # food_drink (20)
    "cooking", "restaurants", "wine", "craft_beer", "street_food", "coffee",
    "veganism", "sushi", "cocktails", "baking", "tea", "barbecue",
    "food_festivals", "cheese", "sake", "mixology", "foraging", "fermentation",
    "ramen", "pastry",
    # sports (20)
    "running", "gym", "yoga", "swimming", "football", "tennis", "padel",
    "basketball", "volleyball", "boxing", "martial_arts", "pilates", "crossfit",
    "golf", "rugby", "archery", "dance", "rowing", "hockey", "badminton",
    # creative (20)
    "music", "drawing", "writing", "diy", "gaming", "coding", "painting",
    "pottery", "knitting", "woodworking", "film_making", "singing", "djing",
    "digital_art", "cosplay", "jewelry_making", "embroidery", "leathercraft",
    "printmaking", "glass_blowing",
    # wellness (20)
    "meditation", "mindfulness", "journaling", "spa", "breathwork", "nutrition",
    "therapy", "cold_exposure", "sound_healing", "ayurveda", "reiki",
    "stretching", "sleep_hygiene", "herbal_medicine", "qi_gong", "forest_bathing",
    "intermittent_fasting", "positive_psychology", "aromatherapy",
    "somatic_practices",
    # tech_science (20)
    "ai", "blockchain", "cybersecurity", "robotics", "vr_ar", "open_source",
    "data_science", "smart_home", "esports", "astronomy", "chemistry", "biology",
    "engineering", "drones", "quantum_computing", "3d_printing",
    "space_exploration", "neuroscience", "environmental_science", "biohacking",
    # music_genres (20)
    "jazz", "classical_music", "hip_hop", "electronic_music", "rock",
    "indie_music", "reggae", "pop_music", "metal", "country_music", "blues",
    "r_and_b", "folk_music", "gospel", "bossa_nova", "afrobeat", "techno",
    "house_music", "punk", "soul",
    # social (20)
    "travel", "volunteering", "languages", "pets", "parties", "board_games",
    "networking", "karaoke", "escape_rooms", "trivia_nights", "activism",
    "mentoring", "astrology", "tarot", "stand_up", "improv", "storytelling",
    "cultural_exchange", "language_exchange", "community_events",
    # lifestyle (20)
    "fashion", "interior_design", "sustainability", "minimalism", "van_life",
    "urban_exploration", "thrifting", "luxury_lifestyle", "tattoos",
    "personal_growth", "entrepreneurship", "parenting", "spirituality",
    "digital_nomad", "homesteading", "zero_waste", "slow_living", "nightlife",
    "brunch_culture", "self_improvement",
]

TAG_TO_IDX: dict[str, int] = {tag: i for i, tag in enumerate(TAG_VOCAB)}
NUM_TAGS = len(TAG_VOCAB)  # 200
TAG_DIM  = NUM_TAGS        # tag node feature size (identity matrix)

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

# ─── Feature vector layouts ────────────────────────────────────────────────────## Model Dimension Layouts
#
# User (USER_DIM = NUM_TAGS + 19 = 59):
#   [0:NUM_TAGS]  tags multi-hot         (NUM_TAGS)
#   [NUM_TAGS]    age norm               (1)
#   [+1:+4]       gender one-hot         (3)
#   [+4:+8]       rel_intent multi-hot   (4)
#   [+8:+11]      smoking one-hot        (3)
#   [+11:+14]     drinking one-hot       (3)
#   [+14:+19]     activity one-hot       (5)
USER_DIM  = NUM_TAGS + 19
#
# Event (EVENT_DIM = NUM_TAGS + 11 = 211):
#   [0:NUM_TAGS]  tags multi-hot         (NUM_TAGS)
#   [NUM_TAGS]    avg attendee age norm  (1)
#   [+1]          attendee count norm    (1)
#   [+2]          days until event norm  (1)
#   [+3]          capacity fill rate     (1)
#   [+4]          is_paid                (1)
#   [+5]          price log norm         (1)
#   [+6:+8]       start hour cyclical    (2)
#   [+8:+10]      start weekday cyclical (2)
#   [+10]         is_weekend             (1)
#
# Space (SPACE_DIM = NUM_TAGS + 3 = 203):
#   [0:NUM_TAGS]  tags multi-hot         (NUM_TAGS)
#   [NUM_TAGS]    avg member age norm    (1)
#   [+1]          member count norm      (1)
#   [+2]          event count norm       (1)

USER_DIM  = NUM_TAGS + 1 + len(GENDER_VOCAB) + len(REL_INTENT_VOCAB) + len(SMOKING_VOCAB) + len(DRINKING_VOCAB) + len(ACTIVITY_VOCAB) + 1 # +1 for num_tags
EVENT_DIM = NUM_TAGS + 1 + 1 + 1 + 1 + 1 + 1 + 5 + 1 # +1 for num_tags
SPACE_DIM = NUM_TAGS + 1 + 1 + 1                    # tags + age + member_count + event_count

# Entity types
ENTITY_TYPES = ["user", "event", "space"]
ENTITY_TYPE_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(ENTITY_TYPES)}

# Age normalization range
AGE_MIN = 18.0
AGE_MAX = 65.0

# ─── Graph topology (HGTConv metadata) ────────────────────────────────────────
NODE_TYPES: list[str] = ["user", "event", "space", "tag"]

EDGE_TYPES: list[tuple[str, str, str]] = [
    # behavioural edges
    ("user",  "attends",                "event"),
    ("event", "rev_attends",            "user"),
    ("user",  "joins",                  "space"),
    ("space", "rev_joins",              "user"),
    ("event", "hosted_by",              "space"),
    ("space", "rev_hosted_by",          "event"),
    ("user",  "similar_to",             "user"),
    # tag edges
    ("user",  "likes",                  "tag"),
    ("tag",   "rev_likes",              "user"),
    ("event", "tagged_with",            "tag"),
    ("tag",   "rev_tagged_with_event",  "event"),
    ("space", "tagged_with_space",      "tag"),
    ("tag",   "rev_tagged_with_space",  "space"),
]

METADATA: tuple = (NODE_TYPES, EDGE_TYPES)

HGT_HEADS  = 4
HGT_LAYERS = 3

# ─── Paths ─────────────────────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH  = os.path.join(_SERVICE_DIR, "hgt_weights.pt")
TRAINING_DATA_DIR   = os.path.join(_SERVICE_DIR, "training-data")
