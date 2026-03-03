"""
Synthetic training data generator.

Produces JSON files in training-data/ that mirror 'npm run ml:export'.

─── Default volumes ──────────────────────────────────────────────────────────
Change the four constants below to adjust dataset size.
"""

from __future__ import annotations

# ┌─────────────────────────────────────────────────────────────────────────────
# │ CONFIGURE HERE
N_USERS        = 10_000
N_EVENTS       = 1_400
N_SPACES       = 700
N_INTERACTIONS = 210_000
# └─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import math
import os
import random
import uuid
from collections import defaultdict
from datetime import date, timedelta

from hgt.config import TRAINING_DATA_DIR
from openai import OpenAI
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

_oai_client = None

def get_openai_client():
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI()
    return _oai_client


# ─── Canonical categories ──────────────────────────────────────────────────────
#
# These are the 24 category IDs used throughout the app.
# All simulation logic works directly with these IDs.

ALL_CATEGORY_IDS: list[str] = [
    "sport", "outdoor", "music", "art", "food", "travel", "wellness", "tech",
    "culture", "cinema", "social", "animals", "fashion", "sustainability",
    "entrepreneurship", "science", "spirituality", "volunteering", "nightlife",
    "photography", "dance", "crafts", "languages", "comedy",
]

# Category index for fast lookup
CAT_INDEX: dict[str, int] = {c: i for i, c in enumerate(ALL_CATEGORY_IDS)}
N_CATS = len(ALL_CATEGORY_IDS)

# Cluster groups: each cluster is a list of category IDs that belong together.
# Used to assign persona affinities and sort items by relevance.
CLUSTERS: list[list[str]] = [
    ["outdoor", "sport"],                                           # 0 outdoor/sport
    ["culture", "cinema", "art", "photography"],                   # 1 culture
    ["food"],                                                       # 2 food
    ["sport", "dance"],                                             # 3 sports/dance
    ["music", "art", "crafts"],                                     # 4 creative
    ["wellness", "spirituality"],                                   # 5 wellness
    ["tech", "science"],                                            # 6 tech/science
    ["music", "nightlife"],                                         # 7 music/nightlife
    ["social", "travel", "volunteering", "languages", "comedy"],   # 8 social
    ["fashion", "sustainability", "entrepreneurship", "nightlife"], # 9 lifestyle
]
N_CLUSTERS = len(CLUSTERS)

# category_id → set of cluster indices it belongs to
CAT_TO_CLUSTERS: dict[str, list[int]] = defaultdict(list)
for _ci, _cats in enumerate(CLUSTERS):
    for _cat in _cats:
        CAT_TO_CLUSTERS[_cat].append(_ci)


# ─── Personas ─────────────────────────────────────────────────────────────────
#
# Each persona represents an archetypal user type.
# `category_pool` lists categories this persona primarily engages with.
# `primary_categories` (first 2–4) carry the highest affinity weight.

PERSONAS = [
    # ── cluster 0: outdoor ───────────────────────────────────────────────────
    {
        "name":               "outdoor_adventurer",
        "cluster":            0,
        "primary_categories": ["outdoor", "sport", "travel"],
        "category_pool":      ["outdoor", "sport", "travel", "photography", "animals", "wellness"],
        "age_range":          (22, 32),
        "gender_w":           [0.40, 0.50, 0.10],
        "rel_intent_w":       [0.25, 0.40, 0.25, 0.10],
        "smoking_w":          [0.88, 0.10, 0.02],
        "drinking_w":         [0.25, 0.60, 0.15],
        "activity_choices":   ["active", "very_active"],
        "activity_w":         [0.35, 0.65],
    },
    # ── cluster 1: culture ───────────────────────────────────────────────────
    {
        "name":               "visual_culture_fan",
        "cluster":            1,
        "primary_categories": ["cinema", "photography", "art"],
        "category_pool":      ["cinema", "photography", "art", "culture", "travel", "crafts"],
        "age_range":          (20, 38),
        "gender_w":           [0.38, 0.48, 0.14],
        "rel_intent_w":       [0.32, 0.30, 0.28, 0.10],
        "smoking_w":          [0.52, 0.34, 0.14],
        "drinking_w":         [0.18, 0.54, 0.28],
        "activity_choices":   ["light", "moderate"],
        "activity_w":         [0.45, 0.55],
    },
    {
        "name":               "performing_arts_fan",
        "cluster":            1,
        "primary_categories": ["culture", "music", "art"],
        "category_pool":      ["culture", "music", "art", "cinema", "comedy", "social"],
        "age_range":          (28, 48),
        "gender_w":           [0.30, 0.60, 0.10],
        "rel_intent_w":       [0.45, 0.20, 0.25, 0.10],
        "smoking_w":          [0.50, 0.36, 0.14],
        "drinking_w":         [0.12, 0.50, 0.38],
        "activity_choices":   ["light", "moderate"],
        "activity_w":         [0.42, 0.58],
    },
    {
        "name":               "literary_craft_fan",
        "cluster":            1,
        "primary_categories": ["art", "crafts", "culture"],
        "category_pool":      ["art", "crafts", "culture", "photography", "wellness", "languages"],
        "age_range":          (26, 45),
        "gender_w":           [0.32, 0.58, 0.10],
        "rel_intent_w":       [0.42, 0.22, 0.28, 0.08],
        "smoking_w":          [0.58, 0.28, 0.14],
        "drinking_w":         [0.15, 0.55, 0.30],
        "activity_choices":   ["sedentary", "light", "moderate"],
        "activity_w":         [0.22, 0.50, 0.28],
    },
    # ── cluster 2: food ──────────────────────────────────────────────────────
    {
        "name":               "foodie",
        "cluster":            2,
        "primary_categories": ["food", "travel"],
        "category_pool":      ["food", "travel", "social", "sustainability", "culture", "nightlife"],
        "age_range":          (24, 40),
        "gender_w":           [0.40, 0.50, 0.10],
        "rel_intent_w":       [0.35, 0.30, 0.25, 0.10],
        "smoking_w":          [0.60, 0.28, 0.12],
        "drinking_w":         [0.08, 0.47, 0.45],
        "activity_choices":   ["light", "moderate"],
        "activity_w":         [0.50, 0.50],
    },
    # ── cluster 3: sports ────────────────────────────────────────────────────
    {
        "name":               "team_sports_fan",
        "cluster":            3,
        "primary_categories": ["sport", "social"],
        "category_pool":      ["sport", "social", "travel", "outdoor", "nightlife", "dance"],
        "age_range":          (18, 30),
        "gender_w":           [0.68, 0.28, 0.04],
        "rel_intent_w":       [0.18, 0.48, 0.24, 0.10],
        "smoking_w":          [0.90, 0.08, 0.02],
        "drinking_w":         [0.28, 0.58, 0.14],
        "activity_choices":   ["active", "very_active"],
        "activity_w":         [0.30, 0.70],
    },
    {
        "name":               "fitness_enthusiast",
        "cluster":            3,
        "primary_categories": ["sport", "wellness"],
        "category_pool":      ["sport", "wellness", "outdoor", "dance", "spirituality", "food"],
        "age_range":          (20, 35),
        "gender_w":           [0.50, 0.45, 0.05],
        "rel_intent_w":       [0.25, 0.40, 0.25, 0.10],
        "smoking_w":          [0.95, 0.04, 0.01],
        "drinking_w":         [0.35, 0.52, 0.13],
        "activity_choices":   ["active", "very_active"],
        "activity_w":         [0.28, 0.72],
    },
    # ── cluster 4: creative ──────────────────────────────────────────────────
    {
        "name":               "digital_creative",
        "cluster":            4,
        "primary_categories": ["music", "tech", "art"],
        "category_pool":      ["music", "tech", "art", "cinema", "nightlife", "crafts"],
        "age_range":          (18, 34),
        "gender_w":           [0.48, 0.38, 0.14],
        "rel_intent_w":       [0.18, 0.38, 0.32, 0.12],
        "smoking_w":          [0.50, 0.34, 0.16],
        "drinking_w":         [0.18, 0.52, 0.30],
        "activity_choices":   ["sedentary", "light", "moderate"],
        "activity_w":         [0.28, 0.48, 0.24],
    },
    {
        "name":               "craft_creative",
        "cluster":            4,
        "primary_categories": ["crafts", "art", "music"],
        "category_pool":      ["crafts", "art", "music", "photography", "fashion", "culture"],
        "age_range":          (22, 40),
        "gender_w":           [0.22, 0.60, 0.18],
        "rel_intent_w":       [0.28, 0.30, 0.32, 0.10],
        "smoking_w":          [0.40, 0.40, 0.20],
        "drinking_w":         [0.15, 0.55, 0.30],
        "activity_choices":   ["sedentary", "light", "moderate"],
        "activity_w":         [0.18, 0.52, 0.30],
    },
    # ── cluster 5: wellness ──────────────────────────────────────────────────
    {
        "name":               "wellness_seeker",
        "cluster":            5,
        "primary_categories": ["wellness", "spirituality"],
        "category_pool":      ["wellness", "spirituality", "sport", "food", "animals", "volunteering"],
        "age_range":          (26, 44),
        "gender_w":           [0.22, 0.65, 0.13],
        "rel_intent_w":       [0.38, 0.25, 0.28, 0.09],
        "smoking_w":          [0.90, 0.08, 0.02],
        "drinking_w":         [0.35, 0.52, 0.13],
        "activity_choices":   ["light", "moderate", "active"],
        "activity_w":         [0.25, 0.50, 0.25],
    },
    # ── cluster 6: tech ──────────────────────────────────────────────────────
    {
        "name":               "tech_geek",
        "cluster":            6,
        "primary_categories": ["tech", "science"],
        "category_pool":      ["tech", "science", "entrepreneurship", "music", "crafts", "social"],
        "age_range":          (20, 38),
        "gender_w":           [0.70, 0.22, 0.08],
        "rel_intent_w":       [0.22, 0.38, 0.28, 0.12],
        "smoking_w":          [0.78, 0.17, 0.05],
        "drinking_w":         [0.28, 0.55, 0.17],
        "activity_choices":   ["sedentary", "light", "moderate"],
        "activity_w":         [0.30, 0.45, 0.25],
    },
    # ── cluster 7: music/nightlife ───────────────────────────────────────────
    {
        "name":               "music_lover",
        "cluster":            7,
        "primary_categories": ["music", "nightlife"],
        "category_pool":      ["music", "nightlife", "dance", "art", "cinema", "social"],
        "age_range":          (18, 36),
        "gender_w":           [0.45, 0.45, 0.10],
        "rel_intent_w":       [0.22, 0.40, 0.28, 0.10],
        "smoking_w":          [0.45, 0.38, 0.17],
        "drinking_w":         [0.10, 0.50, 0.40],
        "activity_choices":   ["light", "moderate", "active"],
        "activity_w":         [0.35, 0.45, 0.20],
    },
    # ── cluster 8: social ────────────────────────────────────────────────────
    {
        "name":               "social_butterfly",
        "cluster":            8,
        "primary_categories": ["social", "travel", "comedy"],
        "category_pool":      ["social", "travel", "comedy", "languages", "food", "nightlife"],
        "age_range":          (20, 34),
        "gender_w":           [0.38, 0.52, 0.10],
        "rel_intent_w":       [0.18, 0.35, 0.32, 0.15],
        "smoking_w":          [0.48, 0.35, 0.17],
        "drinking_w":         [0.05, 0.43, 0.52],
        "activity_choices":   ["moderate", "active"],
        "activity_w":         [0.55, 0.45],
    },
    # ── cluster 9: lifestyle ─────────────────────────────────────────────────
    {
        "name":               "lifestyle_explorer",
        "cluster":            9,
        "primary_categories": ["sustainability", "fashion", "entrepreneurship"],
        "category_pool":      ["sustainability", "fashion", "entrepreneurship", "travel", "volunteering", "languages"],
        "age_range":          (22, 40),
        "gender_w":           [0.30, 0.58, 0.12],
        "rel_intent_w":       [0.28, 0.33, 0.28, 0.11],
        "smoking_w":          [0.55, 0.32, 0.13],
        "drinking_w":         [0.12, 0.50, 0.38],
        "activity_choices":   ["light", "moderate", "active"],
        "activity_w":         [0.30, 0.50, 0.20],
    },
]

REL_INTENTS = ["serious_relationship", "casual_dating", "friendship", "chat"]
GENDERS     = ["man", "woman", "non_binary"]
SMOKING     = ["never", "sometimes", "regularly"]
DRINKING    = ["never", "sometimes", "regularly"]

# ─── Space archetypes ──────────────────────────────────────────────────────────
# Each archetype carries a `category_pool` (canonical category IDs) and a
# `cluster` (for persona affinity matching).

SPACE_ARCHETYPES: list[dict] = [
    {"name": "nerd_hub",                  "cluster": 4, "category_pool": ["tech", "music", "art", "crafts", "social"]},
    {"name": "board_games_society",       "cluster": 5, "category_pool": ["social", "comedy", "food", "crafts"]},
    {"name": "indie_gaming_club",         "cluster": 4, "category_pool": ["tech", "music", "nightlife", "social"]},
    {"name": "tech_founders_circle",      "cluster": 4, "category_pool": ["tech", "entrepreneurship", "languages", "science"]},
    {"name": "ai_builders_lab",           "cluster": 4, "category_pool": ["tech", "science", "entrepreneurship"]},
    {"name": "cinephile_collective",      "cluster": 1, "category_pool": ["cinema", "culture", "art", "photography"]},
    {"name": "book_cafe_club",            "cluster": 1, "category_pool": ["culture", "art", "crafts", "languages"]},
    {"name": "street_photo_crew",         "cluster": 1, "category_pool": ["photography", "art", "travel", "culture"]},
    {"name": "modern_art_collective",     "cluster": 1, "category_pool": ["art", "photography", "culture", "crafts"]},
    {"name": "live_music_tribe",          "cluster": 4, "category_pool": ["music", "nightlife", "art", "social"]},
    {"name": "foodie_circle",             "cluster": 2, "category_pool": ["food", "travel", "social", "sustainability"]},
    {"name": "wine_tasting_society",      "cluster": 2, "category_pool": ["food", "social", "nightlife", "culture"]},
    {"name": "coffee_explorers",          "cluster": 2, "category_pool": ["food", "social", "travel", "crafts"]},
    {"name": "urban_runners",             "cluster": 3, "category_pool": ["sport", "outdoor", "wellness", "social"]},
    {"name": "yoga_wellness",             "cluster": 3, "category_pool": ["wellness", "sport", "spirituality", "dance"]},
    {"name": "mountain_hikers",           "cluster": 0, "category_pool": ["outdoor", "sport", "travel", "photography"]},
    {"name": "climbing_crew",             "cluster": 0, "category_pool": ["outdoor", "sport", "social"]},
    {"name": "travel_backpackers",        "cluster": 8, "category_pool": ["travel", "languages", "photography", "social"]},
    {"name": "pet_lovers_club",           "cluster": 8, "category_pool": ["animals", "volunteering", "travel", "social"]},
    {"name": "language_exchange_lounge",  "cluster": 8, "category_pool": ["languages", "culture", "social", "travel"]},
    {"name": "wellness_studio",           "cluster": 5, "category_pool": ["wellness", "spirituality", "sport", "dance"]},
    {"name": "mindfulness_center",        "cluster": 5, "category_pool": ["spirituality", "wellness", "culture"]},
    {"name": "holistic_health_hub",       "cluster": 5, "category_pool": ["wellness", "food", "sustainability", "spirituality"]},
    {"name": "maker_space",               "cluster": 6, "category_pool": ["tech", "science", "crafts", "art"]},
    {"name": "ai_research_lab",           "cluster": 6, "category_pool": ["tech", "science", "entrepreneurship"]},
    {"name": "astronomy_club",            "cluster": 6, "category_pool": ["science", "outdoor", "photography"]},
    {"name": "music_production_studio",   "cluster": 7, "category_pool": ["music", "nightlife", "dance", "art"]},
    {"name": "jazz_lounge",               "cluster": 7, "category_pool": ["music", "nightlife", "culture", "social"]},
    {"name": "vinyl_collectors_club",     "cluster": 7, "category_pool": ["music", "culture", "crafts"]},
    {"name": "debate_society",            "cluster": 8, "category_pool": ["social", "languages", "culture", "volunteering"]},
    {"name": "improv_theater_group",      "cluster": 8, "category_pool": ["comedy", "culture", "social", "art"]},
    {"name": "sustainable_living_co",     "cluster": 9, "category_pool": ["sustainability", "food", "volunteering", "crafts"]},
    {"name": "digital_nomad_hub",         "cluster": 9, "category_pool": ["entrepreneurship", "tech", "travel", "languages"]},
    {"name": "fashion_collective",        "cluster": 9, "category_pool": ["fashion", "photography", "art", "crafts"]},
    {"name": "anime_comics_club",         "cluster": 1, "category_pool": ["culture", "cinema", "art", "social"]},
    {"name": "photo_cinema_society",      "cluster": 1, "category_pool": ["photography", "cinema", "culture", "art"]},
    {"name": "theater_company",           "cluster": 1, "category_pool": ["culture", "music", "comedy", "art"]},
    {"name": "ceramics_pottery_studio",   "cluster": 1, "category_pool": ["crafts", "art", "culture", "photography"]},
    {"name": "creative_writing_circle",   "cluster": 1, "category_pool": ["crafts", "culture", "languages", "art"]},
    {"name": "team_sports_club",          "cluster": 3, "category_pool": ["sport", "social", "outdoor"]},
    {"name": "racket_sports_club",        "cluster": 3, "category_pool": ["sport", "social", "wellness"]},
    {"name": "combat_sports_gym",         "cluster": 3, "category_pool": ["sport", "wellness", "social"]},
    {"name": "cosplay_gaming_crew",       "cluster": 4, "category_pool": ["culture", "music", "tech", "social"]},
    {"name": "film_dj_collective",        "cluster": 4, "category_pool": ["music", "cinema", "nightlife", "art"]},
    {"name": "craft_atelier",             "cluster": 4, "category_pool": ["crafts", "art", "fashion", "photography"]},
    {"name": "woodwork_print_studio",     "cluster": 4, "category_pool": ["crafts", "art", "culture"]},
]

EVENT_ARCHETYPES: list[dict] = [
    {"name": "board_game_night",          "cluster": 5, "category_pool": ["social", "comedy", "food"],         "price_choices": [0, 500, 1000],    "hour_choices": [19, 20, 21], "max_choices": [12, 20, 30]},
    {"name": "esports_tournament",        "cluster": 4, "category_pool": ["tech", "music", "social"],           "price_choices": [0, 1000, 1500],   "hour_choices": [18, 19, 20], "max_choices": [20, 40, 80]},
    {"name": "hackathon",                 "cluster": 4, "category_pool": ["tech", "entrepreneurship", "science"], "price_choices": [0, 2000, 3000], "hour_choices": [9, 10],      "max_choices": [30, 60, 120]},
    {"name": "ai_workshop",               "cluster": 6, "category_pool": ["tech", "science"],                    "price_choices": [0, 1500, 2500],  "hour_choices": [18, 19],     "max_choices": [20, 40, 60]},
    {"name": "startup_pitch_night",       "cluster": 9, "category_pool": ["entrepreneurship", "tech", "social"], "price_choices": [0, 1000, 2000], "hour_choices": [18, 19, 20], "max_choices": [20, 40, 80]},
    {"name": "movie_screening",           "cluster": 1, "category_pool": ["cinema", "culture", "social"],        "price_choices": [0, 700, 1200],   "hour_choices": [20, 21],     "max_choices": [20, 40, 80]},
    {"name": "book_discussion",           "cluster": 1, "category_pool": ["culture", "crafts", "social"],        "price_choices": [0, 500, 1000],   "hour_choices": [18, 19, 20], "max_choices": [10, 16, 24]},
    {"name": "photo_walk",                "cluster": 1, "category_pool": ["photography", "art", "travel"],       "price_choices": [0, 1000],        "hour_choices": [9, 10, 16],  "max_choices": [12, 20, 30]},
    {"name": "art_workshop",              "cluster": 1, "category_pool": ["art", "crafts", "photography"],       "price_choices": [1000, 2000, 3000],"hour_choices": [17, 18, 19], "max_choices": [10, 16, 24]},
    {"name": "open_mic_live",             "cluster": 4, "category_pool": ["music", "nightlife", "social"],       "price_choices": [0, 1000, 1500],  "hour_choices": [20, 21],     "max_choices": [20, 40, 70]},
    {"name": "cooking_class",             "cluster": 2, "category_pool": ["food", "social", "sustainability"],   "price_choices": [1500, 2500, 3500],"hour_choices": [11, 18, 19], "max_choices": [8, 12, 20]},
    {"name": "wine_tasting",              "cluster": 2, "category_pool": ["food", "social", "culture"],          "price_choices": [2000, 3000, 5000],"hour_choices": [19, 20],     "max_choices": [12, 20, 30]},
    {"name": "coffee_cupping",            "cluster": 2, "category_pool": ["food", "social"],                     "price_choices": [0, 1000, 1500],  "hour_choices": [10, 11, 16], "max_choices": [10, 16, 24]},
    {"name": "running_session",           "cluster": 3, "category_pool": ["sport", "outdoor", "wellness"],       "price_choices": [0, 500, 1000],   "hour_choices": [7, 8, 18],   "max_choices": [12, 25, 40]},
    {"name": "yoga_session",              "cluster": 3, "category_pool": ["wellness", "sport", "spirituality"],  "price_choices": [0, 1000, 1500],  "hour_choices": [7, 8, 19],   "max_choices": [10, 20, 30]},
    {"name": "mountain_trek",             "cluster": 0, "category_pool": ["outdoor", "sport", "travel"],         "price_choices": [0, 1500, 2500],  "hour_choices": [7, 8, 9],    "max_choices": [10, 20, 30]},
    {"name": "climbing_session",          "cluster": 0, "category_pool": ["outdoor", "sport"],                   "price_choices": [1000, 2000, 3000],"hour_choices": [17, 18, 19], "max_choices": [8, 16, 24]},
    {"name": "city_trip",                 "cluster": 8, "category_pool": ["travel", "photography", "social"],    "price_choices": [0, 2000, 4000],  "hour_choices": [8, 9, 10],   "max_choices": [12, 20, 35]},
    {"name": "language_meetup",           "cluster": 8, "category_pool": ["languages", "travel", "social"],      "price_choices": [0, 500, 1000],   "hour_choices": [18, 19, 20], "max_choices": [12, 24, 40]},
    {"name": "meditation_session",        "cluster": 5, "category_pool": ["wellness", "spirituality"],           "price_choices": [0, 1000, 1500],  "hour_choices": [7, 8, 19],   "max_choices": [10, 20, 30]},
    {"name": "nutrition_workshop",        "cluster": 5, "category_pool": ["wellness", "food", "sustainability"], "price_choices": [1000, 2000, 3000],"hour_choices": [18, 19],     "max_choices": [12, 20, 30]},
    {"name": "sound_healing_circle",      "cluster": 5, "category_pool": ["spirituality", "wellness", "music"],  "price_choices": [0, 1500, 2000],  "hour_choices": [19, 20],     "max_choices": [10, 16, 24]},
    {"name": "science_night",             "cluster": 6, "category_pool": ["science", "tech", "social"],          "price_choices": [0, 1000, 1500],  "hour_choices": [19, 20],     "max_choices": [20, 40, 60]},
    {"name": "jazz_concert",              "cluster": 7, "category_pool": ["music", "nightlife", "culture"],      "price_choices": [500, 1500, 3000], "hour_choices": [20, 21],     "max_choices": [30, 60, 120]},
    {"name": "dj_workshop",               "cluster": 7, "category_pool": ["music", "nightlife", "dance"],        "price_choices": [1000, 2000, 3000],"hour_choices": [18, 19],     "max_choices": [10, 16, 24]},
    {"name": "singer_songwriter_night",   "cluster": 7, "category_pool": ["music", "culture", "social"],         "price_choices": [0, 500, 1000],   "hour_choices": [20, 21],     "max_choices": [20, 40, 60]},
    {"name": "storytelling_night",        "cluster": 8, "category_pool": ["comedy", "culture", "social"],        "price_choices": [0, 500, 1000],   "hour_choices": [19, 20, 21], "max_choices": [20, 40, 60]},
    {"name": "cultural_exchange_dinner",  "cluster": 8, "category_pool": ["languages", "food", "social"],        "price_choices": [0, 1500, 2500],  "hour_choices": [19, 20],     "max_choices": [12, 20, 30]},
    {"name": "thrift_fair",               "cluster": 9, "category_pool": ["fashion", "sustainability", "crafts"],"price_choices": [0, 500, 1000],   "hour_choices": [10, 11, 12], "max_choices": [50, 100, 200]},
    {"name": "sustainability_workshop",   "cluster": 9, "category_pool": ["sustainability", "food", "volunteering"],"price_choices": [0, 1000, 1500],"hour_choices": [10, 18, 19], "max_choices": [20, 40, 60]},
    {"name": "theater_show",              "cluster": 1, "category_pool": ["culture", "comedy", "music"],         "price_choices": [500, 1500, 3000], "hour_choices": [20, 21],     "max_choices": [30, 80, 200]},
    {"name": "ceramics_workshop",         "cluster": 1, "category_pool": ["crafts", "art", "culture"],           "price_choices": [1500, 2500, 4000],"hour_choices": [10, 17, 18], "max_choices": [6, 10, 16]},
    {"name": "writing_workshop",          "cluster": 1, "category_pool": ["crafts", "culture", "languages"],     "price_choices": [0, 1000, 2000],  "hour_choices": [18, 19],     "max_choices": [10, 20, 30]},
    {"name": "team_sports_match",         "cluster": 3, "category_pool": ["sport", "social", "outdoor"],         "price_choices": [0, 500, 1000],   "hour_choices": [9, 15, 18],  "max_choices": [10, 20, 40]},
    {"name": "martial_arts_class",        "cluster": 3, "category_pool": ["sport", "wellness", "dance"],         "price_choices": [1000, 1500, 2500],"hour_choices": [7, 18, 19],  "max_choices": [8, 16, 24]},
    {"name": "dance_class",               "cluster": 3, "category_pool": ["dance", "music", "social"],           "price_choices": [500, 1000, 2000], "hour_choices": [18, 19, 20], "max_choices": [10, 20, 30]},
    {"name": "craft_fair",                "cluster": 4, "category_pool": ["crafts", "art", "fashion"],           "price_choices": [0, 500, 1000],   "hour_choices": [10, 11],     "max_choices": [30, 60, 120]},
    {"name": "life_drawing_session",      "cluster": 4, "category_pool": ["art", "crafts", "photography"],       "price_choices": [1000, 1500, 2500],"hour_choices": [18, 19],     "max_choices": [8, 12, 20]},
    {"name": "volunteering_day",          "cluster": 8, "category_pool": ["volunteering", "sustainability", "social"], "price_choices": [0],        "hour_choices": [9, 10],      "max_choices": [20, 40, 80]},
    {"name": "comedy_night",              "cluster": 8, "category_pool": ["comedy", "social", "nightlife"],      "price_choices": [0, 500, 1500],   "hour_choices": [20, 21],     "max_choices": [30, 60, 100]},
    {"name": "animal_shelter_visit",      "cluster": 8, "category_pool": ["animals", "volunteering", "social"],  "price_choices": [0],              "hour_choices": [10, 11],     "max_choices": [10, 20, 30]},
]

SPACE_EVENT_COMPATIBILITY: dict[str, list[str]] = {
    "nerd_hub": ["esports_tournament", "hackathon", "ai_workshop", "board_game_night"],
    "board_games_society": ["board_game_night", "comedy_night", "coffee_cupping"],
    "indie_gaming_club": ["esports_tournament", "open_mic_live", "board_game_night"],
    "tech_founders_circle": ["hackathon", "ai_workshop", "startup_pitch_night", "language_meetup"],
    "ai_builders_lab": ["hackathon", "ai_workshop", "startup_pitch_night"],
    "cinephile_collective": ["movie_screening", "book_discussion", "photo_walk"],
    "book_cafe_club": ["book_discussion", "coffee_cupping", "language_meetup", "writing_workshop"],
    "street_photo_crew": ["photo_walk", "city_trip", "art_workshop"],
    "modern_art_collective": ["art_workshop", "life_drawing_session", "photo_walk"],
    "live_music_tribe": ["open_mic_live", "jazz_concert", "singer_songwriter_night"],
    "foodie_circle": ["cooking_class", "wine_tasting", "coffee_cupping", "cultural_exchange_dinner"],
    "wine_tasting_society": ["wine_tasting", "cooking_class", "open_mic_live"],
    "coffee_explorers": ["coffee_cupping", "book_discussion", "language_meetup"],
    "urban_runners": ["running_session", "city_trip", "yoga_session"],
    "yoga_wellness": ["yoga_session", "running_session", "meditation_session", "dance_class"],
    "mountain_hikers": ["mountain_trek", "city_trip", "climbing_session"],
    "climbing_crew": ["climbing_session", "mountain_trek", "running_session"],
    "travel_backpackers": ["city_trip", "language_meetup", "photo_walk"],
    "pet_lovers_club": ["animal_shelter_visit", "volunteering_day", "city_trip"],
    "language_exchange_lounge": ["language_meetup", "cultural_exchange_dinner", "book_discussion"],
    "wellness_studio": ["meditation_session", "yoga_session", "sound_healing_circle", "nutrition_workshop", "dance_class"],
    "mindfulness_center": ["meditation_session", "sound_healing_circle", "yoga_session"],
    "holistic_health_hub": ["nutrition_workshop", "meditation_session", "cooking_class"],
    "maker_space": ["hackathon", "ai_workshop", "science_night", "craft_fair"],
    "ai_research_lab": ["ai_workshop", "hackathon", "science_night"],
    "astronomy_club": ["science_night", "city_trip", "photo_walk"],
    "music_production_studio": ["dj_workshop", "open_mic_live", "singer_songwriter_night"],
    "jazz_lounge": ["jazz_concert", "open_mic_live", "singer_songwriter_night"],
    "vinyl_collectors_club": ["jazz_concert", "singer_songwriter_night", "open_mic_live"],
    "debate_society": ["storytelling_night", "language_meetup", "book_discussion", "volunteering_day"],
    "improv_theater_group": ["storytelling_night", "comedy_night", "theater_show"],
    "sustainable_living_co": ["sustainability_workshop", "nutrition_workshop", "cooking_class", "volunteering_day"],
    "digital_nomad_hub": ["hackathon", "ai_workshop", "language_meetup", "startup_pitch_night"],
    "fashion_collective": ["thrift_fair", "photo_walk", "craft_fair"],
    "anime_comics_club": ["movie_screening", "board_game_night", "craft_fair"],
    "photo_cinema_society": ["photo_walk", "movie_screening", "art_workshop"],
    "theater_company": ["theater_show", "open_mic_live", "storytelling_night"],
    "ceramics_pottery_studio": ["ceramics_workshop", "art_workshop", "craft_fair"],
    "creative_writing_circle": ["writing_workshop", "book_discussion", "language_meetup"],
    "team_sports_club": ["team_sports_match", "running_session", "city_trip"],
    "racket_sports_club": ["team_sports_match", "running_session", "yoga_session"],
    "combat_sports_gym": ["martial_arts_class", "running_session", "yoga_session"],
    "cosplay_gaming_crew": ["esports_tournament", "board_game_night", "movie_screening"],
    "film_dj_collective": ["dj_workshop", "open_mic_live", "movie_screening"],
    "craft_atelier": ["craft_fair", "life_drawing_session", "ceramics_workshop"],
    "woodwork_print_studio": ["craft_fair", "ceramics_workshop", "life_drawing_session"],
}

PERSONA_SPACE_PREFS: dict[str, list[str]] = {
    "outdoor_adventurer":   ["mountain_hikers", "climbing_crew", "travel_backpackers", "urban_runners", "yoga_wellness", "pet_lovers_club"],
    "visual_culture_fan":   ["anime_comics_club", "photo_cinema_society", "cinephile_collective", "street_photo_crew", "modern_art_collective", "live_music_tribe"],
    "performing_arts_fan":  ["theater_company", "live_music_tribe", "cinephile_collective", "jazz_lounge", "improv_theater_group"],
    "literary_craft_fan":   ["ceramics_pottery_studio", "creative_writing_circle", "book_cafe_club", "modern_art_collective", "craft_atelier"],
    "foodie":               ["foodie_circle", "wine_tasting_society", "coffee_explorers", "sustainable_living_co", "language_exchange_lounge"],
    "team_sports_fan":      ["team_sports_club", "racket_sports_club", "urban_runners", "travel_backpackers", "language_exchange_lounge"],
    "fitness_enthusiast":   ["combat_sports_gym", "urban_runners", "yoga_wellness", "mountain_hikers", "wellness_studio"],
    "digital_creative":     ["cosplay_gaming_crew", "film_dj_collective", "nerd_hub", "indie_gaming_club", "music_production_studio", "live_music_tribe"],
    "craft_creative":       ["craft_atelier", "woodwork_print_studio", "modern_art_collective", "ceramics_pottery_studio", "fashion_collective"],
    "wellness_seeker":      ["wellness_studio", "mindfulness_center", "holistic_health_hub", "yoga_wellness", "mountain_hikers", "pet_lovers_club"],
    "tech_geek":            ["maker_space", "ai_research_lab", "astronomy_club", "nerd_hub", "tech_founders_circle", "ai_builders_lab"],
    "music_lover":          ["music_production_studio", "jazz_lounge", "vinyl_collectors_club", "live_music_tribe", "film_dj_collective"],
    "social_butterfly":     ["language_exchange_lounge", "pet_lovers_club", "travel_backpackers", "board_games_society", "foodie_circle", "debate_society"],
    "lifestyle_explorer":   ["sustainable_living_co", "digital_nomad_hub", "fashion_collective", "travel_backpackers", "debate_society"],
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
    if status == "published":
        slot = random.choices(["morning", "afternoon", "evening"], weights=[0.18, 0.32, 0.50], k=1)[0]
    else:
        slot = random.choices(["morning", "afternoon", "evening"], weights=[0.22, 0.34, 0.44], k=1)[0]
    if slot == "morning":
        hour = random.randint(7, 11)
    elif slot == "afternoon":
        hour = random.randint(12, 17)
    else:
        hour = random.randint(18, 22)
    return hour, 0


def skewed_age(lo: int, hi: int) -> int:
    raw = random.betavariate(2.0, 3.5)
    return lo + round(raw * (hi - lo))


def power_law_popularity(n: int, exponent: float = 1.8) -> list[float]:
    ranks  = list(range(1, n + 1))
    random.shuffle(ranks)
    scores = [1.0 / (r ** exponent) for r in ranks]
    max_s  = max(scores)
    return [s / max_s for s in scores]


# ─── Category sampling ─────────────────────────────────────────────────────────

def sample_user_category_weights(persona: dict) -> dict[str, float]:
    """
    Assign per-category interest weights for a user.
    Primary categories get high weight (0.70–1.0), pool extras get mid weight,
    and a random sprinkle of others gets low weight for exploration.
    """
    primary = persona["primary_categories"]
    pool    = persona["category_pool"]
    extra   = [c for c in ALL_CATEGORY_IDS if c not in pool]

    weights: dict[str, float] = {}
    for cat in primary:
        weights[cat] = round(random.uniform(0.70, 1.00), 2)
    for cat in pool:
        if cat not in weights:
            weights[cat] = round(random.uniform(0.28, 0.65), 2)
    # Sprinkle 2–4 random categories with low weight
    n_extra = random.randint(2, 4)
    for cat in random.sample(extra, min(n_extra, len(extra))):
        weights[cat] = round(random.uniform(0.03, 0.18), 2)
    return weights


def sample_entity_categories(pool: list[str], min_n: int = 1, max_n: int = 4) -> list[str]:
    """Pick 1–4 categories from the given pool."""
    weights_dist = [0.15, 0.35, 0.35, 0.15][:max_n - min_n + 1]
    n = random.choices(range(min_n, max_n + 1), weights=weights_dist, k=1)[0]
    chosen = random.sample(pool, min(n, len(pool)))
    # Ensure we always have at least one
    if not chosen and pool:
        chosen = [random.choice(pool)]
    return chosen


def bump_category_weights(
    user: dict,
    item_categories: list[str],
    strength: float,
) -> None:
    """Nudge user category weights upward after an interaction."""
    cw = user["category_weights"]
    for cat in item_categories:
        current = float(cw.get(cat, 0.0))
        inc = strength * (1.0 - min(1.0, current))
        cw[cat] = round(min(1.25, current + inc), 4)


def _category_preference_factor(user_category_weights: dict[str, float], item_categories: list[str]) -> float:
    """
    Returns a [0.55, 1.05] factor based on overlap between user interests and item categories.
    """
    if not item_categories:
        return 0.75
    scores = []
    for cat in item_categories:
        w = user_category_weights.get(cat)
        if w is None:
            scores.append(-1.0)
        elif w <= 0.10:
            scores.append(-0.35)
        elif w <= 0.40:
            scores.append(0.15)
        else:
            scores.append(1.0)
    mean_score = sum(scores) / len(scores)
    norm = (mean_score + 1.0) / 2.0
    return 0.55 + (norm ** 0.85) * 0.50


def _item_cluster_affinity(item_categories: list[str]) -> list[float]:
    """
    Returns a vector of length N_CLUSTERS with the fraction of item categories
    belonging to each cluster. Used for persona-level sorting.
    """
    counts = [0.0] * N_CLUSTERS
    for cat in item_categories:
        for ci in CAT_TO_CLUSTERS.get(cat, []):
            counts[ci] += 1.0
    total = sum(counts) or 1.0
    return [c / total for c in counts]


# ─── Entity generators ─────────────────────────────────────────────────────────

def gen_user(persona_idx: int) -> dict:
    p     = PERSONAS[persona_idx]
    lo, hi = p["age_range"]
    today  = date.today()
    age    = skewed_age(lo, hi)
    bd     = today - timedelta(days=round(age * 365.25))

    n_ri  = random.randint(1, 3)
    ri    = random.choices(REL_INTENTS, weights=p["rel_intent_w"], k=n_ri)
    ri    = list(dict.fromkeys(ri))

    cw = sample_user_category_weights(p)
    return {
        "id":               uid(),
        "persona":          p["name"],       # internal only
        "persona_idx":      persona_idx,     # internal only
        "birthdate":        bd.isoformat(),
        "gender":           wc(GENDERS, p["gender_w"]),
        "relationshipIntent": ri,
        "smoking":          wc(SMOKING,   p["smoking_w"])              if random.random() > 0.10 else None,
        "drinking":         wc(DRINKING,  p["drinking_w"])             if random.random() > 0.10 else None,
        "activityLevel":    wc(p["activity_choices"], p["activity_w"]) if random.random() > 0.10 else None,
        "category_weights": cw,              # internal only — used for interaction simulation
    }


def gen_event(space: dict, status: str, popularity: float) -> dict:
    today = date.today()
    preferred_cluster = space["preferred_cluster"]

    if status == "completed":
        starts  = rand_date(today - timedelta(days=365), today - timedelta(days=1))
        a_count = max(2, round(random.gauss(10 + popularity * 70, 5 + popularity * 15)))
        avg_age = round(random.gauss(28 + preferred_cluster * 2, 5), 1)
    else:
        starts  = rand_date(today + timedelta(days=1), today + timedelta(days=180))
        a_count = max(0, round(random.gauss(popularity * 25, 5)))
        avg_age = round(random.gauss(28 + preferred_cluster * 2, 5), 1) if a_count > 0 else None

    # Pick an archetype compatible with the space, or fall back to cluster match
    compatible = SPACE_EVENT_COMPATIBILITY.get(space["archetype"], [])
    cluster_archetypes = [a for a in EVENT_ARCHETYPES if a["cluster"] == preferred_cluster]

    if compatible:
        arch_names = set(compatible)
        candidates = [a for a in EVENT_ARCHETYPES if a["name"] in arch_names]
        archetype  = random.choice(candidates) if candidates else random.choice(cluster_archetypes or EVENT_ARCHETYPES)
    elif cluster_archetypes:
        archetype = random.choice(cluster_archetypes)
    else:
        archetype = random.choice(EVENT_ARCHETYPES)

    categories = sample_entity_categories(
        list(set(space["categories"] + archetype["category_pool"])),
        min_n=1, max_n=3,
    )

    max_att    = random.choice([None, None, 20, 30, 50, 100, 200])
    price      = random.choice(archetype["price_choices"])
    start_hour, start_minute = sample_time_slot(status)

    return {
        "id":              uid(),
        "spaceId":         space["id"],
        "categories":      categories,
        "startsAt":        f"{starts} {start_hour:02d}:{start_minute:02d}:00",
        "maxAttendees":    max_att,
        "isPaid":          price > 0,
        "priceCents":      price,
        "attendeeCount":   a_count,
        "avgAttendeeAge":  avg_age,
        "preferred_cluster": preferred_cluster,  # internal only
        "popularity":       round(popularity, 3),  # internal only
        "_affinity":        None,                  # filled later
    }


def gen_space(archetype: dict, popularity: float) -> dict:
    cluster    = archetype["cluster"]
    categories = sample_entity_categories(archetype["category_pool"], min_n=1, max_n=4)
    return {
        "id":              uid(),
        "categories":      categories,
        "memberCount":     0,
        "avgMemberAge":    None,
        "eventCount":      0,
        "preferred_cluster": cluster,      # internal only
        "archetype":       archetype["name"],  # internal only
        "popularity":      round(popularity, 3),  # internal only
        "_affinity":       None,               # filled later
    }


# ─── Interaction generation ────────────────────────────────────────────────────

def _per_user_counts(
    n_users: int,
    target_total: int,
    event_ratio: float = 0.70,
    min_per_user: int = 5,
) -> list[tuple[int, int]]:
    raw = [
        max(min_per_user, round(random.lognormvariate(math.log(12), 0.85)))
        for _ in range(n_users)
    ]
    scale  = target_total / max(1, sum(raw))
    counts = [max(min_per_user, round(c * scale)) for c in raw]
    result = []
    for n in counts:
        n_ev = max(1, round(n * event_ratio))
        n_sp = max(1, n - n_ev)
        result.append((n_ev, n_sp))
    return result


def assign_interactions(
    users:          list[dict],
    events:         list[dict],
    spaces:         list[dict],
    n_interactions: int,
) -> list[dict]:
    """
    Generates ~n_interactions positive pairs.

    1. PERSONA COHERENCE: users sample primarily from persona-aligned items.
    2. POPULARITY BOOST: high-popularity items have higher base probability.
    3. SERENDIPITY: 5% of interactions are cross-persona exploration.
    4. POWER-LAW engagement: per-user counts follow lognormal distribution.
    """
    n_users     = len(users)
    user_counts = _per_user_counts(n_users, n_interactions)

    # Pre-compute affinity vectors for sorting
    for e in events:
        e["_affinity"] = _item_cluster_affinity(e["categories"])
    for s in spaces:
        s["_affinity"] = _item_cluster_affinity(s["categories"])

    n_personas = len(PERSONAS)
    persona_event_pools: list[list[str]] = []
    persona_space_pools: list[list[str]] = []

    for p_idx in range(n_personas):
        ci   = PERSONAS[p_idx]["cluster"]
        name = PERSONAS[p_idx]["name"]
        ev_sorted = sorted(
            events,
            key=lambda e, ci=ci: (e["_affinity"][ci] * 0.7 + e["popularity"] * 0.3),
            reverse=True,
        )
        persona_event_pools.append([e["id"] for e in ev_sorted])

        sp_sorted = sorted(
            spaces,
            key=lambda s, ci=ci, name=name: (
                s["_affinity"][ci] * 0.6
                + s["popularity"] * 0.25
                + (0.15 if s.get("archetype") in PERSONA_SPACE_PREFS.get(name, []) else 0.0)
            ),
            reverse=True,
        )
        persona_space_pools.append([s["id"] for s in sp_sorted])

    all_event_ids = [e["id"] for e in events]

    interactions: list[dict]   = []
    positive_pairs: set[tuple] = set()
    space_member_data: dict[str, list[dict]] = defaultdict(list)
    event_by_id = {e["id"]: e for e in events}
    space_by_id = {s["id"]: s for s in spaces}
    user_by_id  = {u["id"]: u for u in users}
    today = date.today()

    def _recency(d: date) -> float:
        return math.exp(-(today - d).days / 180.0)

    def _add(user_id: str, iid: str, itype: str) -> bool:
        if (user_id, iid) in positive_pairs:
            return False
        positive_pairs.add((user_id, iid))
        days_back  = round(random.betavariate(1.5, 3.0) * 364) + 1
        created_at = today - timedelta(days=days_back)
        user  = user_by_id[user_id]
        p_idx = user["persona_idx"]
        ci    = PERSONAS[p_idx]["cluster"]

        if itype == "event":
            item      = event_by_id[iid]
            try:
                event_day = date.fromisoformat(str(item["startsAt"]).split(" ")[0])
            except (TypeError, ValueError):
                event_day = today
            type_w       = 1.0 if event_day < today else 0.7
            pref_score   = float(item["_affinity"][ci])
            persona_factor = 0.35 + 0.65 * pref_score
            cat_factor   = _category_preference_factor(user["category_weights"], item["categories"])
            pref_factor  = persona_factor * cat_factor
            bump_category_weights(user, item["categories"], strength=0.02 + 0.08 * pref_factor)
        else:
            item         = space_by_id[iid]
            type_w       = 0.9
            pref_score   = float(item["_affinity"][ci])
            persona_factor = 0.35 + 0.65 * pref_score
            cat_factor   = _category_preference_factor(user["category_weights"], item["categories"])
            pref_factor  = persona_factor * cat_factor
            bump_category_weights(user, item["categories"], strength=0.015 + 0.06 * pref_factor)

        interactions.append({
            "userId":     user_id,
            "itemId":     iid,
            "itemType":   itype,
            "weight":     round(type_w * pref_factor * _recency(created_at), 4),
            "created_at": created_at.isoformat(),
        })
        return True

    for user, (n_ev, n_sp) in zip(users, user_counts):
        user_id = user["id"]
        p_idx   = user["persona_idx"]

        # Events: 95% persona-aligned top-quarter, 5% serendipity
        pool   = persona_event_pools[p_idx]
        top    = pool[:max(1, len(pool) // 4)]
        rest   = pool[len(pool) // 4:]
        n_top  = max(0, round(n_ev * 0.95))
        n_rest = n_ev - n_top
        sampled_ev  = random.sample(top,  min(n_top,  len(top)))
        sampled_ev += random.sample(rest, min(n_rest, len(rest)))
        if len(sampled_ev) < n_ev:
            extra = [e for e in all_event_ids if e not in sampled_ev]
            sampled_ev += random.sample(extra, min(n_ev - len(sampled_ev), len(extra)))

        for eid in sampled_ev:
            _add(user_id, eid, "event")

        # Spaces: 95% persona-aligned top-quarter, 5% serendipity
        pool   = persona_space_pools[p_idx]
        top    = pool[:max(1, len(pool) // 4)]
        rest   = pool[len(pool) // 4:]
        n_top  = max(0, round(n_sp * 0.95))
        n_rest = n_sp - n_top
        sampled_sp  = random.sample(top,  min(n_top,  len(top)))
        sampled_sp += random.sample(rest, min(n_rest, len(rest)))

        for sid in sampled_sp:
            if _add(user_id, sid, "space"):
                space_member_data[sid].append(user)

    for space in spaces:
        members = space_member_data[space["id"]]
        space["memberCount"] = len(members)
        ages = []
        for m in members:
            if m.get("birthdate"):
                bd = date.fromisoformat(m["birthdate"])
                ages.append((date.today() - bd).days / 365.25)
        space["avgMemberAge"] = round(sum(ages) / len(ages), 1) if ages else None

    return interactions


# ─── Strip internal fields before writing ──────────────────────────────────────

def _clean_user(u: dict) -> dict:
    return {k: v for k, v in u.items() if k not in ("persona", "persona_idx", "category_weights")}

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
        cluster = i % N_CLUSTERS
        cluster_candidates = [a for a in SPACE_ARCHETYPES if a["cluster"] == cluster]
        archetype = random.choice(cluster_candidates) if cluster_candidates and random.random() < 0.75 else random.choice(SPACE_ARCHETYPES)
        spaces.append(gen_space(archetype, pop))

    event_popularities = power_law_popularity(n_events)
    events = []
    for i, pop in enumerate(event_popularities):
        status = "completed" if i < round(n_events * 0.6) else "published"
        space  = random.choice(spaces)
        events.append(gen_event(space, status, pop))

    event_counts: dict[str, int] = defaultdict(int)
    for e in events:
        event_counts[e["spaceId"]] += 1
    for space in spaces:
        space["eventCount"] = event_counts[space["id"]]

    print("  Generating users...")
    users = []
    for i in range(n_users):
        users.append(gen_user(i % len(PERSONAS)))
    random.shuffle(users)

    print("  Generating interactions...")
    interactions = assign_interactions(users, events, spaces, n_interactions)

    def _write(name: str, data: list) -> None:
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ {name:<25} {len(data):>8,} records")

    print()
    _write("users.json",         [_clean_user(u) for u in users])
    _write("events.json",        [_clean_event(e) for e in events])
    _write("spaces.json",        [_clean_space(s) for s in spaces])

    event_attendees = [
        {"userId": ix["userId"], "eventId": ix["itemId"], "weight": ix["weight"], "created_at": ix["created_at"]}
        for ix in interactions if ix["itemType"] == "event"
    ]
    members = [
        {"userId": ix["userId"], "spaceId": ix["itemId"], "weight": ix["weight"], "created_at": ix["created_at"]}
        for ix in interactions if ix["itemType"] == "space"
    ]
    _write("event_attendees.json", event_attendees)
    _write("members.json",         members)

    # Generate categories.json with real OpenAI embeddings
    print(f"\n  Fetching OpenAI embeddings for {len(ALL_CATEGORY_IDS)} categories...")
    oai = get_openai_client()
    categories_data: list[dict] = []
    all_embeddings: list = []

    try:
        batch_size = 100
        for i in range(0, len(ALL_CATEGORY_IDS), batch_size):
            batch = ALL_CATEGORY_IDS[i:i + batch_size]
            response = oai.embeddings.create(
                input=[c.replace("_", " ") for c in batch],
                model="text-embedding-3-small",
                dimensions=64,
            )
            all_embeddings.extend([d.embedding for d in response.data])
        for idx, cat_id in enumerate(ALL_CATEGORY_IDS):
            categories_data.append({
                "id":        cat_id,
                "name":      cat_id.replace("_", " ").title(),
                "embedding": all_embeddings[idx],
            })
    except Exception as e:
        print(f"  [!] Failed to fetch OpenAI embeddings: {e}")
        print("  [!] Falling back to zero-embeddings for categories...")
        for cat_id in ALL_CATEGORY_IDS:
            categories_data.append({
                "id":        cat_id,
                "name":      cat_id.replace("_", " ").title(),
                "embedding": [0.0] * 64,
            })

    _write("categories.json", categories_data)

    # Generate category_impressions.json from post-simulation category_weights
    cat_impressions: list[dict] = []
    for u in users:
        for cat_id, weight in u.get("category_weights", {}).items():
            if weight > 0.05:
                cat_impressions.append({
                    "userId":   u["id"],
                    "itemId":   cat_id,
                    "itemType": "category",
                    "action":   "liked",
                    "weight":   round(weight, 4),
                })
    _write("category_impressions.json", cat_impressions)

    n_pos = len(interactions)
    print(f"\n  positive pairs : {n_pos:,}  (target {n_interactions:,})")
    print(f"  avg per user   : {n_pos / n_users:.1f}")

    by_type = defaultdict(int)
    for ix in interactions:
        by_type[ix["itemType"]] += 1
    print(f"\n  interaction breakdown:")
    for t, c in sorted(by_type.items()):
        print(f"    {t:<8}  {c:>8,}  ({100*c/n_pos:.1f}%)")

    avg_w = sum(ix["weight"] for ix in interactions) / max(1, n_pos)
    print(f"\n  avg weight     : {avg_w:.3f}")


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
