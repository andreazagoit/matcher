"""
Synthetic training data generator — realistic edition.

Produces JSON files in training-data/ that mirror 'npm run ml:export'.

─── Default volumes ──────────────────────────────────────────────────────────
Change the four constants below to adjust dataset size.
"""

from __future__ import annotations

# ┌─────────────────────────────────────────────────────────────────────────────
# │ CONFIGURE HERE
N_USERS        = 10_000
N_EVENTS       = 1_400   # scaled +40% for 14 personas (was 1000 for 10 personas)
N_SPACES       = 700     # scaled +40%
N_INTERACTIONS = 210_000  # scaled +40%
# └─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import math
import os
import random
import uuid
from collections import defaultdict
from datetime import date, timedelta
from typing import Optional

from hgt.config import TAG_VOCAB, TRAINING_DATA_DIR

# Clusters mirror the 10 TAG_CATEGORIES in config.py / data.ts.
# Every tag in TAG_VOCAB belongs to exactly one cluster (0–9).
TAG_CLUSTERS: list[list[str]] = [
    # 0 outdoor
    ["trekking", "camping", "climbing", "cycling", "beach", "mountains",
     "gardening", "surfing", "skiing", "kayaking", "fishing", "trail_running",
     "snowboarding", "skateboarding", "horse_riding", "sailing", "scuba_diving",
     "paragliding", "bouldering", "canyoning"],
    # 1 culture
    ["cinema", "theater", "live_music", "museums", "reading", "photography",
     "art", "opera", "ballet", "comedy_shows", "podcasts", "architecture",
     "vintage", "anime", "comics", "street_art", "literary_clubs", "calligraphy",
     "sculpture", "ceramics"],
    # 2 food_drink
    ["cooking", "restaurants", "wine", "craft_beer", "street_food", "coffee",
     "veganism", "sushi", "cocktails", "baking", "tea", "barbecue",
     "food_festivals", "cheese", "sake", "mixology", "foraging", "fermentation",
     "ramen", "pastry"],
    # 3 sports
    ["running", "gym", "yoga", "swimming", "football", "tennis", "padel",
     "basketball", "volleyball", "boxing", "martial_arts", "pilates", "crossfit",
     "golf", "rugby", "archery", "dance", "rowing", "hockey", "badminton"],
    # 4 creative
    ["music", "drawing", "writing", "diy", "gaming", "coding", "painting",
     "pottery", "knitting", "woodworking", "film_making", "singing", "djing",
     "digital_art", "cosplay", "jewelry_making", "embroidery", "leathercraft",
     "printmaking", "glass_blowing"],
    # 5 wellness
    ["meditation", "mindfulness", "journaling", "spa", "breathwork", "nutrition",
     "therapy", "cold_exposure", "sound_healing", "ayurveda", "reiki",
     "stretching", "sleep_hygiene", "herbal_medicine", "qi_gong", "forest_bathing",
     "intermittent_fasting", "positive_psychology", "aromatherapy", "somatic_practices"],
    # 6 tech_science
    ["ai", "blockchain", "cybersecurity", "robotics", "vr_ar", "open_source",
     "data_science", "smart_home", "esports", "astronomy", "chemistry", "biology",
     "engineering", "drones", "quantum_computing", "3d_printing",
     "space_exploration", "neuroscience", "environmental_science", "biohacking"],
    # 7 music_genres
    ["jazz", "classical_music", "hip_hop", "electronic_music", "rock",
     "indie_music", "reggae", "pop_music", "metal", "country_music", "blues",
     "r_and_b", "folk_music", "gospel", "bossa_nova", "afrobeat", "techno",
     "house_music", "punk", "soul"],
    # 8 social
    ["travel", "volunteering", "languages", "pets", "parties", "board_games",
     "networking", "karaoke", "escape_rooms", "trivia_nights", "activism",
     "mentoring", "astrology", "tarot", "stand_up", "improv", "storytelling",
     "cultural_exchange", "language_exchange", "community_events"],
    # 9 lifestyle
    ["fashion", "interior_design", "sustainability", "minimalism", "van_life",
     "urban_exploration", "thrifting", "luxury_lifestyle", "tattoos",
     "personal_growth", "entrepreneurship", "parenting", "spirituality",
     "digital_nomad", "homesteading", "zero_waste", "slow_living", "nightlife",
     "brunch_culture", "self_improvement"],
]

# Fast lookup: tag → cluster index
TAG_TO_CLUSTER: dict[str, int] = {
    tag: ci for ci, cluster in enumerate(TAG_CLUSTERS) for tag in cluster
}

# ─── Personas ─────────────────────────────────────────────────────────────────
#
# Each persona represents an archetypal user type.
# Rules:
#  - age_range:     realistic age bracket for this persona
#  - cluster:       primary tag cluster (60-70% of their tags come from here)
#  - activity/smoking/drinking: persona-specific lifestyle distributions
#  - rel_intent_w:  weights over [serious_relationship, casual_dating, friendship, chat]
#  - gender_w:      weights over [man, woman, non_binary]

PERSONAS = [
    # ── cluster 0: outdoor ───────────────────────────────────────────────────
    {
        "name":         "outdoor_adventurer",
        "cluster":      0,
        "tag_subpool":  ["trekking", "camping", "climbing", "cycling", "mountains",
                         "surfing", "skiing", "trail_running", "kayaking"],
        "age_range":    (22, 32),
        "gender_w":     [0.40, 0.50, 0.10],
        "rel_intent_w": [0.25, 0.40, 0.25, 0.10],
        "smoking_w":    [0.88, 0.10, 0.02],
        "drinking_w":   [0.25, 0.60, 0.15],
        "activity_choices": ["active", "very_active"],
        "activity_w":   [0.35, 0.65],
    },
    # ── cluster 1: culture — split into 3 sub-personas ───────────────────────
    {
        "name":         "visual_culture_fan",   # cinema, photo, anime, comics
        "cluster":      1,
        "tag_subpool":  ["cinema", "photography", "anime", "comics", "street_art",
                         "vintage", "architecture"],
        "age_range":    (20, 38),
        "gender_w":     [0.38, 0.48, 0.14],
        "rel_intent_w": [0.32, 0.30, 0.28, 0.10],
        "smoking_w":    [0.52, 0.34, 0.14],
        "drinking_w":   [0.18, 0.54, 0.28],
        "activity_choices": ["light", "moderate"],
        "activity_w":   [0.45, 0.55],
    },
    {
        "name":         "performing_arts_fan",  # theater, opera, ballet, live music
        "cluster":      1,
        "tag_subpool":  ["theater", "opera", "ballet", "live_music", "comedy_shows",
                         "museums", "architecture"],
        "age_range":    (28, 48),
        "gender_w":     [0.30, 0.60, 0.10],
        "rel_intent_w": [0.45, 0.20, 0.25, 0.10],
        "smoking_w":    [0.50, 0.36, 0.14],
        "drinking_w":   [0.12, 0.50, 0.38],
        "activity_choices": ["light", "moderate"],
        "activity_w":   [0.42, 0.58],
    },
    {
        "name":         "literary_craft_fan",   # reading, ceramics, sculpture
        "cluster":      1,
        "tag_subpool":  ["reading", "podcasts", "literary_clubs", "calligraphy",
                         "sculpture", "ceramics", "art", "museums"],
        "age_range":    (26, 45),
        "gender_w":     [0.32, 0.58, 0.10],
        "rel_intent_w": [0.42, 0.22, 0.28, 0.08],
        "smoking_w":    [0.58, 0.28, 0.14],
        "drinking_w":   [0.15, 0.55, 0.30],
        "activity_choices": ["sedentary", "light", "moderate"],
        "activity_w":   [0.22, 0.50, 0.28],
    },
    # ── cluster 2: food_drink ────────────────────────────────────────────────
    {
        "name":         "foodie",
        "cluster":      2,
        "tag_subpool":  ["cooking", "restaurants", "wine", "coffee", "street_food",
                         "baking", "cocktails", "craft_beer"],
        "age_range":    (24, 40),
        "gender_w":     [0.40, 0.50, 0.10],
        "rel_intent_w": [0.35, 0.30, 0.25, 0.10],
        "smoking_w":    [0.60, 0.28, 0.12],
        "drinking_w":   [0.08, 0.47, 0.45],
        "activity_choices": ["light", "moderate"],
        "activity_w":   [0.50, 0.50],
    },
    # ── cluster 3: sports — split into team vs fitness ───────────────────────
    {
        "name":         "team_sports_fan",
        "cluster":      3,
        "tag_subpool":  ["football", "basketball", "volleyball", "tennis", "padel",
                         "rugby", "hockey", "badminton"],
        "age_range":    (18, 30),
        "gender_w":     [0.68, 0.28, 0.04],
        "rel_intent_w": [0.18, 0.48, 0.24, 0.10],
        "smoking_w":    [0.90, 0.08, 0.02],
        "drinking_w":   [0.28, 0.58, 0.14],
        "activity_choices": ["active", "very_active"],
        "activity_w":   [0.30, 0.70],
    },
    {
        "name":         "fitness_enthusiast",
        "cluster":      3,
        "tag_subpool":  ["running", "gym", "yoga", "swimming", "pilates",
                         "crossfit", "boxing", "martial_arts"],
        "age_range":    (20, 35),
        "gender_w":     [0.50, 0.45, 0.05],
        "rel_intent_w": [0.25, 0.40, 0.25, 0.10],
        "smoking_w":    [0.95, 0.04, 0.01],
        "drinking_w":   [0.35, 0.52, 0.13],
        "activity_choices": ["active", "very_active"],
        "activity_w":   [0.28, 0.72],
    },
    # ── cluster 4: creative — split into digital vs craft ────────────────────
    {
        "name":         "digital_creative",
        "cluster":      4,
        "tag_subpool":  ["gaming", "coding", "digital_art", "film_making",
                         "djing", "music", "cosplay", "singing"],
        "age_range":    (18, 34),
        "gender_w":     [0.48, 0.38, 0.14],
        "rel_intent_w": [0.18, 0.38, 0.32, 0.12],
        "smoking_w":    [0.50, 0.34, 0.16],
        "drinking_w":   [0.18, 0.52, 0.30],
        "activity_choices": ["sedentary", "light", "moderate"],
        "activity_w":   [0.28, 0.48, 0.24],
    },
    {
        "name":         "craft_creative",
        "cluster":      4,
        "tag_subpool":  ["drawing", "painting", "writing", "pottery", "knitting",
                         "woodworking", "jewelry_making", "embroidery", "printmaking"],
        "age_range":    (22, 40),
        "gender_w":     [0.22, 0.60, 0.18],
        "rel_intent_w": [0.28, 0.30, 0.32, 0.10],
        "smoking_w":    [0.40, 0.40, 0.20],
        "drinking_w":   [0.15, 0.55, 0.30],
        "activity_choices": ["sedentary", "light", "moderate"],
        "activity_w":   [0.18, 0.52, 0.30],
    },
    # ── cluster 5: wellness ──────────────────────────────────────────────────
    {
        "name":         "wellness_seeker",
        "cluster":      5,
        "tag_subpool":  ["meditation", "mindfulness", "yoga", "breathwork",
                         "journaling", "nutrition", "spa", "sound_healing"],
        "age_range":    (26, 44),
        "gender_w":     [0.22, 0.65, 0.13],
        "rel_intent_w": [0.38, 0.25, 0.28, 0.09],
        "smoking_w":    [0.90, 0.08, 0.02],
        "drinking_w":   [0.35, 0.52, 0.13],
        "activity_choices": ["light", "moderate", "active"],
        "activity_w":   [0.25, 0.50, 0.25],
    },
    # ── cluster 6: tech_science ──────────────────────────────────────────────
    {
        "name":         "tech_geek",
        "cluster":      6,
        "tag_subpool":  ["ai", "coding", "data_science", "open_source",
                         "cybersecurity", "robotics", "vr_ar", "blockchain"],
        "age_range":    (20, 38),
        "gender_w":     [0.70, 0.22, 0.08],
        "rel_intent_w": [0.22, 0.38, 0.28, 0.12],
        "smoking_w":    [0.78, 0.17, 0.05],
        "drinking_w":   [0.28, 0.55, 0.17],
        "activity_choices": ["sedentary", "light", "moderate"],
        "activity_w":   [0.30, 0.45, 0.25],
    },
    # ── cluster 7: music_genres ──────────────────────────────────────────────
    {
        "name":         "music_lover",
        "cluster":      7,
        "tag_subpool":  ["jazz", "electronic_music", "hip_hop", "rock",
                         "indie_music", "live_music", "classical_music", "r_and_b"],
        "age_range":    (18, 36),
        "gender_w":     [0.45, 0.45, 0.10],
        "rel_intent_w": [0.22, 0.40, 0.28, 0.10],
        "smoking_w":    [0.45, 0.38, 0.17],
        "drinking_w":   [0.10, 0.50, 0.40],
        "activity_choices": ["light", "moderate", "active"],
        "activity_w":   [0.35, 0.45, 0.20],
    },
    # ── cluster 8: social ────────────────────────────────────────────────────
    {
        "name":         "social_butterfly",
        "cluster":      8,
        "tag_subpool":  ["travel", "parties", "board_games", "karaoke",
                         "languages", "trivia_nights", "networking", "escape_rooms"],
        "age_range":    (20, 34),
        "gender_w":     [0.38, 0.52, 0.10],
        "rel_intent_w": [0.18, 0.35, 0.32, 0.15],
        "smoking_w":    [0.48, 0.35, 0.17],
        "drinking_w":   [0.05, 0.43, 0.52],
        "activity_choices": ["moderate", "active"],
        "activity_w":   [0.55, 0.45],
    },
    # ── cluster 9: lifestyle ─────────────────────────────────────────────────
    {
        "name":         "lifestyle_explorer",
        "cluster":      9,
        "tag_subpool":  ["sustainability", "digital_nomad", "fashion",
                         "travel", "entrepreneurship", "interior_design",
                         "personal_growth", "minimalism"],
        "age_range":    (22, 40),
        "gender_w":     [0.30, 0.58, 0.12],
        "rel_intent_w": [0.28, 0.33, 0.28, 0.11],
        "smoking_w":    [0.55, 0.32, 0.13],
        "drinking_w":   [0.12, 0.50, 0.38],
        "activity_choices": ["light", "moderate", "active"],
        "activity_w":   [0.30, 0.50, 0.20],
    },
]

REL_INTENTS = ["serious_relationship", "casual_dating", "friendship", "chat"]
GENDERS     = ["man", "woman", "non_binary"]
SMOKING     = ["never", "sometimes", "regularly"]
DRINKING    = ["never", "sometimes", "regularly"]

SPACE_ARCHETYPES: list[dict] = [
    {"name": "nerd_hub", "cluster": 4, "tag_pool": ["gaming", "coding", "board_games", "reading", "coffee"]},
    {"name": "board_games_society", "cluster": 5, "tag_pool": ["board_games", "parties", "coffee", "reading", "gaming"]},
    {"name": "indie_gaming_club", "cluster": 4, "tag_pool": ["gaming", "coding", "music", "parties", "coffee"]},
    {"name": "tech_founders_circle", "cluster": 4, "tag_pool": ["coding", "languages", "coffee", "travel", "reading"]},
    {"name": "ai_builders_lab", "cluster": 4, "tag_pool": ["coding", "reading", "diy", "coffee", "gaming"]},
    {"name": "cinephile_collective", "cluster": 1, "tag_pool": ["cinema", "theater", "reading", "coffee", "museums"]},
    {"name": "book_cafe_club", "cluster": 1, "tag_pool": ["reading", "writing", "coffee", "theater", "museums"]},
    {"name": "street_photo_crew", "cluster": 1, "tag_pool": ["photography", "art", "travel", "coffee", "museums"]},
    {"name": "modern_art_collective", "cluster": 1, "tag_pool": ["art", "museums", "drawing", "photography", "theater"]},
    {"name": "live_music_tribe", "cluster": 4, "tag_pool": ["music", "live_music", "parties", "coffee", "art"]},
    {"name": "foodie_circle", "cluster": 2, "tag_pool": ["restaurants", "street_food", "cooking", "coffee", "travel"]},
    {"name": "wine_tasting_society", "cluster": 2, "tag_pool": ["wine", "restaurants", "craft_beer", "street_food", "coffee"]},
    {"name": "coffee_explorers", "cluster": 2, "tag_pool": ["coffee", "street_food", "restaurants", "reading", "travel"]},
    {"name": "urban_runners", "cluster": 3, "tag_pool": ["running", "gym", "cycling", "yoga", "swimming"]},
    {"name": "yoga_wellness", "cluster": 3, "tag_pool": ["yoga", "running", "swimming", "gym", "meditation"]},
    {"name": "mountain_hikers", "cluster": 0, "tag_pool": ["trekking", "mountains", "camping", "travel", "climbing"]},
    {"name": "climbing_crew", "cluster": 0, "tag_pool": ["climbing", "mountains", "trekking", "camping", "gym"]},
    {"name": "travel_backpackers", "cluster": 8, "tag_pool": ["travel", "languages", "beach", "mountains", "photography"]},
    {"name": "pet_lovers_club", "cluster": 8, "tag_pool": ["pets", "volunteering", "travel", "coffee", "parties"]},
    {"name": "language_exchange_lounge", "cluster": 8, "tag_pool": ["languages", "travel", "coffee", "parties", "reading"]},
    # wellness cluster (5)
    {"name": "wellness_studio", "cluster": 5, "tag_pool": ["meditation", "yoga", "breathwork", "mindfulness", "spa"]},
    {"name": "mindfulness_center", "cluster": 5, "tag_pool": ["meditation", "mindfulness", "journaling", "qi_gong", "sound_healing"]},
    {"name": "holistic_health_hub", "cluster": 5, "tag_pool": ["nutrition", "herbal_medicine", "ayurveda", "reiki", "meditation"]},
    # tech_science cluster (6)
    {"name": "maker_space", "cluster": 6, "tag_pool": ["coding", "3d_printing", "robotics", "drones", "diy"]},
    {"name": "ai_research_lab", "cluster": 6, "tag_pool": ["ai", "data_science", "coding", "open_source", "quantum_computing"]},
    {"name": "astronomy_club", "cluster": 6, "tag_pool": ["astronomy", "space_exploration", "biology", "environmental_science", "neuroscience"]},
    # music_genres cluster (7)
    {"name": "music_production_studio", "cluster": 7, "tag_pool": ["music", "djing", "electronic_music", "rock", "indie_music"]},
    {"name": "jazz_lounge", "cluster": 7, "tag_pool": ["jazz", "blues", "soul", "live_music", "bossa_nova"]},
    {"name": "vinyl_collectors_club", "cluster": 7, "tag_pool": ["music", "jazz", "rock", "indie_music", "folk_music"]},
    # social cluster (8)
    {"name": "debate_society", "cluster": 8, "tag_pool": ["activism", "languages", "storytelling", "networking", "cultural_exchange"]},
    {"name": "improv_theater_group", "cluster": 8, "tag_pool": ["improv", "stand_up", "storytelling", "theater", "comedy_shows"]},
    # lifestyle cluster (9)
    {"name": "sustainable_living_co", "cluster": 9, "tag_pool": ["sustainability", "zero_waste", "homesteading", "veganism", "foraging"]},
    {"name": "digital_nomad_hub", "cluster": 9, "tag_pool": ["digital_nomad", "travel", "networking", "coding", "entrepreneurship"]},
    {"name": "fashion_collective", "cluster": 9, "tag_pool": ["fashion", "thrifting", "photography", "vintage", "interior_design"]},
    # visual_culture_fan (cluster 1 sub)
    {"name": "anime_comics_club", "cluster": 1, "tag_pool": ["anime", "comics", "cosplay", "gaming", "street_art"]},
    {"name": "photo_cinema_society", "cluster": 1, "tag_pool": ["photography", "cinema", "vintage", "street_art", "architecture"]},
    # performing_arts_fan (cluster 1 sub)
    {"name": "theater_company", "cluster": 1, "tag_pool": ["theater", "opera", "ballet", "comedy_shows", "live_music"]},
    # literary_craft_fan (cluster 1 sub)
    {"name": "ceramics_pottery_studio", "cluster": 1, "tag_pool": ["ceramics", "sculpture", "pottery", "calligraphy", "art"]},
    {"name": "creative_writing_circle", "cluster": 1, "tag_pool": ["reading", "writing", "literary_clubs", "podcasts", "calligraphy"]},
    # team_sports_fan (cluster 3 sub)
    {"name": "team_sports_club", "cluster": 3, "tag_pool": ["football", "basketball", "volleyball", "padel", "tennis"]},
    {"name": "racket_sports_club", "cluster": 3, "tag_pool": ["tennis", "padel", "badminton", "volleyball", "running"]},
    # fitness_enthusiast (cluster 3 sub)
    {"name": "combat_sports_gym", "cluster": 3, "tag_pool": ["boxing", "martial_arts", "crossfit", "gym", "running"]},
    # digital_creative (cluster 4 sub)
    {"name": "cosplay_gaming_crew", "cluster": 4, "tag_pool": ["cosplay", "gaming", "anime", "digital_art", "comics"]},
    {"name": "film_dj_collective", "cluster": 4, "tag_pool": ["film_making", "djing", "music", "digital_art", "singing"]},
    # craft_creative (cluster 4 sub)
    {"name": "craft_atelier", "cluster": 4, "tag_pool": ["drawing", "painting", "knitting", "embroidery", "jewelry_making"]},
    {"name": "woodwork_print_studio", "cluster": 4, "tag_pool": ["woodworking", "printmaking", "pottery", "leathercraft", "glass_blowing"]},
]

EVENT_ARCHETYPES: list[dict] = [
    {"name": "board_game_night", "cluster": 5, "tag_pool": ["board_games", "parties", "coffee"], "price_choices": [0, 500, 1000], "hour_choices": [19, 20, 21], "max_choices": [12, 20, 30]},
    {"name": "esports_tournament", "cluster": 4, "tag_pool": ["gaming", "coding", "parties"], "price_choices": [0, 1000, 1500], "hour_choices": [18, 19, 20], "max_choices": [20, 40, 80]},
    {"name": "lan_party", "cluster": 4, "tag_pool": ["gaming", "coding", "music"], "price_choices": [0, 800, 1200], "hour_choices": [19, 20], "max_choices": [16, 24, 40]},
    {"name": "hackathon", "cluster": 4, "tag_pool": ["coding", "diy", "coffee"], "price_choices": [0, 2000, 3000], "hour_choices": [9, 10], "max_choices": [30, 60, 120]},
    {"name": "ai_workshop", "cluster": 4, "tag_pool": ["coding", "reading", "coffee"], "price_choices": [0, 1500, 2500], "hour_choices": [18, 19], "max_choices": [20, 40, 60]},
    {"name": "startup_pitch_night", "cluster": 4, "tag_pool": ["coding", "languages", "coffee"], "price_choices": [0, 1000, 2000], "hour_choices": [18, 19, 20], "max_choices": [20, 40, 80]},
    {"name": "movie_screening", "cluster": 1, "tag_pool": ["cinema", "theater", "coffee"], "price_choices": [0, 700, 1200], "hour_choices": [20, 21], "max_choices": [20, 40, 80]},
    {"name": "book_discussion", "cluster": 1, "tag_pool": ["reading", "writing", "coffee"], "price_choices": [0, 500, 1000], "hour_choices": [18, 19, 20], "max_choices": [10, 16, 24]},
    {"name": "photo_walk", "cluster": 1, "tag_pool": ["photography", "art", "travel"], "price_choices": [0, 1000], "hour_choices": [9, 10, 16], "max_choices": [12, 20, 30]},
    {"name": "art_workshop", "cluster": 1, "tag_pool": ["art", "drawing", "museums"], "price_choices": [1000, 2000, 3000], "hour_choices": [17, 18, 19], "max_choices": [10, 16, 24]},
    {"name": "open_mic_live", "cluster": 4, "tag_pool": ["music", "live_music", "parties"], "price_choices": [0, 1000, 1500], "hour_choices": [20, 21], "max_choices": [20, 40, 70]},
    {"name": "cooking_class", "cluster": 2, "tag_pool": ["cooking", "restaurants", "street_food"], "price_choices": [1500, 2500, 3500], "hour_choices": [11, 18, 19], "max_choices": [8, 12, 20]},
    {"name": "wine_tasting", "cluster": 2, "tag_pool": ["wine", "restaurants", "craft_beer"], "price_choices": [2000, 3000, 5000], "hour_choices": [19, 20], "max_choices": [12, 20, 30]},
    {"name": "coffee_cupping", "cluster": 2, "tag_pool": ["coffee", "street_food", "reading"], "price_choices": [0, 1000, 1500], "hour_choices": [10, 11, 16], "max_choices": [10, 16, 24]},
    {"name": "running_session", "cluster": 3, "tag_pool": ["running", "cycling", "gym"], "price_choices": [0, 500, 1000], "hour_choices": [7, 8, 18], "max_choices": [12, 25, 40]},
    {"name": "yoga_session", "cluster": 3, "tag_pool": ["yoga", "swimming", "running"], "price_choices": [0, 1000, 1500], "hour_choices": [7, 8, 19], "max_choices": [10, 20, 30]},
    {"name": "mountain_trek", "cluster": 0, "tag_pool": ["trekking", "mountains", "camping"], "price_choices": [0, 1500, 2500], "hour_choices": [7, 8, 9], "max_choices": [10, 20, 30]},
    {"name": "climbing_session", "cluster": 0, "tag_pool": ["climbing", "mountains", "gym"], "price_choices": [1000, 2000, 3000], "hour_choices": [17, 18, 19], "max_choices": [8, 16, 24]},
    {"name": "city_trip", "cluster": 8, "tag_pool": ["travel", "languages", "photography"], "price_choices": [0, 2000, 4000], "hour_choices": [8, 9, 10], "max_choices": [12, 20, 35]},
    {"name": "language_meetup", "cluster": 8, "tag_pool": ["languages", "travel", "coffee"], "price_choices": [0, 500, 1000], "hour_choices": [18, 19, 20], "max_choices": [12, 24, 40]},
    # wellness (5)
    {"name": "meditation_session", "cluster": 5, "tag_pool": ["meditation", "mindfulness", "breathwork"], "price_choices": [0, 1000, 1500], "hour_choices": [7, 8, 19], "max_choices": [10, 20, 30]},
    {"name": "nutrition_workshop", "cluster": 5, "tag_pool": ["nutrition", "cooking", "veganism"], "price_choices": [1000, 2000, 3000], "hour_choices": [18, 19], "max_choices": [12, 20, 30]},
    {"name": "sound_healing_circle", "cluster": 5, "tag_pool": ["sound_healing", "meditation", "mindfulness"], "price_choices": [0, 1500, 2000], "hour_choices": [19, 20], "max_choices": [10, 16, 24]},
    # tech_science (6)
    {"name": "drone_racing", "cluster": 6, "tag_pool": ["drones", "engineering", "gaming"], "price_choices": [0, 1000, 2000], "hour_choices": [10, 14, 16], "max_choices": [12, 20, 40]},
    {"name": "science_night", "cluster": 6, "tag_pool": ["astronomy", "biology", "chemistry"], "price_choices": [0, 1000, 1500], "hour_choices": [19, 20], "max_choices": [20, 40, 60]},
    # music_genres (7)
    {"name": "jazz_concert", "cluster": 7, "tag_pool": ["jazz", "live_music", "blues"], "price_choices": [500, 1500, 3000], "hour_choices": [20, 21], "max_choices": [30, 60, 120]},
    {"name": "dj_workshop", "cluster": 7, "tag_pool": ["djing", "electronic_music", "house_music"], "price_choices": [1000, 2000, 3000], "hour_choices": [18, 19], "max_choices": [10, 16, 24]},
    {"name": "singer_songwriter_night", "cluster": 7, "tag_pool": ["singing", "folk_music", "indie_music"], "price_choices": [0, 500, 1000], "hour_choices": [20, 21], "max_choices": [20, 40, 60]},
    # social (8)
    {"name": "storytelling_night", "cluster": 8, "tag_pool": ["storytelling", "improv", "stand_up"], "price_choices": [0, 500, 1000], "hour_choices": [19, 20, 21], "max_choices": [20, 40, 60]},
    {"name": "cultural_exchange_dinner", "cluster": 8, "tag_pool": ["cultural_exchange", "languages", "cooking"], "price_choices": [0, 1500, 2500], "hour_choices": [19, 20], "max_choices": [12, 20, 30]},
    # lifestyle (9)
    {"name": "thrift_fair", "cluster": 9, "tag_pool": ["thrifting", "fashion", "vintage"], "price_choices": [0, 500, 1000], "hour_choices": [10, 11, 12], "max_choices": [50, 100, 200]},
    {"name": "sustainability_workshop", "cluster": 9, "tag_pool": ["sustainability", "zero_waste", "homesteading"], "price_choices": [0, 1000, 1500], "hour_choices": [10, 18, 19], "max_choices": [20, 40, 60]},
    # visual_culture_fan
    {"name": "anime_screening", "cluster": 1, "tag_pool": ["anime", "comics", "cinema", "street_art"], "price_choices": [0, 500, 1000], "hour_choices": [18, 19, 20], "max_choices": [20, 40, 80]},
    {"name": "street_photography_walk", "cluster": 1, "tag_pool": ["photography", "street_art", "architecture", "vintage"], "price_choices": [0, 800], "hour_choices": [9, 10, 16], "max_choices": [10, 20, 30]},
    # performing_arts_fan
    {"name": "theater_show", "cluster": 1, "tag_pool": ["theater", "opera", "ballet", "comedy_shows"], "price_choices": [500, 1500, 3000], "hour_choices": [20, 21], "max_choices": [30, 80, 200]},
    # literary_craft_fan
    {"name": "ceramics_workshop", "cluster": 1, "tag_pool": ["ceramics", "sculpture", "pottery", "calligraphy"], "price_choices": [1500, 2500, 4000], "hour_choices": [10, 17, 18], "max_choices": [6, 10, 16]},
    {"name": "writing_workshop", "cluster": 1, "tag_pool": ["writing", "reading", "literary_clubs", "podcasts"], "price_choices": [0, 1000, 2000], "hour_choices": [18, 19], "max_choices": [10, 20, 30]},
    # team_sports_fan
    {"name": "team_sports_match", "cluster": 3, "tag_pool": ["football", "basketball", "volleyball", "padel"], "price_choices": [0, 500, 1000], "hour_choices": [9, 15, 18], "max_choices": [10, 20, 40]},
    # fitness_enthusiast
    {"name": "martial_arts_class", "cluster": 3, "tag_pool": ["boxing", "martial_arts", "crossfit", "gym"], "price_choices": [1000, 1500, 2500], "hour_choices": [7, 18, 19], "max_choices": [8, 16, 24]},
    # digital_creative
    {"name": "cosplay_contest", "cluster": 4, "tag_pool": ["cosplay", "gaming", "anime", "digital_art"], "price_choices": [0, 1000, 2000], "hour_choices": [14, 15, 16], "max_choices": [30, 80, 200]},
    {"name": "film_screening_indie", "cluster": 4, "tag_pool": ["film_making", "cinema", "music", "digital_art"], "price_choices": [0, 500, 1000], "hour_choices": [19, 20, 21], "max_choices": [20, 40, 80]},
    # craft_creative
    {"name": "craft_fair", "cluster": 4, "tag_pool": ["knitting", "embroidery", "jewelry_making", "drawing", "painting"], "price_choices": [0, 500, 1000], "hour_choices": [10, 11], "max_choices": [30, 60, 120]},
    {"name": "life_drawing_session", "cluster": 4, "tag_pool": ["drawing", "painting", "art", "museums"], "price_choices": [1000, 1500, 2500], "hour_choices": [18, 19], "max_choices": [8, 12, 20]},
]

SPACE_EVENT_COMPATIBILITY: dict[str, list[str]] = {
    "nerd_hub": ["board_game_night", "esports_tournament", "lan_party", "hackathon", "ai_workshop"],
    "board_games_society": ["board_game_night", "language_meetup", "coffee_cupping"],
    "indie_gaming_club": ["esports_tournament", "lan_party", "open_mic_live"],
    "tech_founders_circle": ["hackathon", "ai_workshop", "startup_pitch_night", "language_meetup"],
    "ai_builders_lab": ["hackathon", "ai_workshop", "startup_pitch_night"],
    "cinephile_collective": ["movie_screening", "book_discussion", "photo_walk"],
    "book_cafe_club": ["book_discussion", "coffee_cupping", "language_meetup"],
    "street_photo_crew": ["photo_walk", "city_trip", "art_workshop"],
    "modern_art_collective": ["art_workshop", "photo_walk", "open_mic_live"],
    "live_music_tribe": ["open_mic_live", "movie_screening", "board_game_night"],
    "foodie_circle": ["cooking_class", "wine_tasting", "coffee_cupping"],
    "wine_tasting_society": ["wine_tasting", "cooking_class", "open_mic_live"],
    "coffee_explorers": ["coffee_cupping", "book_discussion", "language_meetup"],
    "urban_runners": ["running_session", "city_trip", "yoga_session"],
    "yoga_wellness": ["yoga_session", "running_session", "coffee_cupping"],
    "mountain_hikers": ["mountain_trek", "city_trip", "climbing_session"],
    "climbing_crew": ["climbing_session", "mountain_trek", "running_session"],
    "travel_backpackers": ["city_trip", "language_meetup", "photo_walk"],
    "pet_lovers_club": ["city_trip", "coffee_cupping", "board_game_night"],
    "language_exchange_lounge": ["language_meetup", "city_trip", "book_discussion"],
    # wellness
    "wellness_studio": ["meditation_session", "yoga_session", "sound_healing_circle", "nutrition_workshop"],
    "mindfulness_center": ["meditation_session", "sound_healing_circle", "yoga_session"],
    "holistic_health_hub": ["nutrition_workshop", "meditation_session", "cooking_class"],
    # tech_science
    "maker_space": ["drone_racing", "hackathon", "ai_workshop"],
    "ai_research_lab": ["ai_workshop", "hackathon", "science_night"],
    "astronomy_club": ["science_night", "city_trip", "photo_walk"],
    # music_genres
    "music_production_studio": ["dj_workshop", "open_mic_live", "singer_songwriter_night"],
    "jazz_lounge": ["jazz_concert", "open_mic_live", "singer_songwriter_night"],
    "vinyl_collectors_club": ["jazz_concert", "singer_songwriter_night", "open_mic_live"],
    # social
    "debate_society": ["storytelling_night", "language_meetup", "book_discussion"],
    "improv_theater_group": ["storytelling_night", "open_mic_live", "movie_screening"],
    # lifestyle
    "sustainable_living_co": ["sustainability_workshop", "nutrition_workshop", "cooking_class"],
    "digital_nomad_hub": ["hackathon", "ai_workshop", "language_meetup"],
    "fashion_collective": ["thrift_fair", "photo_walk", "art_workshop"],
    # visual_culture sub-spaces
    "anime_comics_club": ["anime_screening", "cosplay_contest", "movie_screening"],
    "photo_cinema_society": ["street_photography_walk", "photo_walk", "movie_screening"],
    # performing_arts sub-spaces
    "theater_company": ["theater_show", "open_mic_live", "movie_screening"],
    # literary_craft sub-spaces
    "ceramics_pottery_studio": ["ceramics_workshop", "art_workshop", "photo_walk"],
    "creative_writing_circle": ["writing_workshop", "book_discussion", "language_meetup"],
    # sports sub-spaces
    "team_sports_club": ["team_sports_match", "running_session", "city_trip"],
    "racket_sports_club": ["team_sports_match", "running_session", "yoga_session"],
    "combat_sports_gym": ["martial_arts_class", "running_session", "yoga_session"],
    # digital_creative sub-spaces
    "cosplay_gaming_crew": ["cosplay_contest", "esports_tournament", "anime_screening"],
    "film_dj_collective": ["film_screening_indie", "dj_workshop", "open_mic_live"],
    # craft_creative sub-spaces
    "craft_atelier": ["craft_fair", "life_drawing_session", "art_workshop"],
    "woodwork_print_studio": ["craft_fair", "ceramics_workshop", "life_drawing_session"],
}

PERSONA_SPACE_PREFS: dict[str, list[str]] = {
    "outdoor_adventurer": [
        "mountain_hikers", "climbing_crew", "travel_backpackers", "urban_runners",
        "yoga_wellness", "pet_lovers_club", "language_exchange_lounge", "coffee_explorers",
    ],
    # culture sub-personas
    "visual_culture_fan": [
        "anime_comics_club", "photo_cinema_society", "cinephile_collective",
        "street_photo_crew", "modern_art_collective", "live_music_tribe",
        "coffee_explorers", "travel_backpackers",
    ],
    "performing_arts_fan": [
        "theater_company", "live_music_tribe", "cinephile_collective",
        "jazz_lounge", "vinyl_collectors_club", "improv_theater_group",
        "book_cafe_club", "coffee_explorers",
    ],
    "literary_craft_fan": [
        "ceramics_pottery_studio", "creative_writing_circle", "book_cafe_club",
        "modern_art_collective", "coffee_explorers", "language_exchange_lounge",
        "street_photo_crew", "mindfulness_center",
    ],
    "foodie": [
        "foodie_circle", "wine_tasting_society", "coffee_explorers", "language_exchange_lounge",
        "book_cafe_club", "live_music_tribe", "travel_backpackers", "pet_lovers_club",
    ],
    # sports sub-personas
    "team_sports_fan": [
        "team_sports_club", "racket_sports_club", "urban_runners",
        "travel_backpackers", "language_exchange_lounge", "nerd_hub", "foodie_circle",
    ],
    "fitness_enthusiast": [
        "combat_sports_gym", "urban_runners", "yoga_wellness",
        "mountain_hikers", "climbing_crew", "wellness_studio", "coffee_explorers",
    ],
    # creative sub-personas
    "digital_creative": [
        "cosplay_gaming_crew", "film_dj_collective", "nerd_hub",
        "indie_gaming_club", "music_production_studio", "ai_builders_lab",
        "anime_comics_club", "live_music_tribe",
    ],
    "craft_creative": [
        "craft_atelier", "woodwork_print_studio", "modern_art_collective",
        "ceramics_pottery_studio", "book_cafe_club", "street_photo_crew",
        "fashion_collective", "language_exchange_lounge",
    ],
    "wellness_seeker": [
        "wellness_studio", "mindfulness_center", "holistic_health_hub", "yoga_wellness",
        "mountain_hikers", "coffee_explorers", "book_cafe_club", "pet_lovers_club",
    ],
    "tech_geek": [
        "maker_space", "ai_research_lab", "astronomy_club", "nerd_hub",
        "tech_founders_circle", "ai_builders_lab", "indie_gaming_club", "board_games_society",
    ],
    "music_lover": [
        "music_production_studio", "jazz_lounge", "vinyl_collectors_club", "live_music_tribe",
        "film_dj_collective", "modern_art_collective", "coffee_explorers", "language_exchange_lounge",
    ],
    "social_butterfly": [
        "language_exchange_lounge", "pet_lovers_club", "travel_backpackers", "board_games_society",
        "foodie_circle", "live_music_tribe", "coffee_explorers", "urban_runners",
    ],
    "lifestyle_explorer": [
        "sustainable_living_co", "digital_nomad_hub", "fashion_collective", "travel_backpackers",
        "coffee_explorers", "pet_lovers_club", "language_exchange_lounge", "book_cafe_club",
    ],
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
    """
    Sample a realistic start time by time-slot (morning/afternoon/evening),
    including minute granularity.
    """
    if status == "published":
        slot = random.choices(
            ["morning", "afternoon", "evening"],
            weights=[0.18, 0.32, 0.50],
            k=1,
        )[0]
    else:
        slot = random.choices(
            ["morning", "afternoon", "evening"],
            weights=[0.22, 0.34, 0.44],
            k=1,
        )[0]

    if slot == "morning":
        hour = random.randint(7, 11)
    elif slot == "afternoon":
        hour = random.randint(12, 17)
    else:
        hour = random.randint(18, 22)

    return hour, 0  # round to nearest hour for cleaner time bands


def skewed_age(lo: int, hi: int) -> int:
    """
    Beta-distributed age skewed toward the lower third of [lo, hi].
    Models the reality that dating app users peak in their mid-20s.
    """
    raw = random.betavariate(2.0, 3.5)   # peaks ~36% of range
    return lo + round(raw * (hi - lo))


def power_law_popularity(n: int, exponent: float = 1.8) -> list[float]:
    """
    Returns n popularity scores following a power law (Zipf-like).
    A few items are very popular, most are niche.
    """
    ranks  = list(range(1, n + 1))
    random.shuffle(ranks)
    scores = [1.0 / (r ** exponent) for r in ranks]
    max_s  = max(scores)
    return [s / max_s for s in scores]


# ─── Tag sampling ──────────────────────────────────────────────────────────────

def sample_user_tags(persona: dict) -> dict[str, float]:
    """
    Sample dense user interests biased toward the persona's tag_subpool.

    Tier breakdown:
      core (2-4):  almost exclusively from tag_subpool  → very high weight
      mid  (4-7):  subpool remainder + cluster spillover + small secondary
      low  (rest): random exploration across all tags   → low weight
    """
    prim    = TAG_CLUSTERS[persona["cluster"]]
    subpool = persona.get("tag_subpool", prim)
    sec_pool = [t for t in TAG_VOCAB if t not in prim]

    n_total = random.randint(14, 24)
    n_core  = random.randint(2, 4)
    n_mid   = random.randint(4, 7)
    n_low   = max(0, n_total - n_core - n_mid)

    # Core: drawn almost entirely from subpool
    core_tags = random.sample(subpool, min(n_core, len(subpool)))

    # Mid: subpool remainder first, then rest-of-cluster, then secondary
    remaining_sub  = [t for t in subpool if t not in core_tags]
    remaining_prim = [t for t in prim    if t not in core_tags and t not in subpool]

    n_mid_sub  = min(len(remaining_sub),  max(0, round(n_mid * 0.55)))
    n_mid_prim = min(len(remaining_prim), max(0, round(n_mid * 0.20)))
    n_mid_sec  = max(0, n_mid - n_mid_sub - n_mid_prim)

    mid_tags  = random.sample(remaining_sub,  n_mid_sub)
    mid_tags += random.sample(remaining_prim, n_mid_prim)
    mid_tags += random.sample(sec_pool, min(n_mid_sec, len(sec_pool)))

    low_pool = [t for t in TAG_VOCAB if t not in set(core_tags + mid_tags)]
    low_tags = random.sample(low_pool, min(n_low, len(low_pool)))

    result: dict[str, float] = {}
    for tag in core_tags:
        result[tag] = round(random.uniform(0.75, 1.00), 2)
    for tag in mid_tags:
        result[tag] = round(random.uniform(0.28, 0.65), 2)
    for tag in low_tags:
        result[tag] = round(random.uniform(0.03, 0.20), 2)
    return result


def bump_user_tag_weights(
    user: dict,
    item_tags: list[str],
    strength: float,
    decay_existing: float = 0.0,
) -> None:
    """
    Simulates incremental preference updates after page visits/interactions.
    Users accumulate many low signals and a few strong ones.
    """
    if not item_tags:
        return
    tw = user["tag_weights"]
    if decay_existing > 0:
        for tag, w in list(tw.items()):
            tw[tag] = max(0.0, round(w * (1.0 - decay_existing), 4))
    for tag in item_tags:
        current = float(tw.get(tag, 0.0))
        inc = strength * (1.0 - min(1.0, current))
        tw[tag] = round(min(1.25, current + inc), 4)


def _sample_tag_count(min_n: int, max_n: int, weights: list[float]) -> int:
    values = list(range(min_n, max_n + 1))
    return random.choices(values, weights=weights, k=1)[0]


def sample_space_tags(archetype: dict) -> list[str]:
    n = _sample_tag_count(1, 5, [0.10, 0.20, 0.40, 0.20, 0.10])
    pool = [t for t in archetype["tag_pool"] if t in TAG_VOCAB]
    if len(pool) < n:
        fallback = [t for t in TAG_VOCAB if t not in pool]
        pool = pool + random.sample(fallback, min(n - len(pool), len(fallback)))
    return random.sample(pool, min(n, len(pool)))


def sample_event_tags(space_tags: list[str], event_archetype: dict) -> list[str]:
    n = _sample_tag_count(1, 4, [0.15, 0.35, 0.35, 0.15])
    n_from_space = max(1, min(len(space_tags), round(n * random.uniform(0.7, 1.0))))
    chosen = set(random.sample(space_tags, n_from_space)) if space_tags else set()

    event_pool = [t for t in event_archetype["tag_pool"] if t in TAG_VOCAB and t not in chosen]
    if len(chosen) < n and event_pool:
        chosen |= set(random.sample(event_pool, min(n - len(chosen), len(event_pool))))

    if len(chosen) < n:
        fallback = [t for t in TAG_VOCAB if t not in chosen]
        chosen |= set(random.sample(fallback, min(n - len(chosen), len(fallback))))

    return list(chosen)


def sample_entity_tags(preferred_cluster: int = -1) -> list[str]:
    if preferred_cluster >= 0:
        pool = list(TAG_CLUSTERS[preferred_cluster])
    else:
        pool = list(random.choice(TAG_CLUSTERS))
    n = _sample_tag_count(1, 4, [0.15, 0.35, 0.35, 0.15])
    if len(pool) < n:
        rest = [t for t in TAG_VOCAB if t not in pool]
        pool += random.sample(rest, min(n - len(pool), len(rest)))
    return random.sample(pool, min(n, len(pool)))


def _tag_score(weight: float | None) -> float:
    if weight is None:
        return -1.0
    if weight <= 0.10:
        return -0.35
    if weight <= 0.40:
        return 0.15
    return 1.0


def _implicit_tag_preference_factor(user_tag_weights: dict[str, float], item_tags: list[str]) -> float:
    if not item_tags:
        return 0.75
    scores = [_tag_score(user_tag_weights.get(tag)) for tag in item_tags]
    mean_score = sum(scores) / len(scores)
    norm = (mean_score + 1.0) / 2.0
    return 0.55 + (norm ** 0.85) * 0.50


# ─── Entity generators ─────────────────────────────────────────────────────────

def gen_user(persona_idx: int) -> dict:
    p     = PERSONAS[persona_idx]
    lo, hi = p["age_range"]
    today  = date.today()
    age    = skewed_age(lo, hi)
    bd     = today - timedelta(days=round(age * 365.25))

    n_ri  = random.randint(1, 3)
    ri    = random.choices(REL_INTENTS, weights=p["rel_intent_w"], k=n_ri)
    ri    = list(dict.fromkeys(ri))  # deduplicate keeping order

    return {
        "id":                  uid(),
        "persona":             p["name"],          # informational only, not in real export
        "persona_idx":         persona_idx,        # used internally for interaction generation
        "birthdate":           bd.isoformat(),
        "gender":              wc(GENDERS, p["gender_w"]),
        "relationship_intent": ri,
        "smoking":             wc(SMOKING,   p["smoking_w"])              if random.random() > 0.10 else None,
        "drinking":            wc(DRINKING,  p["drinking_w"])             if random.random() > 0.10 else None,
        "activity_level":      wc(p["activity_choices"], p["activity_w"]) if random.random() > 0.10 else None,
        "interaction_count":   0,
        "tag_weights":         (_tw := sample_user_tags(p)),  # internal only — used for interaction simulation
        "tags":                list(_tw.keys()),               # exported to JSON (same sample)
    }


def gen_event(space_id: str, status: str, preferred_cluster: int, popularity: float) -> dict:
    today = date.today()
    if status == "completed":
        starts      = rand_date(today - timedelta(days=365), today - timedelta(days=1))
        base_count  = round(10 + popularity * 70)
        a_count     = round(random.gauss(base_count, base_count * 0.25))
        a_count     = max(2, a_count)
        avg_age     = round(random.gauss(28 + preferred_cluster * 2, 5), 1)
    else:
        starts      = rand_date(today + timedelta(days=1), today + timedelta(days=180))
        a_count     = round(random.gauss(popularity * 25, 5))
        a_count     = max(0, a_count)
        avg_age     = round(random.gauss(28 + preferred_cluster * 2, 5), 1) if a_count > 0 else None

    max_att = random.choice([None, None, 20, 30, 50, 100, 200])
    price   = 0 if random.random() < 0.65 else random.choice([5, 10, 15, 20, 30])
    start_hour, start_minute = sample_time_slot(status)

    return {
        "id":               uid(),
        "space_id":         space_id,
        "tags":             sample_entity_tags(preferred_cluster),
        "starts_at":        f"{starts} {start_hour:02d}:{start_minute:02d}:00",
        "max_attendees":    max_att,
        "is_paid":          price > 0,
        "price_cents":      price * 100,
        "attendee_count":   a_count,
        "avg_attendee_age": avg_age,
        "preferred_cluster": preferred_cluster,
        "popularity":        round(popularity, 3),
    }


def gen_space(archetype: dict, popularity: float) -> dict:
    return {
        "id":             uid(),
        "tags":           sample_space_tags(archetype),
        "member_count":   0,
        "avg_member_age": None,
        "event_count":    0,
        "preferred_cluster": archetype["cluster"],
        "archetype":      archetype["name"],
        "popularity":        round(popularity, 3),
    }


# ─── Interaction generation ────────────────────────────────────────────────────

def compute_item_persona_affinity(tags: list[str]) -> list[float]:
    """Returns a vector (one entry per cluster) with the fraction of tags belonging to each cluster."""
    counts = [0] * len(TAG_CLUSTERS)
    for tag in tags:
        ci = TAG_TO_CLUSTER.get(tag)
        if ci is not None:
            counts[ci] += 1
    total = sum(counts) or 1
    return [c / total for c in counts]


def _per_user_counts(
    n_users: int,
    target_total: int,
    event_ratio: float = 0.70,
    min_per_user: int = 5,
) -> list[tuple[int, int]]:
    """
    Sample per-user interaction counts from a lognormal distribution,
    mirroring real app engagement (power-law: few power-users, many casual).

    Returns [(n_events, n_spaces), ...] per user, scaled to target_total.
    """
    raw = [
        max(min_per_user, round(random.lognormvariate(math.log(12), 0.85)))
        for _ in range(n_users)
    ]
    scale = target_total / max(1, sum(raw))
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

    Rules:
    1. PERSONA COHERENCE: users sample events/spaces primarily from their
       persona's pool (items with high affinity for their cluster).
    2. POPULARITY BOOST: high-popularity items have higher base probability.
    3. SERENDIPITY: 8% of each user's interactions are cross-persona (exploration).
    4. POWER-LAW engagement: per-user counts follow a lognormal distribution.
    """
    n_users     = len(users)
    user_counts = _per_user_counts(n_users, n_interactions)

    for e in events:
        e["_affinity"] = compute_item_persona_affinity(e["tags"])
    for s in spaces:
        s["_affinity"] = compute_item_persona_affinity(s["tags"])

    n_personas = len(PERSONAS)
    persona_event_pools: list[list[str]] = []
    persona_space_pools: list[list[str]] = []

    for p_idx in range(n_personas):
        ci   = PERSONAS[p_idx]["cluster"]   # cluster index (0-9) for affinity lookup
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
    user_event_cnt: dict[str, int] = defaultdict(int)
    user_space_cnt: dict[str, int] = defaultdict(int)
    space_member_data: dict[str, list[dict]] = defaultdict(list)
    event_by_id = {e["id"]: e for e in events}
    space_by_id = {s["id"]: s for s in spaces}
    user_by_id  = {u["id"]: u for u in users}
    today = date.today()

    def _recency(d: date) -> float:
        return math.exp(-(today - d).days / 180.0)

    def _add(user_id: str, iid: str, itype: str) -> bool:
        if (user_id, iid) not in positive_pairs:
            positive_pairs.add((user_id, iid))
            days_back  = round(random.betavariate(1.5, 3.0) * 364) + 1
            created_at = today - timedelta(days=days_back)
            user  = user_by_id[user_id]
            p_idx = user["persona_idx"]
            ci    = PERSONAS[p_idx]["cluster"]  # cluster index for affinity lookup
            pref_factor = 1.0
            if itype == "event":
                event = event_by_id[iid]
                try:
                    event_day = date.fromisoformat(str(event["starts_at"]).split(" ")[0])
                except (TypeError, ValueError):
                    event_day = today
                type_w = 1.0 if event_day < today else 0.7
                pref_score = float(event.get("_affinity", [0.0] * len(TAG_CLUSTERS))[ci])
                persona_factor = 0.35 + 0.65 * pref_score
                tag_factor = _implicit_tag_preference_factor(user["tag_weights"], event["tags"])
                pref_factor = persona_factor * tag_factor
                bump_user_tag_weights(user, event["tags"], strength=0.02 + 0.08 * pref_factor)
            else:
                space = space_by_id[iid]
                type_w = 0.9
                pref_score = float(space.get("_affinity", [0.0] * len(TAG_CLUSTERS))[ci])
                persona_factor = 0.35 + 0.65 * pref_score
                tag_factor = _implicit_tag_preference_factor(user["tag_weights"], space["tags"])
                pref_factor = persona_factor * tag_factor
                bump_user_tag_weights(user, space["tags"], strength=0.015 + 0.06 * pref_factor)

            interactions.append({
                "user_id":    user_id,
                "item_id":    iid,
                "item_type":  itype,
                "weight":     round(type_w * pref_factor * _recency(created_at), 4),
                "created_at": created_at.isoformat(),
            })
            return True
        return False

    for user, (n_ev, n_sp) in zip(users, user_counts):
        user_id = user["id"]
        p_idx   = user["persona_idx"]

        # ── Events ──────────────────────────────────────────────────────────
        pool = persona_event_pools[p_idx]
        # 92% from persona-aligned top-third, 8% serendipity
        top  = pool[: max(1, len(pool) // 3)]
        rest = pool[len(pool) // 3 :]

        n_top  = max(0, round(n_ev * 0.92))
        n_rest = n_ev - n_top
        sampled_ev  = random.sample(top,  min(n_top,  len(top)))
        sampled_ev += random.sample(rest, min(n_rest, len(rest)))
        if len(sampled_ev) < n_ev:
            extra = [e for e in all_event_ids if e not in sampled_ev]
            sampled_ev += random.sample(extra, min(n_ev - len(sampled_ev), len(extra)))

        for eid in sampled_ev:
            if _add(user_id, eid, "event"):
                user_event_cnt[user_id] += 1

        # ── Spaces ──────────────────────────────────────────────────────────
        pool = persona_space_pools[p_idx]
        top  = pool[: max(1, len(pool) // 3)]
        rest = pool[len(pool) // 3 :]

        n_top  = max(0, round(n_sp * 0.92))
        n_rest = n_sp - n_top
        sampled_sp  = random.sample(top,  min(n_top,  len(top)))
        sampled_sp += random.sample(rest, min(n_rest, len(rest)))

        for sid in sampled_sp:
            if _add(user_id, sid, "space"):
                space_member_data[sid].append(user)
                user_space_cnt[user_id] += 1

    for user in users:
        user["interaction_count"] = user_event_cnt[user["id"]] + user_space_cnt[user["id"]]

    for space in spaces:
        members = space_member_data[space["id"]]
        space["member_count"] = len(members)
        ages = []
        for m in members:
            if m.get("birthdate"):
                bd = date.fromisoformat(m["birthdate"])
                ages.append((date.today() - bd).days / 365.25)
        space["avg_member_age"] = round(sum(ages) / len(ages), 1) if ages else None

    return interactions


# ─── Strip internal fields before writing ──────────────────────────────────────

def _clean_user(u: dict) -> dict:
    # Remove internal-only fields; tag_weights is internal, tags is the exported field
    return {k: v for k, v in u.items() if k not in ("persona", "persona_idx", "tag_weights")}

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
        cluster = i % len(TAG_CLUSTERS)
        cluster_candidates = [a for a in SPACE_ARCHETYPES if a["cluster"] == cluster]
        if cluster_candidates and random.random() < 0.75:
            archetype = random.choice(cluster_candidates)
        else:
            archetype = random.choice(SPACE_ARCHETYPES)
        spaces.append(gen_space(archetype, pop))

    event_popularities = power_law_popularity(n_events)
    events = []
    for i, pop in enumerate(event_popularities):
        cluster = i % len(TAG_CLUSTERS)
        status  = "completed" if i < round(n_events * 0.6) else "published"
        events.append(gen_event(random.choice(spaces)["id"], status, cluster, pop))

    event_counts: dict[str, int] = defaultdict(int)
    for e in events:
        event_counts[e["space_id"]] += 1
    for space in spaces:
        space["event_count"] = event_counts[space["id"]]

    print("  Generating users...")
    users = []
    for i in range(n_users):
        p_idx = i % len(PERSONAS)
        users.append(gen_user(p_idx))
    random.shuffle(users)

    print("  Generating interactions...")
    interactions = assign_interactions(users, events, spaces, n_interactions)

    def _write(name: str, data: list) -> None:
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ {name:<25} {len(data):>8,} records")

    print()
    _write("users.json",        [_clean_user(u) for u in users])
    _write("events.json",       [_clean_event(e) for e in events])
    _write("spaces.json",       [_clean_space(s) for s in spaces])
    _write("interactions.json", interactions)

    n_pos = len(interactions)
    print(f"\n  positive pairs : {n_pos:,}  (target {n_interactions:,})")
    print(f"  avg per user   : {n_pos / n_users:.1f}")

    by_type = defaultdict(int)
    for ix in interactions:
        by_type[ix["item_type"]] += 1
    print(f"\n  interaction breakdown:")
    for t, c in sorted(by_type.items()):
        print(f"    {t:<8}  {c:>8,}  ({100*c/n_pos:.1f}%)")

    avg_w = sum(ix["weight"] for ix in interactions) / max(1, n_pos)
    print(f"\n  avg weight     : {avg_w:.3f}  (1.0 = very recent attended, ~0.1 = older interaction)")


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
