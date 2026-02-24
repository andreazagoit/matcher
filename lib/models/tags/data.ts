/**
 * Shared tag vocabulary used across Profiles, Events, and Spaces.
 *
 * Tags are plain strings. Localization is handled at the UI layer.
 * Order within each category matters — must stay in sync with
 * python-ml/ml/config.py TAG_VOCAB (same order, same strings).
 */

export const TAG_CATEGORIES: Record<string, string[]> = {
  outdoor: [
    "trekking", "camping", "climbing", "cycling", "beach", "mountains",
    "gardening", "surfing", "skiing", "kayaking", "fishing", "trail_running",
    "snowboarding", "skateboarding", "horse_riding", "sailing", "scuba_diving",
    "paragliding", "bouldering", "canyoning",
  ],
  culture: [
    "cinema", "theater", "live_music", "museums", "reading", "photography",
    "art", "opera", "ballet", "comedy_shows", "podcasts", "architecture",
    "vintage", "anime", "comics", "street_art", "literary_clubs", "calligraphy",
    "sculpture", "ceramics",
  ],
  food_drink: [
    "cooking", "restaurants", "wine", "craft_beer", "street_food", "coffee",
    "veganism", "sushi", "cocktails", "baking", "tea", "barbecue",
    "food_festivals", "cheese", "sake", "mixology", "foraging", "fermentation",
    "ramen", "pastry",
  ],
  sports: [
    "running", "gym", "yoga", "swimming", "football", "tennis", "padel",
    "basketball", "volleyball", "boxing", "martial_arts", "pilates", "crossfit",
    "golf", "rugby", "archery", "dance", "rowing", "hockey", "badminton",
  ],
  creative: [
    "music", "drawing", "writing", "diy", "gaming", "coding", "painting",
    "pottery", "knitting", "woodworking", "film_making", "singing", "djing",
    "digital_art", "cosplay", "jewelry_making", "embroidery", "leathercraft",
    "printmaking", "glass_blowing",
  ],
  wellness: [
    "meditation", "mindfulness", "journaling", "spa", "breathwork", "nutrition",
    "therapy", "cold_exposure", "sound_healing", "ayurveda", "reiki",
    "stretching", "sleep_hygiene", "herbal_medicine", "qi_gong", "forest_bathing",
    "intermittent_fasting", "positive_psychology", "aromatherapy",
    "somatic_practices",
  ],
  tech_science: [
    "ai", "blockchain", "cybersecurity", "robotics", "vr_ar", "open_source",
    "data_science", "smart_home", "esports", "astronomy", "chemistry", "biology",
    "engineering", "drones", "quantum_computing", "3d_printing",
    "space_exploration", "neuroscience", "environmental_science", "biohacking",
  ],
  music_genres: [
    "jazz", "classical_music", "hip_hop", "electronic_music", "rock",
    "indie_music", "reggae", "pop_music", "metal", "country_music", "blues",
    "r_and_b", "folk_music", "gospel", "bossa_nova", "afrobeat", "techno",
    "house_music", "punk", "soul",
  ],
  social: [
    "travel", "volunteering", "languages", "pets", "parties", "board_games",
    "networking", "karaoke", "escape_rooms", "trivia_nights", "activism",
    "mentoring", "astrology", "tarot", "stand_up", "improv", "storytelling",
    "cultural_exchange", "language_exchange", "community_events",
  ],
  lifestyle: [
    "fashion", "interior_design", "sustainability", "minimalism", "van_life",
    "urban_exploration", "thrifting", "luxury_lifestyle", "tattoos",
    "personal_growth", "entrepreneurship", "parenting", "spirituality",
    "digital_nomad", "homesteading", "zero_waste", "slow_living", "nightlife",
    "brunch_culture", "self_improvement",
  ],
};

// ─── Flat list and validation ───────────────────────────────────────

export const ALL_TAGS: string[] = Object.values(TAG_CATEGORIES).flat();

const TAG_SET = new Set(ALL_TAGS);

export function isValidTag(tag: string): boolean {
  return TAG_SET.has(tag);
}
