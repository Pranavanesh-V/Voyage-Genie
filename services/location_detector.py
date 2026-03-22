import os
import re
import logging
from collections import Counter
from typing import List, Dict, Optional

import spacy

# ---------------- CONFIG ---------------- #
MIN_SCORE = int(os.getenv("MIN_SCORE", 4))

TRAVEL_KEYWORDS = [
    "Beach", "Temple", "Fort", "Mountain", "Peak", "Lake", "Garden", "Falls",
    "Museum", "Park", "Palace", "Resort", "Sanctuary", "Valley", "Church",
    "Mosque", "Square", "Bridge", "Tower", "Monastery", "Caves", "National Park",
    "Viewpoint", "Hill", "Dam", "Statue", "Cathedral", "Basilica", "Zoo",
    "Point", "Island", "Waterfall", "Canyon", "Gorge", "Mansion", "Castle",
    "Fortress", "Memorial", "Gallery", "Bazaar", "Market", "Street", "Lane",
    "Road", "Highway", "Plaza", "Mall", "Village", "Town", "Estate", "Plantation",
    "Forest", "Reserve", "Meadow", "Basin", "Harbor", "Harbour", "Port", "Pier",
    "Marina", "Stadium", "Arena", "Library", "Chapel", "Shrine", "Ashram",
    "Pagoda", "Stupa", "Rock", "Cliff", "Ridge", "Plateau", "Aquarium", "Cafe",
    "Bay"
]

BLACKLIST = {
    "Trip", "Travel", "Day", "Guide", "Itinerary", "Budget", "Tour",
    "Vacation", "Holiday", "Morning", "Night", "Hotel", "Flight",
    "Airport", "Visit", "Exploring", "Instagram", "Reel", "Video",
    "Follow", "Like", "Subscribe", "Comment",
}

BAD_WORDS = [
    "rs", "hours", "day", "budget", "itinerary", "time",
    "activities", "read", "caption", "follow", "like",
    "comment", "visit", "best", "top"
]

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- SPACY ---------------- #
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Run: python -m spacy download en_core_web_sm")
    return _nlp


# ---------------- HELPERS ---------------- #
def is_valid_place_name(name: str) -> bool:
    """Check if a place name is valid"""
    if any(char.isdigit() for char in name):
        return False
    if len(name.split()) > 5:
        return False
    if any(word in name.lower() for word in BAD_WORDS):
        return False
    return True


def detect_main_location(doc) -> Optional[str]:
    """Detect main GPE in text"""
    gpe_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    if not gpe_entities:
        return None
    return Counter(gpe_entities).most_common(1)[0][0]


def remove_substrings(locations: List[str]) -> List[str]:
    """Remove substrings that are included in other locations"""
    result = []
    for loc in locations:
        if not any(
            loc.lower() != other.lower() and loc.lower() in other.lower()
            for other in locations
        ):
            result.append(loc)
    return result


# ---------------- MAIN FUNCTION ---------------- #
def find_locations_in_text(text: str) -> Dict[str, Optional[List[str]]]:
    """
    Extract place names from text using SpaCy entities and regex keyword matching
    Returns dict with 'places' and 'context'
    """
    if not text:
        return {"places": [], "context": None}

    text = re.sub(r'\s+', ' ', text)
    doc = get_nlp()(text)

    extracted: List[str] = []

    # Extract entities
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"]:
            if ent.text not in BLACKLIST:
                extracted.append(ent.text.strip())

    # Regex-based keyword extraction
    for kw in TRAVEL_KEYWORDS:
        pattern = rf"\b(?:The\s+)?([A-Z][a-z']+(?:\s+[A-Z][a-z']+){{0,3}}\s+{kw})\b"
        extracted.extend(re.findall(pattern, text))

    scored: List[tuple[str, int]] = []

    for loc in extracted:
        clean = loc.strip(",. \n\t-").title()
        if clean.lower().startswith("the "):
            clean = clean[4:]

        if not is_valid_place_name(clean):
            continue

        score = sum(3 for kw in TRAVEL_KEYWORDS if kw.lower() in clean.lower())
        score += len(clean.split())

        if score >= MIN_SCORE:
            scored.append((clean, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate
    seen = set()
    ordered: List[str] = []
    for loc, _ in scored:
        if loc.lower() not in seen:
            ordered.append(loc)
            seen.add(loc.lower())

    final_locations = remove_substrings(ordered)[:20]

    context = detect_main_location(doc)

    logger.info(f"Places: {final_locations}, Context: {context}")

    return {
        "places": final_locations,
        "context": context
    }

