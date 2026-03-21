import logging
from typing import Dict

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- CONFIDENCE CALCULATION ---------------- #
def calculate_confidence(detected_name: str, geo_result: Dict) -> float:
    """
    Calculates a confidence score (0.0 to 1.0) for a geocoded result.

    Factors considered:
    - String similarity between detected name and display name.
    - Type of the result (e.g., city, attraction, park).
    - Presence of address details / importance score from Nominatim.

    Args:
        detected_name (str): Name detected from text or metadata.
        geo_result (dict): Geocoded result from Nominatim.

    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    score = 0.5  # Base score

    display_name = geo_result.get("display_name", "").lower()
    detected_name_lower = detected_name.lower()

    # 1. String match score
    if detected_name_lower in display_name:
        score += 0.2

    # 2. Importance score from Nominatim (default 0.5)
    importance = geo_result.get("importance", 0.5)
    score += (importance * 0.2)

    # 3. Category relevance
    place_type = geo_result.get("type", "")
    high_relevance_types = ["attraction", "tourism", "park", "museum", "monument", "beach"]
    if place_type in high_relevance_types:
        score += 0.1

    final_score = min(1.0, score)
    logger.info(f"Calculated confidence for '{detected_name}' ({place_type}): {final_score:.2f}")

    return final_score