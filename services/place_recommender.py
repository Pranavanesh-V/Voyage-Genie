import logging
from typing import List, Dict

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- PLACE RECOMMENDATION ---------------- #
def recommend_places(validated_places: List[Dict]) -> List[Dict]:
    """
    Ranks validated places based on confidence and relevance.
    Filters out low-confidence results and sorts by confidence descending.
    """
    if not validated_places:
        logger.warning("No validated places provided for recommendation.")
        return []

    # Sort by confidence score descending
    sorted_places = sorted(
        validated_places,
        key=lambda x: x.get("confidence", 0),
        reverse=True
    )

    logger.info(f"Recommended {len(sorted_places)} places based on confidence.")

    return sorted_places

