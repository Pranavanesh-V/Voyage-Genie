import os
import asyncio
import logging
from typing import List, Dict, Tuple, Optional
from config import NOMINATIM_USER_AGENT, CONFIDENCE_THRESHOLD
import httpx

from utils.scoring import calculate_confidence

# ---------------- CONFIG ---------------- #
#NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "VoyageGenie/1.0")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
BASE_URL = "https://nominatim.openstreetmap.org/search"
REQUEST_DELAY = 1.0  # seconds between requests
MAX_RETRIES = 3

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- CACHE ---------------- #
# Simple in-memory cache to reduce repeated API hits
_location_cache: Dict[str, List[dict]] = {}


# ---------------- MATCHING ---------------- #
def is_match_flexible(input_name: str, result: dict) -> bool:
    input_name = input_name.lower()
    display_name = result.get("display_name", "").lower()

    if input_name in display_name:
        return True

    words = [w for w in input_name.split() if len(w) > 3]
    if not words:
        return False

    match_count = sum(1 for w in words if w in display_name)
    return match_count >= 1


# ---------------- API CALL ---------------- #
async def fetch_location_data(name: str, context: Optional[str] = None) -> List[dict]:
    """Fetch location data from Nominatim with retries and backoff"""
    if context and len(context) > 3:
        query = f"{name}, {context}"
    else:
        query = name

    # Return cached results if available
    cache_key = query.lower()
    if cache_key in _location_cache:
        logger.info(f"Cache hit for '{query}'")
        return _location_cache[cache_key]

    params = {
        "q": query,
        "format": "json",
        "limit": 15,
        "addressdetails": 1,
        "accept-language": "en"
    }
    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    backoff = 1.0

    async with httpx.AsyncClient(timeout=10) as client:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.get(BASE_URL, params=params, headers=headers)

                if response.status_code == 429:
                    logger.warning(f"Rate limited for '{name}', retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                if response.status_code != 200:
                    logger.error(f"API error {response.status_code} for '{name}'")
                    return []

                data = response.json()
                _location_cache[cache_key] = data
                logger.info(f"{name} → {len(data)} results")
                return data

            except httpx.RequestError as e:
                logger.error(f"Network error for '{name}': {e}")
                await asyncio.sleep(backoff)
                backoff *= 2

    return []


# ---------------- BEST MATCH ---------------- #
def get_best_match(name: str, results: List[dict]) -> Tuple[Optional[dict], float]:
    best_match = None
    best_score = 0

    for result in results:
        if not is_match_flexible(name, result):
            continue

        score = calculate_confidence(name, result)

        if name.lower() in result.get("display_name", "").lower():
            score += 0.2

        if score > best_score:
            best_score = score
            best_match = result

    return best_match, best_score


# ---------------- ADDRESS EXTRACTION ---------------- #
def extract_address_fields(result: dict):
    address = result.get("address", {})

    city = (
        address.get("city") or
        address.get("town") or
        address.get("village") or
        address.get("county") or
        ""
    )
    state = address.get("state", "")
    country = address.get("country", "")

    return city, state, country


# ---------------- MAIN VALIDATION ---------------- #
async def validate_locations(location_names: List[str], context: Optional[str] = None) -> List[Dict]:
    if not location_names:
        return []

    validated_places = []

    for name in location_names:
        logger.info(f"🔍 Processing: {name} | Context: {context}")

        results = await fetch_location_data(name, context)

        # fallback without context
        if not results and context:
            logger.info("🔁 Retrying without context...")
            results = await fetch_location_data(name, None)

        if not results:
            logger.warning(f"❌ No results found for '{name}'")
            continue

        best_match, confidence = get_best_match(name, results)
        threshold = max(0.3, CONFIDENCE_THRESHOLD - 0.1)

        if best_match and confidence >= threshold:
            city, state, country = extract_address_fields(best_match)
            validated_places.append({
                "name": name,
                "city": city,
                "state": state,
                "country": country,
                "latitude": float(best_match.get("lat", 0)),
                "longitude": float(best_match.get("lon", 0)),
                "confidence": round(confidence, 2)
            })
            logger.info("✔ Accepted")
        else:
            # fallback to first result
            fallback = results[0]
            city, state, country = extract_address_fields(fallback)
            validated_places.append({
                "name": name,
                "city": city,
                "state": state,
                "country": country,
                "latitude": float(fallback.get("lat", 0)),
                "longitude": float(fallback.get("lon", 0)),
                "confidence": round(confidence if confidence else 0.3, 2)
            })
            logger.warning("⚠ Fallback used")

        await asyncio.sleep(REQUEST_DELAY)

    logger.info(f"🎯 FINAL VALIDATED: {validated_places}")
    return validated_places


