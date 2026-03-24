import os
import asyncio
import logging
import re
from typing import List, Dict, Tuple, Optional
from config import NOMINATIM_USER_AGENT, CONFIDENCE_THRESHOLD
import httpx

from utils.scoring import calculate_confidence

# ---------------- CONFIG ---------------- #
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
BASE_URL = "https://nominatim.openstreetmap.org/search"

REQUEST_DELAY = 2.0   # 🔥 Increased (VERY IMPORTANT for Render)
MAX_RETRIES = 2       # avoid too many retries
MAX_LOCATIONS = 5     # 🔥 LIMIT requests (critical fix)

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- CACHE ---------------- #
_location_cache: Dict[str, List[dict]] = {}

# ---------------- CLEAN CONTEXT ---------------- #
def clean_context(context: Optional[str]) -> Optional[str]:
    if not context:
        return None
    # remove emojis/special chars like 1️⃣
    context = re.sub(r'[^\w\s,]', '', context)
    return context.strip()


# ---------------- MATCHING ---------------- #
def is_match_flexible(input_name: str, result: dict) -> bool:
    input_name = input_name.lower()
    display_name = result.get("display_name", "").lower()

    if input_name in display_name:
        return True

    words = [w for w in input_name.split() if len(w) > 3]
    if not words:
        return False

    return any(w in display_name for w in words)


# ---------------- API CALL ---------------- #
async def fetch_location_data(client: httpx.AsyncClient, name: str, context: Optional[str]) -> List[dict]:
    context = clean_context(context)

    query = f"{name}, {context}" if context else name
    cache_key = query.lower()

    if cache_key in _location_cache:
        logger.info(f"Cache hit: {query}")
        return _location_cache[cache_key]

    params = {
        "q": query,
        "format": "json",
        "limit": 5,  # 🔥 reduce load
        "addressdetails": 1
    }

    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    backoff = 1.5

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(BASE_URL, params=params, headers=headers)

            if response.status_code == 429:
                logger.warning(f"429 for {name}, retrying in {backoff}s")
                await asyncio.sleep(backoff)
                backoff *= 2
                continue

            if response.status_code != 200:
                logger.error(f"API error {response.status_code} for {name}")
                return []

            data = response.json()
            _location_cache[cache_key] = data
            return data

        except httpx.RequestError as e:
            logger.error(f"Network error: {e}")
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

    return city, address.get("state", ""), address.get("country", "")


# ---------------- MAIN VALIDATION ---------------- #
async def validate_locations(location_names: List[str], context: Optional[str] = None) -> List[Dict]:
    if not location_names:
        return []

    # 🔥 LIMIT locations (CRITICAL for Render)
    location_names = list(set(location_names))[:MAX_LOCATIONS]

    validated_places = []

    async with httpx.AsyncClient(timeout=10) as client:
        for name in location_names:
            logger.info(f"Processing: {name}")

            results = await fetch_location_data(client, name, context)

            # fallback without context
            if not results and context:
                results = await fetch_location_data(client, name, None)

            if not results:
                logger.warning(f"No results for {name}")
                continue

            best_match, confidence = get_best_match(name, results)

            if best_match:
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
            else:
                fallback = results[0]
                city, state, country = extract_address_fields(fallback)
                validated_places.append({
                    "name": name,
                    "city": city,
                    "state": state,
                    "country": country,
                    "latitude": float(fallback.get("lat", 0)),
                    "longitude": float(fallback.get("lon", 0)),
                    "confidence": 0.3
                })

            # 🔥 VERY IMPORTANT
            await asyncio.sleep(REQUEST_DELAY)

    logger.info(f"FINAL: {validated_places}")
    return validated_places