import logging
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.metadata_extractor import extract_metadata
from services.location_detector import find_locations_in_text

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- FASTAPI ---------------- #
app = FastAPI(title="VoyageGenie Reel Analytics API")

# ------------------ MODELS ------------------ #
class ReelRequest(BaseModel):
    reelUrl: str

class Place(BaseModel):
    name: str
    city: Optional[str] = ""
    state: Optional[str] = ""
    country: Optional[str] = ""

class ReelResponse(BaseModel):
    caption: str
    hashtags: List[str]
    detected_locations: List[str]
    validated_places: List[Place]

# ------------------ UTILITIES ------------------ #
def format_place_name(place):
    """Combine name, city, country for display."""
    parts = [place.get("name", "")]
    if place.get("city"):
        parts.append(place["city"])
    if place.get("country"):
        parts.append(place["country"])
    return ", ".join(parts)

# ------------------ API ENDPOINTS ------------------ #
@app.post("/process-reel", response_model=ReelResponse)
async def process_reel(request: ReelRequest):
    try:
        # 1. Extract metadata (caption, hashtags)
        metadata = extract_metadata(request.reelUrl)
        caption = metadata.get("caption", "")
        hashtags = metadata.get("hashtags", [])

        # 2. Detect locations from caption + hashtags
        combined_text = f"{caption} {' '.join(hashtags)}"
        location_data = find_locations_in_text(combined_text)
        detected_locations = location_data.get("places", [])
        location_context = location_data.get("context", "")

        # 3. Format validated places
        validated_places = [
            Place(
                name=format_place_name(place),
                city=place.get("city", ""),
                state=place.get("state", ""),
                country=place.get("country", "")
            )
            for place in detected_locations
        ]

        # 4. Return results
        return {
            "caption": caption,
            "hashtags": hashtags,
            "detected_locations": [place.get("name", "") for place in detected_locations],
            "validated_places": validated_places
        }

    except Exception as e:
        logger.error(f"Error processing reel: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ HEALTH CHECK ------------------ #
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
def home():
    return {"message": "VoyageGenie AI backend running"}
