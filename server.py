import uuid
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.metadata_extractor import extract_metadata
from services.location_detector import find_locations_in_text
from services.geo_validator import validate_locations  # <-- your updated validator

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
    latitude: float
    longitude: float
    confidence: float


class ReelResponse(BaseModel):
    caption: str
    hashtags: List[str]
    detected_locations: List[str]
    validated_places: List[Place]


# ------------------ API ------------------ #
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
        detected_locations = location_data.get("places", [])  # list of strings
        location_context = location_data.get("context", "")

        # 3. Validate locations asynchronously using your Nominatim validator
        validated_places_data = await validate_locations(detected_locations, location_context)

        # 4. Convert to Place objects
        validated_places = [Place(**place) for place in validated_places_data]

        # 5. Return API response
        return {
            "caption": caption,
            "hashtags": hashtags,
            "detected_locations": detected_locations,
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