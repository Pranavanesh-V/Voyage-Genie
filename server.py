import os
import shutil
import uuid
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from config import VIDEO_DIR, AUDIO_DIR, FRAME_DIR
from services.reel_downloader import download_reel
from services.metadata_extractor import extract_metadata
from services.frame_extractor import extract_frames, get_audio_from_video
from services.vision_detector import detect_scenes
from services.ocr_engine import extract_text_from_frames
from services.speech_recognition import transcribe_audio
from services.location_detector import find_locations_in_text
from services.geo_validator import validate_locations
from utils.text_cleaner import clean_text

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


# ------------------ CLEANUP ------------------ #
def cleanup_temp_files(session_id: str):
    for folder in [VIDEO_DIR, AUDIO_DIR, FRAME_DIR]:
        path = folder / session_id
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                logger.info(f"Cleaned up temp folder: {path}")
            except Exception as e:
                logger.error(f"Failed to clean up folder {path}: {e}")


def format_place_name(place):
    name = place.get("name", "")
    city = place.get("city", "")
    country = place.get("country", "")

    parts = [name]
    if city:
        parts.append(city)
    if country:
        parts.append(country)

    return ", ".join(parts)


# ------------------ API ------------------ #
@app.post("/process-reel", response_model=ReelResponse)
async def process_reel(request: ReelRequest, background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())

    try:
        # 1. Download Reel
        video_path = await download_reel(request.reelUrl, session_id)
        logger.info(f"Video downloaded: {video_path}")

        # 2. Metadata
        metadata = extract_metadata(request.reelUrl)
        caption = metadata.get("caption", "")
        hashtags = metadata.get("hashtags", [])

        # 3. Audio extraction & transcription
        audio_path = get_audio_from_video(video_path, session_id)
        transcription = transcribe_audio(audio_path)

        # 4. Frame extraction + OCR + Vision detection
        frames_path = extract_frames(video_path, session_id)
        ocr_text = extract_text_from_frames(frames_path)
        vision_results = detect_scenes(frames_path)

        # 5. Combine all text sources
        combined_text = f"{caption} {' '.join(hashtags)} {transcription} {ocr_text} {' '.join(vision_results)}"
        cleaned_text = clean_text(combined_text)
        logger.info(f"Cleaned text length: {len(cleaned_text)}")

        # 6. Detect locations
        location_data = find_locations_in_text(cleaned_text)
        detected_locations = location_data["places"]
        location_context = location_data["context"]

        # 7. Validate locations
        raw_places = await validate_locations(detected_locations, location_context)
        validated_places = [
            {
                "name": format_place_name(place),
                "city": place.get("city", ""),
                "state": place.get("state", ""),
                "country": place.get("country", ""),
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "confidence": place["confidence"]
            }
            for place in raw_places
        ]

        # 8. Cleanup in background
        background_tasks.add_task(cleanup_temp_files, session_id)

        return {
            "caption": caption,
            "hashtags": hashtags,
            "detected_locations": list(set(detected_locations)),
            "validated_places": validated_places
        }

    except Exception as e:
        cleanup_temp_files(session_id)
        logger.error(f"Error processing reel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------ HEALTH CHECK ------------------ #
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
def home():
    return {"message": "VoyageGenie AI backend running"}