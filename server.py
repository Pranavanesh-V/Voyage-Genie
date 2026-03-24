import os
import shutil
import uuid
import logging
import json
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

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

# ---------------- JOB STORAGE ---------------- #
JOB_RESULTS_DIR = Path("/tmp/jobs")
JOB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

# ------------------ UTILITIES ------------------ #
def cleanup_temp_files(session_id: str):
    for folder in [VIDEO_DIR, AUDIO_DIR, FRAME_DIR]:
        path = folder / session_id
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                logger.info(f"Cleaned temp folder: {path}")
            except Exception as e:
                logger.error(f"Error cleaning folder {path}: {e}")

def format_place_name(place):
    parts = [place.get("name", "")]
    if place.get("city"):
        parts.append(place["city"])
    if place.get("country"):
        parts.append(place["country"])
    return ", ".join(parts)

def save_job_result(session_id, result):
    job_file = JOB_RESULTS_DIR / f"{session_id}.json"
    with open(job_file, "w") as f:
        json.dump(result, f)

def load_job_result(session_id):
    job_file = JOB_RESULTS_DIR / f"{session_id}.json"
    if job_file.exists():
        with open(job_file, "r") as f:
            return json.load(f)
    return None

# ------------------ BACKGROUND PROCESS ------------------ #
def process_reel_background(reel_url: str, session_id: str):
    try:
        logger.info(f"[{session_id}] Starting reel processing")

        # 1. Download video
        video_path = download_reel(reel_url, session_id)
        logger.info(f"[{session_id}] Video downloaded: {video_path}")

        # 2. Metadata
        metadata = extract_metadata(reel_url)
        caption = metadata.get("caption", "")
        hashtags = metadata.get("hashtags", [])

        # 3. Audio + transcription
        audio_path = get_audio_from_video(video_path, session_id)
        transcription = transcribe_audio(audio_path)

        # 4. Frames + OCR + Vision
        frames_path = extract_frames(video_path, session_id)
        ocr_text = extract_text_from_frames(frames_path)
        vision_results = detect_scenes(frames_path)

        # 5. Combine all text sources
        combined_text = f"{caption} {' '.join(hashtags)} {transcription} {ocr_text} {' '.join(vision_results)}"
        cleaned_text = clean_text(combined_text)

        # 6. Detect locations
        location_data = find_locations_in_text(cleaned_text)
        detected_locations = location_data["places"]
        location_context = location_data["context"]

        # 7. Validate locations
        raw_places = validate_locations(detected_locations, location_context)
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

        # 8. Save job result
        save_job_result(session_id, {
            "status": "completed",
            "caption": caption,
            "validated_places": validated_places
        })

        # 9. Cleanup temp files
        cleanup_temp_files(session_id)
        logger.info(f"[{session_id}] Processing completed successfully")

    except Exception as e:
        logger.error(f"[{session_id}] Error: {e}")
        save_job_result(session_id, {"status": "failed", "error": str(e)})
        cleanup_temp_files(session_id)

# ------------------ API ENDPOINTS ------------------ #
@app.post("/process-reel")
async def start_reel_processing(request: ReelRequest, background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())
    background_tasks.add_task(process_reel_background, request.reelUrl, session_id)
    return {"job_id": session_id, "status": "processing"}

@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    result = load_job_result(job_id)
    if not result:
        return {"status": "processing"}
    return result

@app.get("/health")
async def health_check():
    return {"status": "ready"}

@app.get("/")
def home():
    return {"message": "VoyageGenie AI backend running"}

# ------------------ RUN ------------------ #
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render uses $PORT
    uvicorn.run(app, host="0.0.0.0", port=port)