import os
import logging
from pathlib import Path

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- BASE DIRECTORIES ---------------- #
BASE_DIR = Path(__file__).resolve().parent

# Temporary Storage for processing
TEMP_DIR = BASE_DIR / "temp"
VIDEO_DIR = TEMP_DIR / "videos"
AUDIO_DIR = TEMP_DIR / "audio"
FRAME_DIR = TEMP_DIR / "frames"

# Create all required directories
for folder in [VIDEO_DIR, AUDIO_DIR, FRAME_DIR]:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {folder}")
    except Exception as e:
        logger.error(f"Failed to create directory '{folder}': {e}")

# ---------------- API KEYS & CONFIG ---------------- #
# Replace with actual keys or use environment variables
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "VoyageGenie/1.0")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------- PROCESSING CONSTANTS ---------------- #
FRAME_EXTRACT_RATE = 2       # Extract 1 frame every 2 seconds
MAX_VIDEO_DURATION = 60      # Limit processing to 60 seconds
CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence for validated places

logger.info("Configuration initialized successfully.")