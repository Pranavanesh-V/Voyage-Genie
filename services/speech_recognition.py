import os
import logging
from typing import Optional

import whisper

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- LOAD WHISPER MODEL ---------------- #
# Load once for the entire module (base model balances speed & accuracy)
try:
    model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise RuntimeError("Whisper model could not be loaded. Run `pip install openai-whisper` and ensure model download works.")


# ---------------- AUDIO TRANSCRIPTION ---------------- #
def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes audio to text using OpenAI Whisper.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Transcribed text.
    """
    if not audio_path or not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}")
        return ""

    try:
        result = model.transcribe(audio_path)
        text = result.get("text", "").strip()
        logger.info(f"Audio transcription completed for file: {audio_path}")
        return text

    except Exception as e:
        logger.error(f"Whisper transcription error for file '{audio_path}': {e}")
        return ""