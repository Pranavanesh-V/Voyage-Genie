import os
import logging

logger = logging.getLogger(__name__)

_model = None

def load_whisper_model():
    global _model
    if _model is None:
        import whisper
        logger.info("Loading Whisper model...")
        _model = whisper.load_model("tiny")  # Use tiny or small for CPU
        logger.info("Whisper model loaded")
    return _model

def transcribe_audio(audio_path: str) -> str:
    if not audio_path or not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}")
        return ""
    try:
        model = load_whisper_model()
        result = model.transcribe(audio_path)
        text = result.get("text", "").strip()
        logger.info(f"Transcription completed for {audio_path}")
        return text
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return ""