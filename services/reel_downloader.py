import os
import logging
from pathlib import Path

import yt_dlp
from config import VIDEO_DIR  # VIDEO_DIR should be a Path object

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- DOWNLOAD REEL ---------------- #
async def download_reel(url: str, session_id: str) -> str:
    """
    Download Instagram reel/video and return the exact file path.

    Args:
        url (str): URL of the reel/video.
        session_id (str): Unique session identifier to organize downloads.

    Returns:
        str: Path to the downloaded video file.

    Raises:
        Exception: If download fails or file not found.
    """
    session_path: Path = VIDEO_DIR / session_id
    os.makedirs(session_path, exist_ok=True)

    ydl_opts = {
        "format": "best",
        "outtmpl": str(session_path / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)

        if os.path.exists(file_path):
            logger.info(f"Video downloaded successfully: {file_path}")
            return file_path

        raise Exception("Video download failed: File not found")

    except Exception as e:
        logger.error(f"Failed to download reel from '{url}': {e}")
        raise Exception(f"Failed to download reel: {str(e)}")