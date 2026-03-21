import re
import logging
from typing import Dict

import yt_dlp

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- METADATA EXTRACTION ---------------- #
def extract_metadata(url: str) -> Dict[str, object]:
    """
    Extract caption, hashtags, and location from a reel/video URL.
    Uses yt-dlp to fetch video description.
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'dump_single_json': True,
        'skip_download': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            description = info.get('description', '')

            # Extract hashtags
            hashtags = re.findall(r"#(\w+)", description)

            # Extract location if available
            location = info.get('location', '')

            logger.info(f"Metadata extracted: hashtags={hashtags}, location={location}")

            return {
                "caption": description,
                "hashtags": hashtags,
                "location": location
            }

    except Exception as e:
        logger.error(f"Metadata extraction error for URL '{url}': {e}")
        return {"caption": "", "hashtags": [], "location": ""}


