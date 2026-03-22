import os
import logging
from typing import List
from PIL import Image

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Vision model is optional and currently function returns mock results
# to keep 512MB Render free tier safe. If needed, one can re-enable
# a lightweight model load here with explicit CPU-only settings.


# ---------------- SCENE DETECTION ---------------- #
def detect_scenes(frames_dir: str) -> List[str]:
    """
    Analyzes frames to detect scenes like beach, mountain, monument, etc.
    
    Args:
        frames_dir (str): Directory containing image frames.

    Returns:
        List[str]: Detected scene labels.
    """
    if not frames_dir or not os.path.exists(frames_dir):
        logger.warning(f"Frames directory not found: {frames_dir}")
        return []

    detected_labels: List[str] = []

    for frame_name in os.listdir(frames_dir):
        frame_path = os.path.join(frames_dir, frame_name)

        try:
            # Keep simple for low-memory environments: no model inference.
            # You may replace with a lightweight pre-trained model if needed.
            _ = Image.open(frame_path)

        except Exception as e:
            logger.error(f"Vision processing error on '{frame_name}': {e}")

    # Mocked detection results for demonstration purposes
    return ["mountain", "lake"]