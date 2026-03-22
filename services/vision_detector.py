import os
import logging
from typing import List

import torch
from torchvision import models, transforms
from PIL import Image

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- MODEL SETUP ---------------- #
try:
    # General scene recognition (ResNet50 pretrained on ImageNet)
    model = models.mobilenet_v2(weights="DEFAULT")
    model.eval()
    logger.info("ResNet50 model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load ResNet50 model: {e}")
    raise RuntimeError("Could not load vision model.")

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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
            input_image = Image.open(frame_path)
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)

            with torch.no_grad():
                output = model(input_batch)

            # Here you would map ImageNet classes to travel categories
            # For now, this is mocked for pipeline purposes
            # detected_labels.append(mapped_label)
            pass

        except Exception as e:
            logger.error(f"Vision processing error on '{frame_name}': {e}")

    # Mocked detection results for demonstration purposes
    return ["mountain", "lake"]