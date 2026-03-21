import pytesseract
from PIL import Image
import os
import cv2
import shutil

# Auto-detect Tesseract path (works for Render & local)
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "tesseract"


def extract_text_from_frames(frames_dir: str) -> str:
    """
    Runs OCR on all frames in a directory to detect text.
    Returns a combined string of all detected text.
    """

    all_text = []

    if not os.path.exists(frames_dir):
        print("Frames directory not found")
        return ""

    # Loop through frames
    for frame_name in sorted(os.listdir(frames_dir)):

        # Process only images
        if not frame_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        frame_path = os.path.join(frames_dir, frame_name)

        try:
            # Read image using OpenCV
            img = cv2.imread(frame_path)

            if img is None:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Convert to PIL Image
            pil_img = Image.fromarray(gray)

            # OCR config (good for scattered text)
            config = "--psm 11"

            text = pytesseract.image_to_string(pil_img, config=config)

            if text.strip():
                all_text.append(text.strip())

        except Exception as e:
            print(f"OCR Error on {frame_name}: {e}")

    # Remove duplicates
    unique_text = list(set(all_text))

    return " ".join(unique_text)