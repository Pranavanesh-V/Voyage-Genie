import re
import logging

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- TEXT CLEANING ---------------- #
def clean_text(text: str) -> str:
    """
    Cleans extracted text by removing noise, extra whitespace,
    and non-alphanumeric characters (except spaces and common punctuation).

    Args:
        text (str): Raw text to clean.

    Returns:
        str: Cleaned text.
    """
    if not text:
        logger.warning("Received empty text for cleaning.")
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Replace newlines and tabs with space
    text = text.replace('\n', ' ').replace('\t', ' ')

    # Remove special characters except alphanumeric, spaces, commas, periods, hyphens
    text = re.sub(r'[^\w\s,.-]', ' ', text)

    # Remove multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    logger.info(f"Cleaned text length: {len(cleaned_text)} characters")
    return cleaned_text