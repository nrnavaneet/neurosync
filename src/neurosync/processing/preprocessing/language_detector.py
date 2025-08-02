"""
Language detection module.
"""
from typing import Optional

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

from neurosync.core.logging.logger import get_logger

logger = get_logger(__name__)

# Ensure consistent results
DetectorFactory.seed = 0


def detect_language(text: str) -> Optional[str]:
    """
    Detects the language of a given text.

    Args:
        text: The text to analyze.

    Returns:
        A string representing the language code (e.g., 'en', 'fr'), or None if
        detection fails.
    """
    try:
        # Take a sample of the text for efficiency
        sample = text[:500]
        return detect(sample)
    except LangDetectException:
        logger.warning("Language detection failed for the given text.")
        return None
