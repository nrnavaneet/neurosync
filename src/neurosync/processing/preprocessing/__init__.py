"""
NeuroSync Text Preprocessing Components

This module provides text preprocessing capabilities for cleaning and normalizing
content before chunking and embedding generation.

Available Preprocessors:
    - HTMLCleaner: Removes HTML tags and artifacts
    - WhitespaceNormalizer: Normalizes whitespace and removes extra spaces

Available Functions:
    - detect_language: Detects the language of text content

Author: NeuroSync Team
Created: 2025
"""

from .cleaners import HTMLCleaner, WhitespaceNormalizer
from .language_detector import detect_language

__all__ = [
    "HTMLCleaner",
    "WhitespaceNormalizer",
    "detect_language",
]
