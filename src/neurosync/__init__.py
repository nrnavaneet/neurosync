"""
NeuroSync - AI-Native ETL Pipeline for RAG and LLM Applications
"""

__version__ = "0.1.0"
__author__ = "NeuroSync"
__email__ = "navaneetnr@gmail.com"
__description__ = (
    "AI-native ETL pipeline designed for RAG (Retrieval-Augmented Generation) "
    "and LLM (Large Language Model) applications. It provides a robust "
    "framework for data extraction, transformation, and loading, enabling "
    "efficient integration with AI models."
)

from neurosync.core.config.settings import Settings
from neurosync.core.logging.logger import get_logger

__all__ = [
    "Settings",
    "get_logger",
]
