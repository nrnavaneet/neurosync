"""
NeuroSync - AI-Native ETL Pipeline for RAG and LLM Applications

NeuroSync is a comprehensive, production-ready framework for building AI-powered
data processing pipelines specifically designed for Retrieval-Augmented Generation
(RAG) and Large Language Model (LLM) applications.

Key Features:
    - Intelligent data ingestion from multiple sources (files, APIs, databases)
    - Advanced text processing and chunking strategies
    - Flexible embedding generation with multiple model backends
    - Optimized vector storage with FAISS and Qdrant support
    - Integrated LLM serving with multi-provider support
    - Real-time chat interface with context-aware responses
    - Production-ready orchestration with Apache Airflow
    - Comprehensive monitoring and logging

Modules:
    core: Core configuration, logging, and utility components
    ingestion: Data ingestion from various sources
    processing: Text processing, cleaning, and chunking
    pipelines: End-to-end pipeline orchestration
    serving: LLM serving and chat interface
    storage: Vector storage and retrieval
    cli: Command-line interface tools

Example:
    >>> from neurosync import Settings, get_logger
    >>> settings = Settings()
    >>> logger = get_logger(__name__)
    >>> logger.info("NeuroSync initialized")

For detailed usage examples and API documentation, see:
    - README.md: Quick start guide
    - docs/: Comprehensive documentation
    - examples/: Usage examples and tutorials
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
