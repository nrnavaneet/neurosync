"""
NeuroSync Ingestion Module - Multi-source data ingestion engine
"""

from .base import (
    BaseConnector,
    ConnectorFactory,
    ContentType,
    IngestionResult,
    SourceMetadata,
    SourceType,
)
from .manager import IngestionManager

__all__ = [
    "IngestionManager",
    "BaseConnector",
    "ConnectorFactory",
    "SourceType",
    "ContentType",
    "SourceMetadata",
    "IngestionResult",
]
