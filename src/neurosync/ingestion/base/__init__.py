"""
Base ingestion components
"""

from .connector import (
    BaseConnector,
    ConnectorFactory,
    ContentType,
    IngestionResult,
    SourceMetadata,
    SourceType,
)

__all__ = [
    "BaseConnector",
    "ConnectorFactory",
    "SourceType",
    "ContentType",
    "SourceMetadata",
    "IngestionResult",
]
