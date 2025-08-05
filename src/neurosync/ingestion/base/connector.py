"""
Base connector interface for all data source connectors
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from neurosync.core.exceptions.custom_exceptions import ConfigurationError
from neurosync.core.logging.logger import get_logger

logger = get_logger(__name__)


class SourceType(Enum):
    """Supported source types"""

    FILE = "file"
    API = "api"
    DATABASE = "database"


class ContentType(Enum):
    """Content types for processing"""

    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    BINARY = "binary"


@dataclass
class SourceMetadata:
    """Metadata for ingested content"""

    source_id: str
    source_type: SourceType
    content_type: ContentType
    file_path: Optional[str] = None
    url: Optional[str] = None
    size_bytes: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: Optional[datetime] = None
    checksum: Optional[str] = None
    encoding: str = "utf-8"
    language: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate checksum if content is provided"""
        if not self.checksum and hasattr(self, "_content"):
            self.checksum = self._generate_checksum(getattr(self, "_content", ""))

    @staticmethod
    def _generate_checksum(content: Union[str, bytes]) -> str:
        """Generate SHA 256 checksum for content"""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "content_type": self.content_type.value,
            "file_path": self.file_path,
            "url": self.url,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "checksum": self.checksum,
            "encoding": self.encoding,
            "language": self.language,
            "title": self.title,
            "author": self.author,
            "description": self.description,
            "tags": self.tags,
            "custom_metadata": self.custom_metadata,
        }


@dataclass
class IngestionResult:
    """Result of ingestion operation"""

    success: bool
    source_id: str
    content: Optional[str] = None
    metadata: Optional[SourceMetadata] = None
    error: Optional[str] = None
    processing_time_seconds: float = 0.0
    chunks_count: int = 0
    raw_size_bytes: int = 0
    processed_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "source_id": self.source_id,
            "content": self.content,
            "content_length": len(self.content) if self.content else 0,
            "error": self.error,
            "processing_time_seconds": self.processing_time_seconds,
            "chunks_count": self.chunks_count,
            "raw_size_bytes": self.raw_size_bytes,
            "processed_size_bytes": self.processed_size_bytes,
            "metadata": self.metadata.__dict__ if self.metadata else None,
        }


class BaseConnector(ABC):
    """Abstract base class for all data connectors"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.logger = get_logger(f"{__name__}.{self.name}")
        # Note: _validate_config() should be called by child classes

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate connector-specific configuration"""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source"""
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if connection is working"""
        pass

    @abstractmethod
    async def list_sources(self) -> List[str]:
        """List available sources/files/endpoints"""
        pass

    @abstractmethod
    async def ingest(self, source_id: str, **kwargs) -> IngestionResult:
        """Ingest data from a specific source"""
        pass

    @abstractmethod
    async def ingest_batch(
        self, source_ids: List[str], **kwargs
    ) -> List[IngestionResult]:
        """Ingest data from multiple sources"""
        pass

    async def get_source_info(self, source_id: str) -> Dict[str, Any]:
        """Get metadata about a specific source"""
        return {
            "source_id": source_id,
            "connector": self.name,
            "last_checked": datetime.now(timezone.utc).isoformat(),
        }

    def _create_source_metadata(
        self,
        source_id: str,
        source_type: SourceType,
        content_type: ContentType,
        **kwargs,
    ) -> SourceMetadata:
        """Create metadata object for a source"""
        return SourceMetadata(
            source_id=source_id,
            source_type=source_type,
            content_type=content_type,
            **kwargs,
        )

    def _detect_content_type(self, file_path: str) -> ContentType:
        """Detect content type from file extension"""
        suffix = Path(file_path).suffix.lower()
        mapping = {
            ".txt": ContentType.TEXT,
            ".md": ContentType.MARKDOWN,
            ".json": ContentType.JSON,
            ".csv": ContentType.CSV,
            ".xml": ContentType.XML,
            ".pdf": ContentType.PDF,
            ".docx": ContentType.DOCX,
            ".html": ContentType.HTML,
            ".htm": ContentType.HTML,
        }
        return mapping.get(suffix, ContentType.TEXT)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()


class ConnectorFactory:
    """Factory for creating connector instances"""

    _connectors: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, connector_class: type):
        """Register a connector class"""
        cls._connectors[name] = connector_class

    @classmethod
    def create(cls, connector_type: str, config: Dict[str, Any]) -> BaseConnector:
        """Create a connector instance"""
        if connector_type not in cls._connectors:
            raise ConfigurationError(f"Unknown connector type: {connector_type}")

        connector_class = cls._connectors[connector_type]
        return connector_class(config)

    @classmethod
    def list_connectors(cls) -> List[str]:
        """List available connector types"""
        return list(cls._connectors.keys())
