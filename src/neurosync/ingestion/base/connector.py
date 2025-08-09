"""
Base connector interface and data structures for data source connectors.

This module defines the core abstractions and data structures used by all
data source connectors in the NeuroSync ingestion system. It provides a
unified interface for extracting data from various sources while maintaining
consistent metadata and error handling across all connector types.

Key Components:
    - BaseConnector: Abstract base class for all data source connectors
    - ConnectorFactory: Factory pattern for connector instantiation
    - IngestionResult: Standardized result structure for ingestion operations
    - SourceMetadata: Comprehensive metadata tracking for ingested content
    - Enums: Type definitions for sources and content formats

Design Principles:
    - Consistent interface across all data source types
    - Comprehensive metadata tracking for provenance and debugging
    - Robust error handling with detailed context information
    - Asynchronous operations for scalable concurrent processing
    - Extensible architecture supporting custom connector implementations

Connector Architecture:
    All connectors follow a common lifecycle:
    1. Configuration validation and initialization
    2. Connection establishment and authentication
    3. Source discovery and enumeration
    4. Data extraction with progress monitoring
    5. Content validation and metadata enrichment
    6. Result packaging and error handling
    7. Resource cleanup and connection management

Supported Source Types:
    - FILE: Local files, directories, and file systems
    - API: REST APIs, GraphQL endpoints, webhooks
    - DATABASE: SQL databases, NoSQL stores, data warehouses
    - CLOUD: Cloud storage services (S3, GCS, Azure Blob)
    - STREAM: Real-time data streams and message queues

Content Type Detection:
    The system automatically detects content types based on:
    - File extensions and MIME types
    - HTTP Content-Type headers
    - Content analysis and magic number detection
    - User-specified type hints in configuration

Error Handling Strategy:
    - Graceful degradation with partial results
    - Detailed error context for debugging
    - Retry mechanisms for transient failures
    - Comprehensive logging for audit trails

Example:
    >>> # Implement custom connector
    >>> class CustomConnector(BaseConnector):
    ...     async def connect(self) -> None:
    ...         # Establish connection to data source
    ...         pass
    ...
    ...     async def ingest(self, source_id: str) -> IngestionResult:
    ...         # Extract data and return structured result
    ...         pass
    >>>
    >>> # Register and use connector
    >>> ConnectorFactory.register("custom", CustomConnector)
    >>> connector = ConnectorFactory.create("custom", config)

For connector development and advanced configuration, see:
    - docs/connector-development.md
    - docs/data-source-integration.md
    - examples/custom-connectors.py
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
    """
    Enumeration of supported data source types.

    This enum defines the categories of data sources that the NeuroSync
    ingestion system can handle. Each source type corresponds to a different
    connector implementation with specialized logic for that data source.

    Attributes:
        FILE: Local or network-accessible file systems including:
            - Local files and directories
            - Network file shares (SMB, NFS)
            - FTP/SFTP servers
            - File archives (ZIP, TAR, etc.)

        API: Web-based APIs and services including:
            - REST APIs with JSON/XML responses
            - GraphQL endpoints
            - SOAP web services
            - Webhook endpoints
            - Real-time APIs with pagination

        DATABASE: Database systems and data warehouses including:
            - SQL databases (PostgreSQL, MySQL, SQLite)
            - NoSQL databases (MongoDB, Cassandra, Redis)
            - Data warehouses (Snowflake, BigQuery, Redshift)
            - In-memory databases and caches

    Usage:
        >>> source_type = SourceType.FILE
        >>> connector = ConnectorFactory.create(source_type, config)
        >>> result = await connector.ingest(source_id)

    Note:
        Additional source types can be added by extending this enum and
        implementing corresponding connector classes.
    """

    FILE = "file"
    API = "api"
    DATABASE = "database"


class ContentType(Enum):
    """
    Enumeration of supported content types for processing.

    This enum defines the various content formats that the NeuroSync system
    can automatically detect, parse, and process. Each content type has
    specialized parsing logic and extraction strategies.

    Text-Based Formats:
        TEXT: Plain text files with various encodings
        MARKDOWN: Markdown documents with formatting preservation
        HTML: Web pages and HTML documents with structure extraction
        JSON: JSON data with schema inference and nested object handling
        CSV: Comma-separated values with header detection
        XML: XML documents with namespace and schema support

    Document Formats:
        PDF: Portable Document Format with text and metadata extraction
        DOCX: Microsoft Word documents with styles and structure

    Binary Formats:
        BINARY: Non-text files requiring specialized processing

    Content Detection:
        The system uses multiple methods for content type detection:
        - File extension analysis (.txt, .json, .pdf, etc.)
        - MIME type detection from HTTP headers
        - Magic number analysis for binary formats
        - Content sampling and heuristic analysis
        - User-specified type hints in configuration

    Processing Capabilities:
        Each content type supports:
        - Text extraction with encoding detection
        - Metadata extraction (title, author, creation date)
        - Structure preservation (headings, lists, tables)
        - Error handling for malformed content
        - Incremental processing for large files

    Usage:
        >>> content_type = ContentType.JSON
        >>> processor = ProcessorFactory.create(content_type)
        >>> extracted_text = processor.extract_text(content)

    Note:
        Custom content types can be added by extending this enum and
        implementing corresponding processor classes.
    """

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
    """
    Comprehensive metadata container for ingested content.

    This dataclass captures all relevant metadata about ingested content to
    support provenance tracking, debugging, caching, and downstream processing.
    It provides a standardized way to track content origins and characteristics
    across all connector types.

    Core Identification:
        source_id: Unique identifier for the content source
        source_type: Category of data source (file, API, database)
        content_type: Format of the content for processing selection

    Location Information:
        file_path: Local or network file path (for file sources)
        url: Web URL or API endpoint (for web sources)

    Size and Timing:
        size_bytes: Content size in bytes for memory management
        created_at: Timestamp when metadata was created
        modified_at: Last modification time of source content
        ingested_at: When the content was actually ingested

    Content Characteristics:
        encoding: Character encoding of text content
        language: Detected or specified language code
        checksum: Content hash for duplicate detection and integrity

    Processing Context:
        tags: User-defined labels for categorization
        custom_metadata: Flexible container for source-specific data

    Relationships:
        parent_id: Reference to parent content (for nested sources)
        version: Content version for change tracking

    Usage:
        >>> metadata = SourceMetadata(
        ...     source_id="doc_123",
        ...     source_type=SourceType.FILE,
        ...     content_type=ContentType.PDF,
        ...     file_path="/data/documents/report.pdf",
        ...     size_bytes=1024576,
        ...     tags=["financial", "quarterly"]
        ... )
        >>> result = IngestionResult(content=text, metadata=metadata)

    Note:
        All optional fields default to None/empty to minimize memory usage
        for simple ingestion scenarios while supporting rich metadata when
        needed for complex workflows.
    """

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
    """
    Standardized result container for ingestion operations.

    This dataclass provides a consistent structure for returning results from
    all connector ingestion operations. It captures both successful extractions
    and error conditions with comprehensive context for debugging and monitoring.

    Success Tracking:
        success: Boolean flag indicating operation outcome
        error: Detailed error message for failed operations

    Content Results:
        source_id: Unique identifier for the ingested source
        content: Extracted text content ready for processing
        metadata: Rich metadata about the source and extraction

    Performance Metrics:
        processing_time_seconds: Total time for ingestion operation
        raw_size_bytes: Original content size before processing
        processed_size_bytes: Final content size after extraction
        chunks_count: Number of chunks if content was segmented

    Quality Indicators:
        extraction_confidence: Confidence score for text extraction
        language_detection_confidence: Confidence in language detection
        content_completeness: Percentage of source content extracted

    Usage Patterns:
        Success Case:
        >>> result = IngestionResult(
        ...     success=True,
        ...     source_id="doc_123",
        ...     content="Extracted text content...",
        ...     metadata=metadata_obj,
        ...     processing_time_seconds=2.5
        ... )

        Error Case:
        >>> result = IngestionResult(
        ...     success=False,
        ...     source_id="doc_456",
        ...     error="Failed to parse PDF: corrupted file",
        ...     processing_time_seconds=0.1
        ... )

    Serialization:
        The to_dict() method provides JSON-serializable representation
        for API responses, logging, and storage operations.

    Note:
        Content field may be None for binary files or when extraction
        fails. Always check success flag before processing content.
    """

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
    """
    Abstract base class defining the interface for all data source connectors.

    This class establishes the contract that all connector implementations must
    follow, ensuring consistent behavior across different data source types.
    It provides common functionality while requiring specific implementations
    for source-specific operations.

    Core Responsibilities:
        - Configuration validation and management
        - Connection lifecycle management (connect/disconnect)
        - Data extraction with error handling
        - Metadata enrichment and result packaging
        - Batch processing capabilities
        - Health monitoring and diagnostics

    Configuration Management:
        All connectors receive configuration through the constructor and must
        validate it using _validate_config(). Configuration typically includes:
        - Connection parameters (URLs, credentials, timeouts)
        - Processing options (formats, filters, limits)
        - Output settings (encoding, compression, chunking)

    Connection Lifecycle:
        1. connect(): Establish connection to data source
        2. test_connection(): Verify connection health
        3. Perform ingestion operations
        4. disconnect(): Clean up resources

    Error Handling Strategy:
        - Validate configuration before any operations
        - Test connections before data operations
        - Handle transient failures with retries
        - Return detailed error context in results
        - Log all significant events for debugging

    Batch Processing:
        Connectors support both single and batch ingestion:
        - ingest(): Process single source with full control
        - ingest_batch(): Process multiple sources efficiently
        - Automatic parallelization where safe
        - Progress monitoring for long operations

    Implementation Requirements:
        Subclasses must implement all abstract methods:
        - _validate_config(): Check configuration validity
        - connect()/disconnect(): Manage connections
        - test_connection(): Health checks
        - list_sources(): Discovery operations
        - ingest(): Single source processing
        - ingest_batch(): Multiple source processing

    Usage Pattern:
        >>> connector = FileConnector(config)
        >>> await connector.connect()
        >>> sources = await connector.list_sources()
        >>> result = await connector.ingest(sources[0])
        >>> await connector.disconnect()

    Threading and Async:
        All operations are async to support concurrent processing.
        Connectors should be thread-safe for batch operations but
        not necessarily for multiple concurrent connections.

    Example Implementation:
        >>> class CustomConnector(BaseConnector):
        ...     def _validate_config(self):
        ...         if 'endpoint' not in self.config:
        ...             raise ConfigurationError("endpoint required")
        ...
        ...     async def connect(self):
        ...         self.client = Client(self.config['endpoint'])
        ...         await self.client.authenticate()
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.logger = get_logger(f"{__name__}.{self.name}")
        # Note: _validate_config() should be called by child classes

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate connector-specific configuration parameters.

        This method must be implemented by each connector to verify that
        all required configuration parameters are present and valid for
        the specific data source type. Should raise ConfigurationError
        for any invalid or missing configuration.

        Common validations include:
        - Required parameters presence
        - Parameter format and type checking
        - Credential format validation
        - URL and endpoint syntax checking
        - File path existence verification
        - Numeric range validation

        Raises:
            ConfigurationError: When configuration is invalid or incomplete
        """
        pass

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the data source.

        This method handles the initial connection setup including:
        - Authentication and credential validation
        - Network connection establishment
        - Resource allocation and initialization
        - Connection pooling setup if applicable
        - SSL/TLS handshake for secure connections

        Should be idempotent - safe to call multiple times.
        Must handle connection failures gracefully and provide
        meaningful error messages.

        Raises:
            ConnectionError: When connection cannot be established
            AuthenticationError: When credentials are invalid
            TimeoutError: When connection times out
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection and clean up resources.

        This method performs graceful shutdown including:
        - Closing active connections and streams
        - Releasing allocated resources
        - Clearing cached data and temporary files
        - Logging connection statistics
        - Finalizing any pending operations

        Should be safe to call even if not connected.
        Must not raise exceptions during cleanup.
        """
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test if connection to data source is working.

        This method performs a lightweight connectivity check without
        performing full data operations. Useful for health checks and
        configuration validation.

        Returns:
            bool: True if connection is healthy, False otherwise

        Note:
            Should not raise exceptions - return False for any failure.
            This method may establish a temporary connection if needed.
        """
        pass

    @abstractmethod
    async def list_sources(self) -> List[str]:
        """
        List available sources, files, or endpoints.

        This method discovers and returns all available data sources
        that can be ingested through this connector. The exact meaning
        depends on the connector type:

        File Connector: Returns file paths in configured directories
        API Connector: Returns available endpoints or resource identifiers
        Database Connector: Returns table/collection names

        Returns:
            List[str]: List of source identifiers that can be passed
                      to ingest() method

        Raises:
            ConnectionError: If not connected to data source
            PermissionError: If lacking permissions to list sources
        """
        pass

    @abstractmethod
    async def ingest(self, source_id: str, **kwargs) -> IngestionResult:
        """
        Ingest data from a specific source.

        This is the core method that extracts data from a single source
        and returns it in standardized format. Implementation should:

        1. Validate source_id exists and is accessible
        2. Extract raw content from the source
        3. Detect content type and encoding
        4. Generate comprehensive metadata
        5. Handle extraction errors gracefully
        6. Return IngestionResult with all context

        Args:
            source_id: Identifier for the specific source to ingest
            **kwargs: Connector-specific parameters for extraction

        Returns:
            IngestionResult: Structured result with content and metadata

        Common kwargs:
            max_size: Maximum content size to extract
            encoding: Force specific text encoding
            timeout: Operation timeout in seconds
            filters: Content filtering criteria
        """
        pass

    @abstractmethod
    async def ingest_batch(
        self, source_ids: List[str], **kwargs
    ) -> List[IngestionResult]:
        """
        Ingest data from multiple sources efficiently.

        This method processes multiple sources in an optimized manner,
        potentially using parallel processing, connection pooling, and
        batch operations to improve performance over sequential single
        ingestion calls.

        Implementation strategies:
        - Parallel processing for independent sources
        - Connection reuse across multiple sources
        - Batch API calls where supported by source
        - Progress monitoring for long operations
        - Graceful error handling per source

        Args:
            source_ids: List of source identifiers to process
            **kwargs: Common parameters applied to all sources

        Returns:
            List[IngestionResult]: Results for each source in same order
                                  as input list, with individual success/error
                                  status per source

        Batch-specific kwargs:
            max_concurrent: Maximum parallel operations
            fail_fast: Stop on first error vs continue processing
            progress_callback: Function to call with progress updates

        Note:
            Results list always matches input source_ids length.
            Failed ingestions return IngestionResult with success=False.
        """
        pass

    async def get_source_info(self, source_id: str) -> Dict[str, Any]:
        """
        Get metadata about a specific source without full ingestion.

        This method provides lightweight metadata discovery for a source
        without performing full content extraction. Useful for source
        validation, preview operations, and inventory management.

        Args:
            source_id: Identifier for the source to inspect

        Returns:
            Dict[str, Any]: Source metadata including:
                - source_id: Original identifier
                - connector: Connector class name
                - last_checked: Timestamp of this check
                - exists: Whether source is accessible
                - size: Content size if available
                - modified: Last modification time
                - content_type: Detected content type
                - permissions: Access level information
        """
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
        """
        Create standardized metadata object for a source.

        This utility method constructs a SourceMetadata object with
        consistent defaults while allowing connector-specific customization
        through keyword arguments.

        Args:
            source_id: Unique identifier for the source
            source_type: Category of data source
            content_type: Format of the content
            **kwargs: Additional metadata fields to set

        Returns:
            SourceMetadata: Initialized metadata object

        Common kwargs:
            file_path: Local file path for file sources
            url: Web URL for API sources
            size_bytes: Content size in bytes
            encoding: Character encoding
            title: Human-readable title
            tags: List of categorization tags
            custom_metadata: Connector-specific data
        """
        return SourceMetadata(
            source_id=source_id,
            source_type=source_type,
            content_type=content_type,
            **kwargs,
        )

    def _detect_content_type(self, file_path: str) -> ContentType:
        """
        Detect content type from file extension.

        This utility method provides basic content type detection based on
        file extensions. More sophisticated detection can be implemented
        in specific connectors using content analysis, MIME types, or
        magic number detection.

        Args:
            file_path: Path to file for extension analysis

        Returns:
            ContentType: Detected content type, defaults to TEXT

        Supported Extensions:
            .txt, .log, .cfg -> TEXT
            .md, .markdown -> MARKDOWN
            .json -> JSON
            .csv, .tsv -> CSV
            .xml, .xsd -> XML
            .pdf -> PDF
            .docx, .doc -> DOCX
            .html, .htm -> HTML

        Note:
            This is a fallback detection method. Connectors should
            implement more sophisticated detection when possible.
        """
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
        """
        Async context manager entry point.

        Automatically establishes connection when entering context.
        Enables usage pattern: async with connector: ...

        Returns:
            BaseConnector: Self reference for context usage
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit point.

        Automatically disconnects and cleans up resources when
        exiting context, regardless of whether exceptions occurred.

        Args:
            exc_type: Exception type if any
            exc_val: Exception value if any
            exc_tb: Exception traceback if any

        Note:
            Always performs cleanup, never suppresses exceptions.
        """
        await self.disconnect()


class ConnectorFactory:
    """
    Factory class for creating and managing connector instances.

    This factory implements the Factory design pattern to provide a centralized
    way to create connector instances based on configuration. It supports
    dynamic registration of connector types and provides discovery capabilities.

    Registry Management:
        The factory maintains a registry of available connector classes,
        allowing dynamic registration of new connector types without
        modifying core code. This enables plugin-style architecture.

    Connector Creation:
        Creates connector instances with proper configuration validation
        and error handling. Ensures consistent initialization across
        all connector types.

    Type Discovery:
        Provides introspection capabilities to list available connector
        types for configuration UIs and documentation generation.

    Usage Patterns:
        Registration (typically at module import):
        >>> ConnectorFactory.register("file", FileConnector)
        >>> ConnectorFactory.register("api", APIConnector)

        Creation:
        >>> config = {"path": "/data", "pattern": "*.txt"}
        >>> connector = ConnectorFactory.create("file", config)

        Discovery:
        >>> available = ConnectorFactory.list_connectors()
        >>> print(f"Available connectors: {available}")

    Error Handling:
        - Validates connector type exists before creation
        - Ensures configuration is properly passed to connector
        - Provides clear error messages for unknown types
        - Handles connector-specific configuration validation

    Thread Safety:
        The registry is thread-safe for read operations but registration
        should be done during module initialization before concurrent access.
    """

    _connectors: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, connector_class: type):
        """
        Register a connector class with a given name.

        Args:
            name: Unique identifier for the connector type
            connector_class: Connector class extending BaseConnector

        Note:
            Registration should typically be done at module import time.
            Duplicate names will overwrite previous registrations.
        """
        cls._connectors[name] = connector_class

    @classmethod
    def create(cls, connector_type: str, config: Dict[str, Any]) -> BaseConnector:
        """
        Create a connector instance of the specified type.

        Args:
            connector_type: Name of registered connector type
            config: Configuration dictionary for the connector

        Returns:
            BaseConnector: Initialized connector instance

        Raises:
            ConfigurationError: If connector type is unknown
        """
        if connector_type not in cls._connectors:
            raise ConfigurationError(f"Unknown connector type: {connector_type}")

        connector_class = cls._connectors[connector_type]
        return connector_class(config)

    @classmethod
    def list_connectors(cls) -> List[str]:
        """
        List all registered connector types.

        Returns:
            List[str]: Names of all registered connector types
        """
        return list(cls._connectors.keys())
