"""
Custom exception hierarchy for comprehensive NeuroSync error handling.

This module defines a structured exception hierarchy that provides detailed
error information, context, and recovery guidance throughout the NeuroSync
application. Each exception includes error codes, detailed messages, and
contextual metadata to facilitate debugging and error recovery.

Exception Hierarchy:
    NeuroSyncError (base)
    ├── ConfigurationError: Configuration and setup issues
    ├── ConnectionError: External service connectivity problems
    ├── IngestionError: Data ingestion and extraction failures
    ├── ProcessingError: Text processing and transformation errors
    ├── StorageError: Database and storage operation failures
    ├── EmbeddingError: Vector embedding generation issues
    ├── ValidationError: Data validation and schema errors
    └── AuthenticationError: API key and authentication failures

Key Features:
    - Structured error information with codes and context
    - Detailed error messages with actionable guidance
    - Contextual metadata for debugging and recovery
    - Integration with logging and monitoring systems
    - Support for error chaining and root cause analysis
    - Localization support for error messages

Error Context:
    Each exception can include:
    - Human-readable error message
    - Machine-readable error code
    - Contextual details dictionary
    - Suggested recovery actions
    - Related configuration or data

Best Practices:
    - Use specific exception types for different failure modes
    - Include relevant context in the details dictionary
    - Provide actionable error messages for users
    - Chain exceptions to preserve root cause information
    - Log exceptions with appropriate severity levels

Example:
    >>> try:
    ...     process_data(config)
    ... except ConfigurationError as e:
    ...     logger.error("Configuration issue",
    ...                  error_code=e.error_code,
    ...                  details=e.details)
    ...     print(f"Error: {e.message}")
    ...     # Handle configuration recovery
    >>>
    >>> # Raising with context
    >>> raise IngestionError(
    ...     "Failed to process file",
    ...     error_code="FILE_PROCESSING_ERROR",
    ...     details={"file_path": "/path/to/file", "file_size": 1024}
    ... )

For error handling patterns and recovery strategies, see:
    - docs/error-handling.md
    - docs/troubleshooting.md
    - examples/error-recovery.py
"""

from typing import Any, Dict, Optional


class NeuroSyncError(Exception):
    """
    Base exception class for all NeuroSync application errors.

    Provides a structured foundation for error handling throughout the
    application with consistent error formatting, contextual information,
    and debugging support. All NeuroSync-specific exceptions should
    inherit from this base class.

    Attributes:
        message (str): Human-readable error description
        error_code (str): Machine-readable error identifier
        details (Dict[str, Any]): Additional contextual information

    The error_code follows a hierarchical naming convention:
        - MODULE_OPERATION_ERROR (e.g., "INGESTION_FILE_READ_ERROR")
        - Defaults to class name if not specified
        - Used for error categorization and monitoring

    The details dictionary can include:
        - Configuration values that caused the error
        - File paths, URLs, or other resource identifiers
        - State information for debugging
        - Suggested recovery actions
        - Related error codes or references

    Example:
        >>> raise NeuroSyncError(
        ...     "Database connection failed",
        ...     error_code="DB_CONNECTION_ERROR",
        ...     details={
        ...         "host": "localhost",
        ...         "port": 5432,
        ...         "timeout": 30,
        ...         "retry_count": 3
        ...     }
        ... )
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class ConfigurationError(NeuroSyncError):
    """
    Raised when configuration validation or setup fails.

    Indicates issues with application configuration, including:
        - Invalid configuration file format or content
        - Missing required configuration parameters
        - Conflicting configuration values
        - Environment variable setup problems
        - Template or schema validation failures

    Common scenarios:
        - YAML/JSON parsing errors in config files
        - Missing API keys or credentials
        - Invalid file paths or permissions
        - Incompatible configuration combinations
        - Schema validation failures

    Example:
        >>> raise ConfigurationError(
        ...     "Missing required API key configuration",
        ...     error_code="CONFIG_MISSING_API_KEY",
        ...     details={
        ...         "required_keys": ["OPENAI_API_KEY"],
        ...         "config_file": "config.yaml"
        ...     }
        ... )
    """

    pass


class ConnectionError(NeuroSyncError):
    """
    Raised when connection to external services fails.

    Indicates network connectivity or service availability issues:
        - Database connection failures
        - API endpoint unreachable or timeout
        - Authentication failures with external services
        - Network configuration problems
        - Service unavailability or maintenance

    Common scenarios:
        - PostgreSQL/Redis connection timeout
        - LLM provider API authentication errors
        - Vector store service unavailable
        - Network firewall or proxy issues
        - SSL/TLS certificate problems

    Example:
        >>> raise ConnectionError(
        ...     "Failed to connect to OpenAI API",
        ...     error_code="OPENAI_CONNECTION_ERROR",
        ...     details={"endpoint": "https://api.openai.com", "status_code": 503}
        ... )
    """

    pass


class IngestionError(NeuroSyncError):
    """
    Raised when data ingestion operations fail.

    Indicates problems during data extraction and loading:
        - File reading or parsing errors
        - API response handling failures
        - Database query execution problems
        - Data format or encoding issues
        - Source authentication or authorization failures

    Common scenarios:
        - Corrupted or unsupported file formats
        - API rate limiting or quota exceeded
        - Database query syntax or permission errors
        - Network interruptions during data transfer
        - Invalid or malformed source data

    Example:
        >>> raise IngestionError(
        ...     "Unable to parse PDF document",
        ...     error_code="PDF_PARSING_ERROR",
        ...     details={"file_path": "/docs/report.pdf", "file_size": 2048576}
        ... )
    """

    pass


class ProcessingError(NeuroSyncError):
    """
    Raised when text processing operations fail.

    Indicates issues during content transformation and preparation:
        - Text cleaning or normalization failures
        - Chunking strategy execution problems
        - Language detection or processing errors
        - Quality assessment failures
        - Memory or resource exhaustion during processing

    Common scenarios:
        - Encoding issues with special characters
        - Chunking algorithm configuration problems
        - NLP model loading or inference failures
        - Insufficient memory for large documents
        - Invalid text format or structure

    Example:
        >>> raise ProcessingError(
        ...     "Text chunking failed due to memory constraints",
        ...     error_code="CHUNKING_MEMORY_ERROR",
        ...     details={"chunk_size": 512, "content_length": 1000000}
        ... )
    """

    pass


class StorageError(NeuroSyncError):
    """
    Raised when storage operations fail.

    Indicates problems with data persistence and retrieval:
        - Database write or read failures
        - Vector store indexing errors
        - File system permission or space issues
        - Index corruption or consistency problems
        - Backup and recovery operation failures

    Common scenarios:
        - Disk space exhaustion during indexing
        - Vector store index corruption
        - Database transaction rollback errors
        - File permission or ownership issues
        - Concurrent access conflicts

    Example:
        >>> raise StorageError(
        ...     "Vector index write failed",
        ...     error_code="VECTOR_INDEX_WRITE_ERROR",
        ...     details={"index_path": "/data/vectors", "available_space": "100MB"}
        ... )
    """

    pass


class EmbeddingError(NeuroSyncError):
    """Raised when embedding generation fails"""

    pass


class ValidationError(NeuroSyncError):
    """Raised when data validation fails"""

    pass


class LLMError(NeuroSyncError):
    """Raised when LLM operation fails"""

    pass


class PipelineError(NeuroSyncError):
    """Raised when pipeline execution fails"""

    pass
