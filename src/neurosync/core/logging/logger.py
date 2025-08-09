"""
Structured logging configuration for NeuroSync.

This module provides comprehensive logging infrastructure with support for
structured logging, JSON formatting, correlation IDs, and rich console output.
It integrates structlog for advanced logging capabilities while maintaining
compatibility with standard Python logging.

Key Features:
    - Structured logging with consistent metadata
    - JSON formatting for production environments
    - Rich console output for development
    - Automatic correlation ID injection
    - Performance-aware log level filtering
    - File and console output support
    - Integration with monitoring systems

Functions:
    setup_logging(): Initialize logging configuration
    get_logger(name): Get configured logger instance

Configuration:
    Logging behavior is controlled by environment variables:
    - LOG_LEVEL: Minimum log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    - LOG_FORMAT: Output format (json/text)
    - LOG_FILE_PATH: Optional file output path
    - DEBUG: Enable development mode with rich formatting

Example:
    >>> from neurosync.core.logging.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started", document_id="doc123", user_id="user456")
    >>> logger.error("Processing failed", error="FileNotFound", path="/data/file.txt")

The logging system automatically adds context such as:
    - Timestamp (ISO format)
    - Logger name (module path)
    - Log level
    - Process/thread information
    - Stack traces for exceptions
    - Custom correlation IDs for request tracking
"""

import logging
import logging.config
import sys
from pathlib import Path

import structlog
from rich.console import Console
from rich.logging import RichHandler

from neurosync.core.config.settings import settings


def setup_logging() -> None:
    """
    Initialize comprehensive logging configuration.

    Sets up structured logging with configurable output formats, handlers,
    and processing pipelines. Configures both structlog for advanced features
    and standard library logging for compatibility.

    Features configured:
        - Structured logging with consistent metadata
        - Configurable output format (JSON/text)
        - Rich console formatting for development
        - File output support (if LOG_FILE_PATH set)
        - Log level filtering and performance optimization
        - Exception stack trace formatting
        - Timestamp normalization (ISO format)

    Environment Variables:
        LOG_LEVEL: Minimum log level (default: INFO)
        LOG_FORMAT: Output format - 'json' or 'text' (default: json)
        LOG_FILE_PATH: Optional file output path
        DEBUG: Enable development mode with rich formatting
        ENVIRONMENT: Deployment environment (affects handler selection)

    The function automatically selects appropriate handlers:
        - Development: Rich console handler with colors and formatting
        - Production: JSON-formatted stream handler for log aggregation
        - File: Optional file handler when LOG_FILE_PATH is configured

    Example:
        >>> from neurosync.core.logging.logger import setup_logging
        >>> setup_logging()  # Call once at application startup
        >>> # Logging is now configured for the entire application
    """

    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.LOG_FORMAT.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,  # Bound logger metadata
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    handlers = []

    # Console handler with Rich formatting
    if settings.DEBUG or settings.ENVIRONMENT == "development":
        console = Console(stderr=True)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
        rich_handler.setLevel(settings.LOG_LEVEL)
        handlers.append(rich_handler)
    else:
        # Production: JSON formatted logs
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(settings.LOG_LEVEL)
        handlers.append(stream_handler)

    # File handler if specified
    if settings.LOG_FILE_PATH:
        file_path = Path(settings.LOG_FILE_PATH)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(settings.LOG_LEVEL)
        handlers.append(file_handler)

    # Root logger configuration
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        handlers=handlers,
        format="%(message)s",
    )

    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured structured logger instance.

    Creates and returns a structlog BoundLogger instance with the specified
    name. The logger inherits all configuration from setup_logging() and
    provides structured logging capabilities with automatic metadata binding.

    Args:
        name (str): Logger name, typically __name__ of the calling module

    Returns:
        structlog.BoundLogger: Configured logger instance with bound metadata

    Features:
        - Automatic metadata binding (logger name, timestamp, level)
        - Structured logging with key-value pairs
        - Exception handling with stack traces
        - Context preservation across async operations
        - Performance-optimized log level filtering

    Example:
        >>> from neurosync.core.logging.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("User action", user_id="123", action="login")
        >>> logger.error("Database error", error_code=500, table="users")
        >>>
        >>> # Bind persistent context
        >>> request_logger = logger.bind(request_id="req-456")
        >>> request_logger.info("Processing started")
        >>> request_logger.info("Processing completed")

    Note:
        If logging hasn't been configured yet, this function will
        automatically call setup_logging() to ensure proper initialization.
    """
    if not structlog.is_configured():
        setup_logging()
    return structlog.get_logger(name)


def add_correlation_id(correlation_id: str) -> structlog.BoundLogger:
    """
    Create a logger with bound correlation ID for request tracking.

    Creates a logger instance with a pre-bound correlation ID that will
    be automatically included in all log messages. This is useful for
    tracking related log entries across distributed operations or
    request processing chains.

    Args:
        correlation_id (str): Unique identifier for correlating log entries

    Returns:
        structlog.BoundLogger: Logger instance with bound correlation ID

    Use Cases:
        - HTTP request tracking across services
        - Pipeline execution monitoring
        - User session activity logging
        - Distributed system debugging

    Example:
        >>> from neurosync.core.logging.logger import add_correlation_id
        >>> import uuid
        >>>
        >>> # Generate unique correlation ID
        >>> corr_id = str(uuid.uuid4())
        >>> logger = add_correlation_id(corr_id)
        >>>
        >>> # All subsequent logs include correlation_id
        >>> logger.info("Request started", endpoint="/api/process")
        >>> logger.info("Database query", table="documents", duration=0.05)
        >>> logger.info("Request completed", status=200)

    Note:
        The correlation ID is automatically included in all log output
        formats and can be used for log aggregation and analysis in
        monitoring systems like ELK stack or CloudWatch.
    """
    logger = get_logger(__name__)
    logger = logger.bind(correlation_id=correlation_id)
    return logger


# Setup logging on import
setup_logging()
