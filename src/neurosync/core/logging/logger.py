"""
Structured logging configuration for NeuroSync
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
    """Setup structured logging with JSON format and correlation IDs"""

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
    """Get a structured logger instance"""
    if not structlog.is_configured():
        setup_logging()
    return structlog.get_logger(name)


def add_correlation_id(correlation_id: str) -> None:
    """Add correlation ID to logger context"""
    logger = get_logger(__name__)
    logger = logger.bind(correlation_id=correlation_id)
    return logger


# Setup logging on import
setup_logging()
