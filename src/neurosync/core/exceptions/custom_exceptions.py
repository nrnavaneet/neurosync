"""
Custom exceptions for NeuroSync application
"""

from typing import Any, Dict, Optional


class NeuroSyncError(Exception):
    """Base exception for all NeuroSync errors"""

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
    """Raised when there's a configuration issue"""

    pass


class ConnectionError(NeuroSyncError):
    """Raised when connection to external service fails"""

    pass


class IngestionError(NeuroSyncError):
    """Raised when data ingestion fails"""

    pass


class ProcessingError(NeuroSyncError):
    """Raised when data processing fails"""

    pass


class StorageError(NeuroSyncError):
    """Raised when storage operation fails"""

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
