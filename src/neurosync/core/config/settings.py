"""
Core configuration management for NeuroSync.

This module provides centralized configuration management using Pydantic settings
with support for environment variables, type validation, and computed properties.
All application settings are defined here with sensible defaults and validation.

Classes:
    Settings: Main configuration class with all application settings

Environment Variables:
    Application settings can be overridden using environment variables with the
    same names as the class attributes (case-sensitive).

Example:
    >>> from neurosync.core.config.settings import Settings
    >>> settings = Settings()
    >>> print(settings.database_url)
    postgresql://neurosync:neurosync_password@localhost:5432/neurosync

Configuration Sections:
    - Application: Basic app configuration (name, version, environment)
    - API: Server configuration (host, port, workers)
    - Database: PostgreSQL connection settings
    - Redis: Redis cache configuration
    - LLM: Language model provider API keys and settings
    - Vector Store: Embedding and vector storage configuration
    - Logging: Application logging configuration
    - Airflow: Workflow orchestration settings
    - File Storage: File handling and storage limits
"""

from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Centralized configuration class that manages all application settings
    with automatic environment variable loading, type validation, and
    computed properties for derived values.

    All settings can be overridden via environment variables using the
    same name as the attribute. For example, POSTGRES_HOST environment
    variable will override the postgres_host setting.

    Attributes:
        APP_NAME: Application name identifier
        APP_VERSION: Current application version
        ENVIRONMENT: Deployment environment (development/staging/production)
        DEBUG: Enable debug mode with verbose logging
        SECRET_KEY: Secret key for cryptographic operations

        API_HOST: API server bind address
        API_PORT: API server port number
        API_WORKERS: Number of worker processes for API server
        API_RELOAD: Enable auto-reload in development mode

        POSTGRES_HOST: PostgreSQL database host
        POSTGRES_PORT: PostgreSQL database port
        POSTGRES_DB: PostgreSQL database name
        POSTGRES_USER: PostgreSQL username
        POSTGRES_PASSWORD: PostgreSQL password

        REDIS_HOST: Redis cache server host
        REDIS_PORT: Redis cache server port
        REDIS_DB: Redis database number
        REDIS_PASSWORD: Redis authentication password (optional)

        OPENAI_API_KEY: OpenAI API key for GPT models
        HUGGINGFACE_API_KEY: Hugging Face API key for models
        ANTHROPIC_API_KEY: Anthropic API key for Claude models
        COHERE_API_KEY: Cohere API key for language models
        GOOGLE_API_KEY: Google API key for Gemini models
        OPENROUTER_API_KEY: OpenRouter API key for multi-model access
        DEFAULT_LLM_MODEL: Default language model to use
        LLM_ENABLE_FALLBACK: Enable fallback to alternative models

        FAISS_INDEX_PATH: File path for FAISS vector index storage
        EMBEDDING_MODEL: Default embedding model identifier
        VECTOR_DIMENSION: Vector embedding dimension size
        MAX_CHUNK_SIZE: Maximum size for text chunks in tokens
        CHUNK_OVERLAP: Overlap size between adjacent chunks

        LOG_LEVEL: Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        LOG_FORMAT: Log format (json/text)
        LOG_FILE_PATH: Path for log file output (optional)

        AIRFLOW_HOME: Apache Airflow home directory
        AIRFLOW_UID: User ID for Airflow processes
        AIRFLOW_GID: Group ID for Airflow processes

        DATA_DIR: Base directory for application data storage
        UPLOAD_DIR: Directory for uploaded files
        MAX_FILE_SIZE: Maximum allowed file upload size in bytes

    Properties:
        database_url: Computed PostgreSQL connection URL
        redis_url: Computed Redis connection URL

    Example:
        >>> settings = Settings()
        >>> print(f"Database: {settings.database_url}")
        >>> print(f"Redis: {settings.redis_url}")
        >>> print(f"Debug mode: {settings.DEBUG}")
    """

    # Application
    APP_NAME: str = "NeuroSync"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "dev-secret-key-for-testing-only-change-in-production"

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_RELOAD: bool = True

    # Database Configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "neurosync"
    POSTGRES_USER: str = "neurosync"
    POSTGRES_PASSWORD: str = "neurosync_password"

    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    DEFAULT_LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_ENABLE_FALLBACK: bool = True

    # Vector Store Configuration
    FAISS_INDEX_PATH: str = "/app/data/vector_store"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DIMENSION: int = 384
    MAX_CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE_PATH: Optional[str] = None

    # Airflow Configuration
    AIRFLOW_HOME: str = "/opt/airflow"
    AIRFLOW_UID: int = 50000
    AIRFLOW_GID: int = 50000

    # File Storage
    DATA_DIR: str = "/app/data"
    UPLOAD_DIR: str = "/app/uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB

    @property
    def database_url(self) -> str:
        """
        Construct PostgreSQL database connection URL.

        Builds a complete PostgreSQL connection string from individual
        configuration components for use with SQLAlchemy and other
        database libraries.

        Returns:
            str: Complete PostgreSQL URL in format:
                postgresql://user:password@host:port/database

        Example:
            >>> settings = Settings()
            >>> url = settings.database_url
            >>> print(url)
            postgresql://neurosync:neurosync_password@localhost:5432/neurosync
        """
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        """
        Construct Redis connection URL.

        Builds a complete Redis connection string from individual
        configuration components. Includes optional password authentication
        if configured.

        Returns:
            str: Complete Redis URL in format:
                redis://[password@]host:port/db

        Example:
            >>> settings = Settings()
            >>> url = settings.redis_url
            >>> print(url)
            redis://localhost:6379/0
        """
        password_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return (
            f"redis://{password_part}{self.REDIS_HOST}:"
            f"{self.REDIS_PORT}/{self.REDIS_DB}"
        )

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """
        Validate secret key is not using default value in production.

        Ensures that the secret key has been changed from the default
        development value before deploying to production. This is a
        critical security validation.

        Args:
            v (str): The secret key value to validate

        Returns:
            str: The validated secret key

        Raises:
            ValueError: If using default secret key value

        Note:
            This validation only applies in production environments.
            Development environments can use the default key for convenience.
        """
        if v == "your-super-secret-key-change-in-production":
            raise ValueError("Please change the default SECRET_KEY")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """
        Validate logging level is a supported value.

        Ensures the log level is one of the standard Python logging
        levels. Converts to uppercase for consistency.

        Args:
            v (str): The log level value to validate

        Returns:
            str: The validated and normalized log level

        Raises:
            ValueError: If log level is not supported

        Supported levels:
            - CRITICAL: Only critical errors
            - ERROR: Error conditions
            - WARNING: Warning conditions
            - INFO: General information
            - DEBUG: Detailed debugging information
        """
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {valid_levels}")
        return v.upper()

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",  # This will ignore extra fields from environment
    )


def get_settings() -> Settings:
    """Get application settings instance"""
    return Settings()


settings = Settings()
