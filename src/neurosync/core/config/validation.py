"""
Configuration validation utilities for NeuroSync
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ValidationError, field_validator

from neurosync.core.exceptions.custom_exceptions import ConfigurationError


class ConnectorConfig(BaseModel):
    """Base connector configuration schema"""

    name: str
    type: str
    config: Dict[str, Any]
    enabled: bool = True

    @field_validator("type")
    @classmethod
    def validate_connector_type(cls, v):
        valid_types = ["file", "api", "database"]
        if v not in valid_types:
            raise ValueError(f"type must be one of: {valid_types}")
        return v


class IngestionConfig(BaseModel):
    """Main ingestion configuration schema"""

    sources: List[ConnectorConfig]
    global_settings: Optional[Dict[str, Any]] = {}

    @field_validator("sources")
    @classmethod
    def validate_sources_not_empty(cls, v):
        if not v:
            raise ValueError("sources cannot be empty")
        return v

    @field_validator("sources")
    @classmethod
    def validate_unique_source_names(cls, v):
        names = [source.name for source in v]
        if len(names) != len(set(names)):
            raise ValueError("source names must be unique")
        return v


class ConfigValidator:
    """Configuration validator for ingestion configs"""

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        path = Path(file_path)

        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")

        try:
            with open(path, "r") as f:
                if path.suffix.lower() in [".yaml", ".yml"]:
                    return yaml.safe_load(f)
                elif path.suffix.lower() == ".json":
                    return json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported file format: {path.suffix}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> IngestionConfig:
        """Validate ingestion configuration"""
        try:
            return IngestionConfig(**config)
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")

    @staticmethod
    def validate_file(file_path: str) -> IngestionConfig:
        """Load and validate configuration file"""
        config = ConfigValidator.load_config(file_path)
        return ConfigValidator.validate_config(config)


class ConfigGenerator:
    """Generate configuration templates"""

    @staticmethod
    def generate_basic_config() -> Dict[str, Any]:
        """Generate basic configuration template"""
        return {
            "sources": [
                {
                    "name": "local_files",
                    "type": "file",
                    "enabled": True,
                    "config": {
                        "base_path": "./data",
                        "supported_extensions": [".txt", ".md", ".pdf"],
                        "recursive": True,
                        "max_file_size_mb": 100,
                    },
                }
            ],
            "global_settings": {
                "max_concurrent_sources": 3,
                "timeout_seconds": 300,
                "retry_attempts": 3,
            },
        }

    @staticmethod
    def generate_multi_source_config() -> Dict[str, Any]:
        """Generate multi-source configuration template"""
        return {
            "sources": [
                {
                    "name": "local_documents",
                    "type": "file",
                    "enabled": True,
                    "config": {
                        "base_path": "./documents",
                        "supported_extensions": [".txt", ".md", ".pdf", ".docx"],
                        "recursive": True,
                    },
                },
                {
                    "name": "company_api",
                    "type": "api",
                    "enabled": True,
                    "config": {
                        "base_url": "https://api.company.com",
                        "auth_type": "bearer",
                        "auth_token": "${API_TOKEN}",
                        "endpoints": [
                            {"path": "/documents", "method": "GET"},
                            {"path": "/articles", "method": "GET"},
                        ],
                    },
                },
                {
                    "name": "postgres_db",
                    "type": "database",
                    "enabled": True,
                    "config": {
                        "database_type": "postgresql",
                        "host": "${DB_HOST}",
                        "port": 5432,
                        "database": "${DB_NAME}",
                        "username": "${DB_USER}",
                        "password": "${DB_PASSWORD}",
                        "tables": ["documents", "knowledge_base"],
                    },
                },
            ],
            "global_settings": {
                "max_concurrent_sources": 3,
                "timeout_seconds": 600,
                "retry_attempts": 3,
                "batch_size": 100,
            },
        }
