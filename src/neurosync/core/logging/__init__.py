"""
NeuroSync Logging Module - Structured Application Logging.

This module provides comprehensive logging infrastructure for the NeuroSync
application with structured logging, configurable formats, and integration
with monitoring systems. It implements best practices for observability
and debugging in production environments.

Key Features:
    - Structured logging with JSON and text output formats
    - Configurable log levels and filtering
    - Rich console formatting for development
    - File output with rotation and compression
    - Correlation ID support for request tracking
    - Integration with monitoring and alerting systems
    - Performance optimization with lazy evaluation
    - Exception handling with stack trace capture

Components:
    - logger: Main logging configuration and factory functions
    - formatters: Custom log formatters for different output targets

Logging Hierarchy:
    - Application logs: Business logic and user actions
    - System logs: Infrastructure and service health
    - Performance logs: Timing and resource usage metrics
    - Security logs: Authentication and authorization events
    - Debug logs: Detailed troubleshooting information

Output Formats:
    - JSON: Structured format for log aggregation systems
    - Text: Human-readable format for development and console output
    - Rich: Enhanced console output with colors and formatting

The logging system automatically configures based on environment:
    - Development: Rich console output with debug information
    - Production: JSON structured logs for aggregation
    - Testing: Minimal output with configurable levels

Example:
    >>> from neurosync.core.logging import get_logger
    >>>
    >>> # Get logger for current module
    >>> logger = get_logger(__name__)
    >>>
    >>> # Structured logging with context
    >>> logger.info("User login", user_id="123", ip_address="192.168.1.100")
    >>> logger.error("Database error", table="users", error_code=500)
    >>>
    >>> # Bind persistent context for request tracking
    >>> request_logger = logger.bind(request_id="req-456")
    >>> request_logger.info("Processing started")
    >>> request_logger.info("Processing completed", duration=1.25)

For logging best practices and configuration options, see:
    - docs/logging-configuration.md
    - docs/monitoring-integration.md
    - examples/logging-patterns.py
"""
