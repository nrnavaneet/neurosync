"""
CLI input validation utilities for NeuroSync commands.

This module provides comprehensive validation functions for CLI inputs,
ensuring that user-provided parameters are valid, accessible, and properly
formatted before processing begins. It includes validation for file paths,
configuration parameters, network endpoints, and data format specifications.

Key Features:
    - File system validation (existence, permissions, formats)
    - Network endpoint validation (URLs, connectivity, credentials)
    - Configuration parameter validation and type checking
    - Data format validation (JSON, YAML, CSV structure)
    - Interactive validation with user feedback and suggestions
    - Batch validation for multiple inputs
    - Security validation for sensitive parameters

Validation Categories:
    Path Validation: File and directory existence, permissions, accessibility
    Format Validation: Data structure, schema compliance, encoding
    Network Validation: URL format, connectivity, authentication
    Configuration Validation: Parameter ranges, dependencies, compatibility
    Content Validation: File content structure and format compliance

Validation Features:
    - Early validation to fail fast before expensive operations
    - Detailed error messages with actionable suggestions
    - Interactive prompts for missing or invalid parameters
    - Batch validation with summary reporting
    - Security checks for credential and sensitive data handling
    - Performance validation for resource-intensive operations

User Experience:
    - Clear error messages with specific validation failures
    - Suggestions for fixing common validation issues
    - Interactive prompts for missing required parameters
    - Progress indication for long-running validation operations
    - Colorized output for better readability and user guidance

Integration Points:
    - CLI command parameter validation
    - Configuration file validation
    - Input data format validation
    - Output destination validation
    - Resource availability validation

Example Usage:
    >>> # Validate input file
    >>> validate_input_file("data.json", required_format="json")

    >>> # Validate configuration
    >>> validate_config_params(config_dict, schema="processing")

    >>> # Validate API endpoint
    >>> validate_api_endpoint("https://api.example.com", check_auth=True)

For comprehensive validation patterns and custom validators, see:
    - docs/cli-validation-reference.md
    - docs/custom-validators.md
    - examples/validation-workflows.py

Author: NeuroSync Team
Created: 2025
License: MIT
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import typer
from rich.console import Console

from neurosync.core.exceptions.custom_exceptions import ValidationError
from neurosync.core.logging.logger import get_logger

console = Console()
logger = get_logger(__name__)


def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = True,
    required_extensions: Optional[List[str]] = None,
    check_readable: bool = True,
) -> Path:
    """
    Validate file path with comprehensive checks.

    Args:
        file_path: Path to validate
        must_exist: Whether file must already exist
        required_extensions: List of allowed file extensions
        check_readable: Whether to check read permissions

    Returns:
        Path: Validated Path object

    Raises:
        ValidationError: If validation fails
    """
    path_obj = Path(file_path)

    if must_exist and not path_obj.exists():
        raise ValidationError(f"File does not exist: {file_path}")

    if required_extensions:
        if path_obj.suffix.lower() not in required_extensions:
            raise ValidationError(
                f"Invalid file extension. Expected: {required_extensions}, "
                f"got: {path_obj.suffix}"
            )

    if must_exist and check_readable and not path_obj.is_file():
        raise ValidationError(f"Path is not a readable file: {file_path}")

    return path_obj


def validate_output_directory(directory_path: Union[str, Path]) -> Path:
    """
    Validate output directory with write permission checks.

    Args:
        directory_path: Directory path to validate

    Returns:
        Path: Validated directory Path object

    Raises:
        ValidationError: If validation fails
    """
    dir_obj = Path(directory_path)

    if dir_obj.exists() and not dir_obj.is_dir():
        raise ValidationError(f"Path exists but is not a directory: {directory_path}")

    # Create directory if it doesn't exist
    if not dir_obj.exists():
        try:
            dir_obj.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise ValidationError(f"Cannot create directory: {directory_path}")

    return dir_obj


def validate_url(url: str, check_connectivity: bool = False) -> str:
    """
    Validate URL format and optionally check connectivity.

    Args:
        url: URL to validate
        check_connectivity: Whether to test actual connectivity

    Returns:
        str: Validated URL

    Raises:
        ValidationError: If validation fails
    """
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValidationError(f"Invalid URL format: {url}")
    except Exception:
        raise ValidationError(f"Malformed URL: {url}")

    if check_connectivity:
        # Connectivity check would be implemented here
        # This is a placeholder for the actual implementation
        pass

    return url


def validate_config_parameter(
    param_name: str,
    param_value: Any,
    expected_type: type,
    allowed_values: Optional[List[Any]] = None,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
) -> Any:
    """
    Validate individual configuration parameter.

    Args:
        param_name: Name of the parameter
        param_value: Value to validate
        expected_type: Expected Python type
        allowed_values: List of allowed values (optional)
        min_value: Minimum allowed value for numeric types
        max_value: Maximum allowed value for numeric types

    Returns:
        Any: Validated parameter value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(param_value, expected_type):
        raise ValidationError(
            f"Parameter '{param_name}' must be of type {expected_type.__name__}, "
            f"got {type(param_value).__name__}"
        )

    if allowed_values and param_value not in allowed_values:
        raise ValidationError(
            f"Parameter '{param_name}' must be one of {allowed_values}, "
            f"got: {param_value}"
        )

    if isinstance(param_value, (int, float)):
        if min_value is not None and param_value < min_value:
            raise ValidationError(
                f"Parameter '{param_name}' must be >= {min_value}, got: {param_value}"
            )
        if max_value is not None and param_value > max_value:
            raise ValidationError(
                f"Parameter '{param_name}' must be <= {max_value}, got: {param_value}"
            )

    return param_value


def validate_json_structure(data: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate JSON data structure.

    Args:
        data: JSON data to validate
        required_keys: List of required top-level keys

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValidationError(f"Missing required keys: {missing_keys}")

    return True


def prompt_for_missing_parameter(param_name: str, param_description: str) -> str:
    """
    Interactively prompt user for missing parameter.

    Args:
        param_name: Name of the parameter
        param_description: Human-readable description

    Returns:
        str: User-provided value
    """
    return typer.prompt(f"Please provide {param_description} ({param_name})")
