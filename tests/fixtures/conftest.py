"""
Pytest configuration and fixtures for NeuroSync tests
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Import NeuroSync components for testing
from neurosync.core.config.settings import Settings


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Test settings with temporary paths"""
    with tempfile.TemporaryDirectory() as temp_dir:
        return Settings(
            ENVIRONMENT="testing",
            DEBUG=True,
            DATABASE_URL="sqlite:///:memory:",
            REDIS_URL="redis://localhost:6379/1",
            DATA_DIR=temp_dir,
            LOG_LEVEL="DEBUG",
        )


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing"""
    return """
    This is a sample document for testing NeuroSync functionality.
    It contains multiple paragraphs and sentences that can be used
    to test chunking, embedding, and retrieval capabilities.

    The second paragraph provides additional context and complexity
    to ensure our processing pipeline works correctly with various
    text structures and formats.
    """


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing connectors"""
    return {
        "sources": [
            {
                "name": "test_files",
                "type": "file",
                "enabled": True,
                "config": {
                    "base_path": "./test_data",
                    "file_patterns": ["*.txt", "*.md"],
                    "batch_size": 10,
                },
            }
        ],
        "global_settings": {"max_concurrent_sources": 2, "timeout_seconds": 30},
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_files(temp_directory):
    """Create sample files for testing"""
    files = []

    # Create text file
    txt_file = temp_directory / "sample.txt"
    txt_file.write_text("This is a sample text file for testing.")
    files.append(txt_file)

    # Create markdown file
    md_file = temp_directory / "README.md"
    md_file.write_text("# Test Document\nThis is a test markdown file.")
    files.append(md_file)

    # Create JSON file
    json_file = temp_directory / "data.json"
    json_file.write_text('{"test": "data", "number": 42}')
    files.append(json_file)

    return files


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a temporary sample file"""
    content = """
    # Sample Document

    This is a test document for NeuroSync ingestion testing.

    ## Section 1

    Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

    ## Section 2

    Ut enim ad minim veniam, quis nostrud exercitation ullamco
    laboris nisi ut aliquip ex ea commodo consequat.
    """

    file_path = tmp_path / "sample.txt"
    file_path.write_text(content.strip())
    return file_path


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    """Create a temporary CSV file"""
    content = """name,age,city
John Doe,30,New York
Jane Smith,25,Los Angeles
Bob Johnson,35,Chicago"""

    file_path = tmp_path / "sample.csv"
    file_path.write_text(content.strip())
    return file_path
