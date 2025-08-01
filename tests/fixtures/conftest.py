"""
Pytest configuration and fixtures for NeuroSync tests
"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neurosync.core.config.settings import Settings
from neurosync.serving.api.main import app


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
def api_client() -> TestClient:
    """FastAPI test client"""
    return TestClient(app)


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
