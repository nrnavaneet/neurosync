"""
Unit tests for the 'process' CLI command.
"""
import json

import pytest
from typer.testing import CliRunner

from neurosync.cli.main import app

runner = CliRunner()


@pytest.fixture
def sample_ingested_file(tmp_path):
    """Creates a sample ingested data file for testing."""
    ingested_data = [
        {
            "success": True,
            "source_id": "test.txt",
            "content": "This is a test document for processing.",
            "metadata": {
                "source_id": "test.txt",
                "source_type": "file",
                "content_type": "text",
            },
        }
    ]
    file_path = tmp_path / "ingested.json"
    file_path.write_text(json.dumps(ingested_data))
    return file_path


@pytest.fixture
def sample_processing_config(tmp_path):
    """Creates a sample processing config file."""
    config_data = {
        "chunking": {"strategy": "recursive", "chunk_size": 500, "chunk_overlap": 50}
    }
    file_path = tmp_path / "processing.json"
    file_path.write_text(json.dumps(config_data))
    return file_path


def test_process_file_command(sample_ingested_file, sample_processing_config):
    """Test the 'process file' command."""
    result = runner.invoke(
        app,
        ["process", "file", str(sample_ingested_file), str(sample_processing_config)],
    )

    assert result.exit_code == 0
    assert "Processing Complete" in result.stdout
    assert "Chunks Generated: 1" in result.stdout
    assert "Sample Chunk" in result.stdout


def test_process_file_with_output(
    sample_ingested_file, sample_processing_config, tmp_path
):
    """Test the 'process file' command with an output file."""
    output_path = tmp_path / "chunks.json"
    result = runner.invoke(
        app,
        [
            "process",
            "file",
            str(sample_ingested_file),
            str(sample_processing_config),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    with open(output_path, "r") as f:
        chunks = json.load(f)
        assert len(chunks) == 1
        assert "chunk_id" in chunks[0]
        assert "This is a test document" in chunks[0]["content"]


def test_process_create_config_command(tmp_path):
    """Test the 'process create-config' command."""
    config_path = tmp_path / "proc_config.json"
    result = runner.invoke(app, ["process", "create-config", str(config_path)])

    assert result.exit_code == 0
    assert "Created default processing configuration" in result.stdout
    assert config_path.exists()
