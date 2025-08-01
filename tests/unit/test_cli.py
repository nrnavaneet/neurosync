"""
Unit tests for CLI functionality
"""

from typer.testing import CliRunner

from neurosync.cli.main import app

runner = CliRunner()


def test_version_command():
    """
    Test version command
    """
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "NeuroSync" in result.output
    assert "Version" in result.output


def test_help_command():
    """
    Test help command
    """
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AI-Native ETL Pipeline" in result.stdout


def test_init_command(tmp_path):
    """Test project initialization"""
    project_name = "test-project"
    result = runner.invoke(app, ["init", project_name, "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert f"Initialized NeuroSync project '{project_name}'" in result.stdout

    # Check if directories were created
    project_dir = tmp_path / project_name
    assert project_dir.exists()
    assert (project_dir / "config").exists()
    assert (project_dir / "data").exists()
    assert (project_dir / "logs").exists()


def test_ingest_file_command_nonexistent():
    """Test file ingestion with non-existent file"""
    result = runner.invoke(app, ["ingest", "file", "/nonexistent/path"])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout


def test_pipeline_list_command():
    """Test pipeline list command"""
    result = runner.invoke(app, ["pipeline", "list"])
    assert result.exit_code == 0
    assert "Available Pipelines" in result.stdout


def test_status_system_command():
    """Test system status command"""
    result = runner.invoke(app, ["status", "system"])
    assert result.exit_code == 0
