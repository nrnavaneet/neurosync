"""
Unit tests for NeuroSync pipeline CLI commands
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from typer.testing import CliRunner as TyperRunner

from neurosync.cli.commands.pipeline import (
    _analyze_content_for_chunking,
    _compare_all_strategies,
    _recommend_chunking_strategy,
    app,
)
from neurosync.ingestion.base import ContentType


class TestPipelineAnalysis:
    """Test content analysis and strategy recommendation functionality"""

    def test_analyze_content_for_chunking_basic(self):
        """Test basic content analysis"""
        content = (
            "This is a simple text with multiple sentences. "
            "It has some structure and paragraphs.\n\n"
            "This is another paragraph with more content."
        )

        analysis = _analyze_content_for_chunking(content, ContentType.TEXT)

        assert analysis["content_length"] == len(content)
        assert analysis["word_count"] > 0
        assert analysis["line_count"] >= 1
        assert analysis["paragraph_count"] >= 2
        assert "recommended_strategy" in analysis
        assert "confidence" in analysis
        assert "reasons" in analysis

    def test_analyze_markdown_content(self):
        """Test analysis of markdown content with structure"""
        content = """# Main Title

## Section 1
This is some content with **bold** text.

### Subsection
- List item 1
- List item 2

```python
print("code block")
```

## Section 2
More content here.
"""

        analysis = _analyze_content_for_chunking(content, ContentType.MARKDOWN)

        assert analysis["has_structure"] is True
        assert analysis["has_lists"] is True
        assert analysis["has_code"] is True
        assert analysis["recommended_strategy"] in [
            "hierarchical",
            "document_structure",
        ]
        assert analysis["confidence"] > 0.5

    def test_analyze_json_content(self):
        """Test analysis of JSON content"""
        content = '{"key": "value", "nested": {"data": [1, 2, 3]}}'

        analysis = _analyze_content_for_chunking(content, ContentType.JSON)

        assert analysis["has_structure"] is True
        assert analysis["recommended_strategy"] is not None

    def test_compare_all_strategies(self):
        """Test strategy comparison functionality"""
        content = """# Document Title

## Section with table
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |

## Code Section
```python
def example():
    return "test"
```
"""

        scores = _compare_all_strategies(content, ContentType.MARKDOWN)

        assert isinstance(scores, dict)
        assert len(scores) == 6  # All strategies
        assert "recursive" in scores
        assert "semantic" in scores
        assert "sliding_window" in scores
        assert "token_aware_sliding" in scores
        assert "hierarchical" in scores
        assert "document_structure" in scores

        # Document structure should score highest for this content
        assert scores["document_structure"] >= scores["hierarchical"]

    def test_recommend_chunking_strategy(self):
        """Test strategy recommendation"""
        content = "Simple text without much structure. Just continuous content flowing."

        strategy, confidence, reasons = _recommend_chunking_strategy(
            content, ContentType.TEXT
        )

        assert strategy in [
            "recursive",
            "semantic",
            "sliding_window",
            "token_aware_sliding",
            "hierarchical",
            "document_structure",
        ]
        assert 0 <= confidence <= 1
        assert isinstance(reasons, list)


class TestPipelineCommands:
    """Test pipeline CLI commands"""

    def setup_method(self):
        """Setup for each test"""
        self.runner = TyperRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Cleanup after each test"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("neurosync.cli.commands.pipeline._ingest_single_file")
    @patch("neurosync.cli.commands.pipeline._execute_direct_pipeline")
    def test_run_with_auto_strategy(self, mock_execute, mock_ingest):
        """Test pipeline run with auto strategy selection"""
        # Create a test file
        test_file = self.temp_path / "test.md"
        test_file.write_text("# Test\nContent here")

        mock_ingest.return_value = [MagicMock()]

        result = self.runner.invoke(
            app,
            [
                "run",
                str(test_file),
                "--type",
                "file",
                "--strategy",
                "auto",
                "--output",
                "test_output.json",
            ],
        )

        assert result.exit_code == 0
        mock_execute.assert_called_once()

    @patch("neurosync.cli.commands.pipeline._ingest_single_file")
    def test_run_dry_run_with_auto(self, mock_ingest):
        """Test dry-run mode with auto strategy"""
        test_file = self.temp_path / "test.txt"
        test_file.write_text("Simple test content")

        result = self.runner.invoke(
            app,
            [
                "run",
                str(test_file),
                "--type",
                "file",
                "--strategy",
                "auto",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "dry-run mode" in result.stdout
        assert "automatically select" in result.stdout

    @patch("requests.get")
    def test_analyze_api_source(self, mock_get):
        """Test analyze command with API source"""
        mock_response = MagicMock()
        mock_response.text = '{"test": "data"}'
        mock_response.headers = {"content-type": "application/json"}
        mock_get.return_value = mock_response

        result = self.runner.invoke(
            app, ["analyze", "https://api.example.com/data", "--type", "api"]
        )

        assert result.exit_code == 0
        assert "Analysis Results" in result.stdout
        assert "Strategy Recommendation" in result.stdout

    def test_analyze_file_source(self):
        """Test analyze command with file source"""
        # Create test file with structured content
        test_file = self.temp_path / "structured.md"
        test_file.write_text(
            """# Main Title

## Section 1
Content with **formatting**.

### Subsection
- List item
- Another item

```python
print("code")
```
"""
        )

        result = self.runner.invoke(app, ["analyze", str(test_file), "--type", "file"])

        assert result.exit_code == 0
        assert "Analysis Results" in result.stdout
        assert "Strategy Recommendation" in result.stdout
        assert "document_structure" in result.stdout or "hierarchical" in result.stdout

    def test_analyze_file_compare_all(self):
        """Test analyze command with compare-all option"""
        test_file = self.temp_path / "test.md"
        test_file.write_text("# Title\nContent here")

        result = self.runner.invoke(
            app, ["analyze", str(test_file), "--type", "file", "--compare-all"]
        )

        assert result.exit_code == 0
        assert "Strategy Comparison" in result.stdout
        assert "recursive" in result.stdout
        assert "semantic" in result.stdout
        assert "hierarchical" in result.stdout

    def test_analyze_directory(self):
        """Test analyze command with directory"""
        # Create multiple test files
        (self.temp_path / "file1.md").write_text("# File 1\nContent")
        (self.temp_path / "file2.txt").write_text("Simple text content")

        result = self.runner.invoke(
            app, ["analyze", str(self.temp_path), "--type", "file"]
        )

        assert result.exit_code == 0
        assert "Analysis Results" in result.stdout
        assert "Overall Analysis" in result.stdout

    def test_analyze_auto_detect_type(self):
        """Test analyze command with auto type detection"""
        test_file = self.temp_path / "auto_detect.txt"
        test_file.write_text("Test content")

        result = self.runner.invoke(app, ["analyze", str(test_file)])

        assert result.exit_code == 0
        assert "Analysis Results" in result.stdout

    def test_analyze_invalid_source(self):
        """Test analyze command with invalid source"""
        result = self.runner.invoke(
            app, ["analyze", "/nonexistent/path", "--type", "file"]
        )

        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_create_pipeline_basic(self):
        """Test pipeline creation with basic template"""
        result = self.runner.invoke(
            app,
            [
                "create",
                "test_pipeline",
                "--template",
                "basic",
                "--output-dir",
                str(self.temp_path),
            ],
        )

        assert result.exit_code == 0
        assert "Created pipeline" in result.stdout

        # Check config file was created
        config_file = self.temp_path / "test_pipeline_pipeline.json"
        assert config_file.exists()

        with open(config_file) as f:
            config = json.load(f)
        assert config["name"] == "test_pipeline"
        assert config["processing"]["strategy"] == "recursive"

    def test_create_pipeline_all_templates(self):
        """Test pipeline creation with all available templates"""
        templates = [
            "basic",
            "advanced",
            "comprehensive",
            "semantic_focused",
            "sliding_window",
            "token_aware",
            "document_structure",
            "performance_test",
            "minimal",
            "custom",
        ]

        for template in templates:
            result = self.runner.invoke(
                app,
                [
                    "create",
                    f"test_{template}",
                    "--template",
                    template,
                    "--output-dir",
                    str(self.temp_path),
                ],
            )

            assert result.exit_code == 0, f"Failed to create {template} template"

            config_file = self.temp_path / f"test_{template}_pipeline.json"
            assert config_file.exists(), f"Config file not created for {template}"

    def test_create_pipeline_invalid_template(self):
        """Test pipeline creation with invalid template"""
        result = self.runner.invoke(
            app, ["create", "test_pipeline", "--template", "nonexistent"]
        )

        assert result.exit_code == 1
        assert "Unknown template" in result.stdout

    def test_templates_command(self):
        """Test templates listing command"""
        result = self.runner.invoke(app, ["templates"])

        assert result.exit_code == 0
        assert "Available Pipeline Templates" in result.stdout
        assert "basic" in result.stdout
        assert "advanced" in result.stdout
        assert "Usage:" in result.stdout

    @patch("neurosync.cli.commands.pipeline._execute_pipeline")
    def test_run_with_config_file(self, mock_execute):
        """Test running pipeline with config file"""
        # Create test config
        config = {
            "name": "test_pipeline",
            "ingestion": {
                "sources": [{"name": "test", "path": "test.json", "type": "file"}]
            },
            "processing": {"strategy": "recursive", "chunk_size": 1024},
        }

        config_file = self.temp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        result = self.runner.invoke(app, ["run", str(config_file)])

        assert result.exit_code == 0
        mock_execute.assert_called_once()

    def test_run_with_invalid_config(self):
        """Test running pipeline with invalid config file"""
        # Create invalid JSON config
        config_file = self.temp_path / "invalid_config.json"
        config_file.write_text("invalid json content")

        result = self.runner.invoke(app, ["run", str(config_file)])

        assert result.exit_code == 1
        assert "Invalid JSON" in result.stdout

    def test_run_with_nonexistent_config(self):
        """Test running pipeline with nonexistent config file"""
        result = self.runner.invoke(app, ["run", "/nonexistent/config.json"])

        assert result.exit_code == 1
        assert "does not exist" in result.stdout

    def test_run_without_arguments(self):
        """Test running pipeline without arguments"""
        result = self.runner.invoke(app, ["run"])

        assert result.exit_code == 1
        assert "Please provide" in result.stdout

    def test_list_pipelines_empty(self):
        """Test listing pipelines when none exist"""
        # Change to temp directory where no configs exist
        import os

        old_cwd = os.getcwd()
        os.chdir(self.temp_path)

        try:
            result = self.runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "No pipeline configurations found" in result.stdout
        finally:
            os.chdir(old_cwd)

    def test_list_pipelines_with_configs(self):
        """Test listing pipelines with existing configs"""
        # Create test config files
        config1 = {
            "name": "pipeline1",
            "processing": {"strategy": "recursive"},
            "ingestion": {"sources": []},
        }
        config2 = {
            "name": "pipeline2",
            "processing": {"strategy": "semantic"},
            "ingestion": {"sources": []},
        }

        (self.temp_path / "pipeline1_pipeline.json").write_text(json.dumps(config1))
        (self.temp_path / "pipeline2_pipeline.json").write_text(json.dumps(config2))

        # Change to temp directory for the test
        import os

        old_cwd = os.getcwd()
        os.chdir(self.temp_path)

        try:
            result = self.runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "Available Pipeline Configurations" in result.stdout
            assert "pipeline1" in result.stdout
            assert "pipeline2" in result.stdout
        finally:
            os.chdir(old_cwd)


class TestPipelineIntegration:
    """Integration tests for pipeline functionality"""

    def setup_method(self):
        """Setup for each test"""
        self.runner = TyperRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Cleanup after each test"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("neurosync.cli.commands.pipeline.ProcessingManager")
    @patch("requests.get")
    def test_full_pipeline_api_integration(self, mock_get, mock_manager):
        """Test full pipeline with API ingestion"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.text = '{"title": "Test", "content": "API test content"}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Mock processing manager
        mock_chunks = [MagicMock()]
        mock_chunks[0].to_dict.return_value = {"chunk_id": "test", "content": "chunk"}
        mock_manager.return_value.process.return_value = mock_chunks

        output_file = self.temp_path / "api_results.json"

        result = self.runner.invoke(
            app,
            [
                "run",
                "https://api.example.com/test",
                "--type",
                "api",
                "--strategy",
                "recursive",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert "Pipeline Execution Complete" in result.stdout
        mock_get.assert_called_once()
        mock_manager.assert_called_once()

    @patch("neurosync.cli.commands.pipeline.ProcessingManager")
    def test_full_pipeline_file_integration(self, mock_manager):
        """Test full pipeline with file ingestion"""
        # Create test file
        test_file = self.temp_path / "test_document.md"
        test_file.write_text(
            """# Test Document

## Section 1
This is test content for integration testing.

### Subsection
- Item 1
- Item 2

```python
print("test code")
```
"""
        )

        # Mock processing manager
        mock_chunks = [MagicMock() for _ in range(3)]
        for i, chunk in enumerate(mock_chunks):
            chunk.to_dict.return_value = {
                "chunk_id": f"chunk_{i}",
                "content": f"content_{i}",
            }
        mock_manager.return_value.process.return_value = mock_chunks

        output_file = self.temp_path / "file_results.json"

        result = self.runner.invoke(
            app,
            [
                "run",
                str(test_file),
                "--type",
                "file",
                "--strategy",
                "auto",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert "Pipeline Execution Complete" in result.stdout
        assert "Auto-selected strategy" in result.stdout
        mock_manager.assert_called_once()

    def test_analyze_and_run_workflow(self):
        """Test the workflow of analyzing content then running pipeline"""
        # Create test file
        test_file = self.temp_path / "workflow_test.md"
        test_file.write_text(
            """# Workflow Test

## Introduction
This document tests the analyze -> run workflow.

### Features
- Feature 1
- Feature 2

## Conclusion
End of document.
"""
        )

        # First analyze the content
        analyze_result = self.runner.invoke(
            app, ["analyze", str(test_file), "--type", "file"]
        )

        assert analyze_result.exit_code == 0
        assert "Strategy Recommendation" in analyze_result.stdout

        # Then run with auto strategy (which should use the same logic)
        with patch("neurosync.processing.manager.ProcessingManager") as mock_manager:
            mock_chunks = [MagicMock()]
            mock_chunks[0].to_dict.return_value = {
                "chunk_id": "test",
                "content": "chunk",
            }
            mock_manager.return_value.process.return_value = mock_chunks

            run_result = self.runner.invoke(
                app,
                [
                    "run",
                    str(test_file),
                    "--type",
                    "file",
                    "--strategy",
                    "auto",
                    "--output",
                    str(self.temp_path / "workflow_results.json"),
                ],
            )

            assert run_result.exit_code == 0
            assert "Auto-selected strategy" in run_result.stdout


class TestPipelineErrorHandling:
    """Test error handling in pipeline commands"""

    def setup_method(self):
        """Setup for each test"""
        self.runner = TyperRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Cleanup after each test"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_source_type(self):
        """Test error handling for invalid source type"""
        test_file = self.temp_path / "test.txt"
        test_file.write_text("test content")

        result = self.runner.invoke(
            app,
            [
                "run",
                str(test_file),
                "--type",
                "invalid_type",
                "--strategy",
                "recursive",
            ],
        )

        assert result.exit_code == 1
        assert "Invalid source type" in result.stdout

    def test_invalid_strategy(self):
        """Test error handling for invalid strategy in config"""
        config = {
            "name": "test",
            "ingestion": {"sources": []},
            "processing": {"strategy": "invalid_strategy"},
        }

        config_file = self.temp_path / "invalid_strategy.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        with patch(
            "neurosync.cli.commands.pipeline._validate_pipeline_config"
        ) as mock_validate:
            mock_validate.side_effect = SystemExit(1)

            result = self.runner.invoke(app, ["run", str(config_file), "--dry-run"])

            assert result.exit_code == 1

    @patch("requests.get")
    def test_api_connection_error(self, mock_get):
        """Test error handling for API connection failure"""
        mock_get.side_effect = Exception("Connection failed")

        result = self.runner.invoke(
            app, ["analyze", "https://nonexistent-api.com/data", "--type", "api"]
        )

        assert result.exit_code == 1
        assert "Error fetching API" in result.stdout

    def test_file_permission_error(self):
        """Test error handling for file permission errors"""
        # This test might not work on all systems, so we'll mock it
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = PermissionError("Permission denied")

            result = self.runner.invoke(
                app, ["analyze", "/restricted/file.txt", "--type", "file"]
            )

            assert result.exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__])
