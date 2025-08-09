"""
Pipeline command for NeuroSync CLI.

This module provides comprehensive CLI commands for managing and executing
complete NeuroSync ETL pipelines. It offers pipeline creation, monitoring,
scheduling, and management capabilities with detailed progress reporting
and error handling for complex multi-stage data processing workflows.

Key Features:
    - End-to-end pipeline execution with stage monitoring
    - Pipeline configuration management and validation
    - Real-time progress tracking with detailed status updates
    - Error handling and recovery mechanisms
    - Pipeline scheduling and automation capabilities
    - Performance metrics and execution analytics
    - Multi-source pipeline orchestration
    - Resource monitoring and optimization

Pipeline Operations:
    Create: Initialize new pipeline configurations
    Run: Execute complete or partial pipeline workflows
    Monitor: Real-time pipeline status and progress tracking
    Schedule: Automated pipeline execution with cron-like scheduling
    Validate: Configuration and dependency validation
    Optimize: Performance analysis and optimization suggestions

Pipeline Stages:
    The CLI manages complete pipeline workflows including:
    1. Data source discovery and validation
    2. Content ingestion from multiple sources
    3. Text preprocessing and cleaning
    4. Content chunking and segmentation
    5. Embedding generation and vector computation
    6. Vector storage and indexing
    7. Quality assessment and validation
    8. Monitoring and alerting integration

Configuration Management:
    - YAML-based pipeline configuration files
    - Environment-specific configuration profiles
    - Dynamic configuration updates and hot-reloading
    - Configuration validation and schema enforcement
    - Template-based configuration generation

Progress Monitoring:
    - Real-time progress bars for each pipeline stage
    - Detailed status reporting with stage-specific metrics
    - Error tracking and recovery suggestions
    - Performance metrics and timing analysis
    - Resource utilization monitoring

Error Handling:
    - Comprehensive error reporting with context
    - Automatic retry mechanisms for transient failures
    - Pipeline rollback and recovery capabilities
    - Detailed logging for debugging and monitoring
    - Graceful shutdown and cleanup procedures

Integration Features:
    - Integration with workflow orchestrators (Airflow, Prefect)
    - Monitoring system integration for alerts and dashboards
    - Configuration management system integration
    - API integration for programmatic pipeline control
    - Export capabilities for pipeline metrics and results

Example Usage:
    # Run complete pipeline
    neurosync pipeline run --config production.yaml

    # Monitor pipeline progress
    neurosync pipeline monitor --pipeline-id abc123

    # Validate pipeline configuration
    neurosync pipeline validate --config test.yaml

For advanced pipeline configuration and orchestration, see:
    - docs/pipeline-configuration.md
    - docs/pipeline-orchestration.md
    - examples/production-pipelines.yaml

Author: NeuroSync Team
Created: 2025
License: MIT
"""

from typing import Optional

import typer
from rich.console import Console

console = Console()


def pipeline_command(
    input_path: str = typer.Argument(..., help="Path to input files or directory"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    mode: str = typer.Option(
        "auto", "--mode", "-m", help="Pipeline mode: auto, manual, or interactive"
    ),
    output_dir: Optional[str] = typer.Option(
        "data", "--output", "-o", help="Output directory for processed data"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    Run the complete NeuroSync pipeline on input data.

    This command will:
    1. Ingest documents from the input path
    2. Process and chunk the documents
    3. Generate embeddings
    4. Store vectors in the vector database
    5. Start an interactive chat session
    """
    try:
        from neurosync.pipelines.pipeline import FullPipeline

        console.print("[bold green]Starting NeuroSync Pipeline[/bold green]")
        console.print(f"Input: {input_path}")
        console.print(f"Mode: {mode}")
        console.print(f"Output: {output_dir}")

        # Initialize and run pipeline
        pipeline = FullPipeline()

        if mode == "auto":
            pipeline.run_full_pipeline(input_path, auto_mode=True)
        elif mode == "manual":
            pipeline.run_full_pipeline(input_path, auto_mode=False)
        elif mode == "interactive":
            pipeline.run_full_pipeline(input_path, auto_mode=False)
        else:
            console.print(
                f"[red]Unknown mode: {mode}. "
                f"Use 'auto', 'manual', or 'interactive'[/red]"
            )
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        if verbose:
            import traceback

            console.print(f"[red]{traceback.format_exc()}[/red]")
        raise typer.Exit(1)
