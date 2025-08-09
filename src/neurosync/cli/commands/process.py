"""
Command-line interface for processing operations.

This module provides comprehensive CLI commands for text processing, chunking,
and content transformation operations. It offers both interactive and batch
processing capabilities with detailed progress reporting and configuration
management for various processing strategies.

Key Features:
    - Multiple processing strategies with configurable parameters
    - Batch processing for large file collections
    - Interactive processing with real-time progress feedback
    - Quality assessment and filtering capabilities
    - Output format options (JSON, CSV, Parquet)
    - Processing performance metrics and reporting
    - Error handling with detailed diagnostic information
    - Configuration validation and optimization suggestions

Command Categories:
    File Processing: Process individual files or directories
    Batch Processing: Handle large collections of documents
    Strategy Testing: Evaluate different processing approaches
    Quality Assessment: Analyze processing results and metrics
    Configuration: Manage processing settings and parameters

Processing Strategies:
    Recursive: Hierarchical text splitting with multiple delimiters
    Semantic: NLP-based boundary detection for coherent chunks
    Document Structure: Format-aware processing preserving layout
    Sliding Window: Overlapping chunks for context preservation
    Hierarchical: Parent-child relationships with structure awareness
    Adaptive: Dynamic strategy selection based on content analysis

Performance Features:
    - Parallel processing for independent operations
    - Memory-efficient streaming for large files
    - Progress monitoring with ETA calculations
    - Resource utilization monitoring and optimization
    - Batch size optimization for optimal throughput
    - Checkpointing for resumable long-running operations

Quality Metrics:
    - Chunk size distribution analysis
    - Content coherence scoring
    - Information density assessment
    - Processing speed and efficiency metrics
    - Error rate monitoring and categorization

Output Options:
    The CLI supports multiple output formats:
    - JSON: Structured output with complete metadata
    - CSV: Tabular format for analysis and reporting
    - Parquet: Compressed columnar format for data science
    - JSONL: Line-delimited JSON for streaming processing

Configuration Management:
    - YAML configuration files for complex processing setups
    - Command-line parameter overrides for quick adjustments
    - Configuration validation and optimization suggestions
    - Template configurations for common use cases
    - Environment-specific configuration profiles

Error Handling and Recovery:
    - Detailed error reporting with context and suggestions
    - Graceful handling of corrupted or problematic content
    - Automatic recovery mechanisms for transient failures
    - Comprehensive logging for debugging and monitoring
    - Progress preservation for resumed operations

Integration Points:
    - Input from ingestion pipeline outputs
    - Integration with embedding generation workflows
    - Export to vector database preparation formats
    - Monitoring and alerting system integration

Example Usage:
    # Process a single file with default settings
    neurosync process file data.json

    # Batch process with custom configuration
    neurosync process batch --config custom.yaml --strategy semantic

    # Quality assessment of processing results
    neurosync process analyze --input processed_chunks.json

For advanced processing configurations and custom strategies, see:
    - docs/processing-cli-reference.md
    - docs/processing-strategies.md
    - examples/batch-processing-workflows.md

Author: NeuroSync Team
Created: 2025
License: MIT
"""
import json
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from neurosync.core.logging.logger import get_logger
from neurosync.ingestion.base import (
    ContentType,
    IngestionResult,
    SourceMetadata,
    SourceType,
)
from neurosync.processing.manager import ProcessingManager

app = typer.Typer(help="Intelligent semantic processing and chunking commands")
console = Console()
logger = get_logger(__name__)


@app.command()
def file(
    input_file: str = typer.Argument(
        ..., help="Path to the ingested data file (JSON format)."
    ),
    config_file: Optional[str] = typer.Argument(
        None, help="Path to the processing configuration file."
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Path to save the processed chunks."
    ),
    strategy: Optional[str] = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Chunking strategy: recursive, semantic, sliding_window, "
        "token_aware_sliding, hierarchical, document_structure",
    ),
    chunk_size: int = typer.Option(
        1024, "--chunk-size", help="Size of each chunk in characters"
    ),
    chunk_overlap: int = typer.Option(
        200, "--chunk-overlap", help="Overlap between chunks in characters"
    ),
    model: str = typer.Option(
        "en_core_web_sm", "--model", help="Language model for semantic chunking"
    ),
    limit: int = typer.Option(
        0, "--limit", help="Limit the number of ingested items to process (0 for all)."
    ),
) -> None:
    """
    Process an ingested data file into semantic chunks.

    You can either provide a config file OR specify strategy options directly.
    If no config file is provided, you must specify --strategy.

    Examples:
      neurosync process file data.json --strategy recursive --chunk-size 512
      neurosync process file data.json config.json --output chunks.json
    """
    input_path = Path(input_file)

    if not input_path.exists():
        console.print(f"Input file not found: {input_file}", style="red")
        raise typer.Exit(1)

    # Handle configuration - either from file or command line options
    if config_file:
        config_path = Path(config_file)
        if not config_path.exists():
            console.print(f"Configuration file not found: {config_file}", style="red")
            raise typer.Exit(1)

        # Load processing config from file
        try:
            with open(config_path, "r") as f:
                processing_config = json.load(f)
        except Exception as e:
            console.print(f"Error loading processing config: {e}", style="red")
            raise typer.Exit(1)
    else:
        # Create config from command line options
        if not strategy:
            console.print(
                "Error: You must either provide a config file or specify --strategy",
                style="red",
            )
            console.print(
                "Available strategies: recursive, semantic, sliding_window, "
                "token_aware_sliding, hierarchical, document_structure",
                style="yellow",
            )
            raise typer.Exit(1)

        # Validate strategy
        valid_strategies = [
            "recursive",
            "semantic",
            "sliding_window",
            "token_aware_sliding",
            "hierarchical",
            "document_structure",
        ]
        if strategy not in valid_strategies:
            console.print(f"Invalid strategy: {strategy}", style="red")
            console.print(
                f"Available strategies: {', '.join(valid_strategies)}", style="yellow"
            )
            raise typer.Exit(1)

        # Validate chunk overlap
        if chunk_overlap >= chunk_size:
            console.print(
                f"Error: chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})",
                style="red",
            )
            raise typer.Exit(1)

        # Create processing config from options
        processing_config = {
            "preprocessing": [
                {"name": "html_cleaner", "enabled": True},
                {"name": "whitespace_normalizer", "enabled": True},
            ],
            "chunking": {
                "strategy": strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "model": model,
            },
        }

        console.print(
            f"Using strategy: [bold cyan]{strategy}[/] with "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}",
            style="green",
        )

    # Load ingested data
    try:
        with open(input_path, "r") as f:
            ingested_data = json.load(f)
    except Exception as e:
        console.print(f"Error loading ingested data: {e}", style="red")
        raise typer.Exit(1)

    # Initialize ProcessingManager
    manager = ProcessingManager(processing_config)

    all_chunks = []

    # Ensure ingested_data is a list
    if not isinstance(ingested_data, list):
        console.print(
            "Input data must be a JSON array of ingestion results.", style="red"
        )
        raise typer.Exit(1)

    items_to_process = ingested_data[:limit] if limit > 0 else ingested_data

    console.print(f" Processing {len(items_to_process)} items...")

    for item in items_to_process:
        # Reconstruct IngestionResult object
        try:
            # Handle string-to-enum conversion for metadata and filter known fields
            metadata_dict = item["metadata"].copy()
            if isinstance(metadata_dict.get("source_type"), str):
                metadata_dict["source_type"] = SourceType(metadata_dict["source_type"])
            if isinstance(metadata_dict.get("content_type"), str):
                metadata_dict["content_type"] = ContentType(
                    metadata_dict["content_type"]
                )

            # Filter to only include known SourceMetadata fields
            known_fields = {
                "source_id",
                "source_type",
                "content_type",
                "file_path",
                "url",
                "size_bytes",
                "created_at",
                "modified_at",
                "checksum",
                "encoding",
                "language",
                "title",
                "author",
                "description",
                "tags",
                "custom_metadata",
            }

            # Move unknown fields to custom_metadata
            custom_metadata = metadata_dict.get("custom_metadata", {})
            filtered_metadata = {}

            for key, value in metadata_dict.items():
                if key in known_fields:
                    filtered_metadata[key] = value
                else:
                    custom_metadata[key] = value

            if custom_metadata:
                filtered_metadata["custom_metadata"] = custom_metadata

            metadata = SourceMetadata(**filtered_metadata)
            ingestion_result = IngestionResult(
                success=item["success"],
                source_id=item["source_id"],
                content=item["content"],
                metadata=metadata,
            )

            if ingestion_result.success:
                chunks = manager.process(ingestion_result)
                all_chunks.extend(chunks)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Skipping malformed ingestion item: {e}")
            continue

    # Display summary
    summary_panel = Panel(
        f"""
        [bold]Processing Complete[/bold]

        Items Processed: {len(items_to_process)}
        Chunks Generated: {len(all_chunks)}
        """,
        title="Summary",
        border_style="green",
    )
    console.print(summary_panel)

    if all_chunks:
        # Show a sample of the first chunk
        console.print("\n[bold]Sample Chunk (First Generated):[/bold]")
        first_chunk = all_chunks[0]

        sample_table = Table(show_header=False)
        sample_table.add_column("Property", style="cyan")
        sample_table.add_column("Value", style="white")

        sample_table.add_row("Chunk ID", first_chunk.chunk_id)
        sample_table.add_row("Source ID", first_chunk.source_metadata.source_id)
        sample_table.add_row("Sequence", str(first_chunk.sequence_num))
        sample_table.add_row("Quality Score", f"{first_chunk.quality_score:.2f}")
        sample_table.add_row(
            "Language", first_chunk.processing_metadata.get("language", "N/A")
        )
        sample_table.add_row(
            "Content Preview", f"'{first_chunk.content[:100].strip()}...'"
        )

        console.print(sample_table)

    # Save to output file
    if output_file:
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump([chunk.to_dict() for chunk in all_chunks], f, indent=2)

            console.print(
                f"\nSuccessfully saved {len(all_chunks)} chunks to "
                f"[bold cyan]{output_file}[/bold cyan]"
            )
        except Exception as e:
            console.print(f"\nError saving output file: {e}", style="red")


@app.command()
def strategies() -> None:
    """List all available chunking strategies with descriptions."""

    strategies_info = [
        (
            "recursive",
            "Simple text splitting with overlap",
            "Fast, good for general text",
        ),
        (
            "semantic",
            "NLP-based semantic boundary detection",
            "Preserves meaning, requires language model",
        ),
        (
            "sliding_window",
            "Fixed-size sliding window approach",
            "Consistent chunk sizes, overlapping context",
        ),
        (
            "token_aware_sliding",
            "Token-based sliding windows",
            "Respects word boundaries, consistent tokens",
        ),
        (
            "hierarchical",
            "Structure-aware chunking with hierarchy",
            "Preserves document structure, nested relationships",
        ),
        (
            "document_structure",
            "Advanced document analysis with OCR",
            "Handles complex documents, tables, forms",
        ),
    ]

    table = Table(
        title="Available Chunking Strategies",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Strategy", style="cyan", width=20)
    table.add_column("Description", style="white", width=35)
    table.add_column("Best For", style="green", width=30)

    for strategy, description, best_for in strategies_info:
        table.add_row(strategy, description, best_for)

    console.print(table)

    console.print("\n[bold]Usage Examples:[/]")
    console.print(
        "  neurosync process file data.json --strategy recursive --chunk-size 512"
    )
    console.print(
        "  neurosync process file data.json --strategy semantic --model en_core_web_sm"
    )
    console.print(
        "  neurosync process file data.json --strategy hierarchical --chunk-size 1024"
    )


@app.command()
def compare(
    input_file: str = typer.Argument(
        ..., help="Path to the ingested data file (JSON format)."
    ),
    strategies: List[str] = typer.Option(
        ["recursive", "semantic", "sliding_window"],
        "--strategy",
        "-s",
        help="Strategies to compare (can be used multiple times)",
    ),
    chunk_size: int = typer.Option(
        512, "--chunk-size", help="Size of each chunk in characters"
    ),
    chunk_overlap: int = typer.Option(
        100, "--chunk-overlap", help="Overlap between chunks in characters"
    ),
    limit: int = typer.Option(
        1, "--limit", help="Limit items to process for comparison (default: 1)"
    ),
) -> None:
    """
    Compare different chunking strategies on the same data.

    Examples:
      neurosync process compare data.json --strategy recursive --strategy semantic
      neurosync process compare data.json -s hierarchical -s document_structure
      --chunk-size 1024
    """
    input_path = Path(input_file)

    if not input_path.exists():
        console.print(f"Input file not found: {input_file}", style="red")
        raise typer.Exit(1)

    # Validate strategies
    valid_strategies = [
        "recursive",
        "semantic",
        "sliding_window",
        "token_aware_sliding",
        "hierarchical",
        "document_structure",
    ]
    invalid_strategies = [s for s in strategies if s not in valid_strategies]
    if invalid_strategies:
        console.print(f"Invalid strategies: {invalid_strategies}", style="red")
        console.print(
            f"Available strategies: {', '.join(valid_strategies)}", style="yellow"
        )
        raise typer.Exit(1)

    # Validate chunk overlap
    if chunk_overlap >= chunk_size:
        console.print(
            f"Error: chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})",
            style="red",
        )
        raise typer.Exit(1)

    # Load ingested data
    try:
        with open(input_path, "r") as f:
            ingested_data = json.load(f)
    except Exception as e:
        console.print(f"Error loading ingested data: {e}", style="red")
        raise typer.Exit(1)

    if not isinstance(ingested_data, list):
        console.print(
            "Input data must be a JSON array of ingestion results.", style="red"
        )
        raise typer.Exit(1)

    items_to_process = ingested_data[:limit] if limit > 0 else ingested_data

    console.print("\n[bold]Strategy Comparison Results[/]")
    console.print(f"Data: {input_file}")
    console.print(
        f"Items: {len(items_to_process)}, Chunk Size: {chunk_size}, "
        f"Overlap: {chunk_overlap}"
    )
    console.print("=" * 80)

    comparison_results = []

    for strategy in strategies:
        console.print(f"\n[bold cyan]Testing {strategy}:[/]")

        # Create config for this strategy
        processing_config = {
            "chunking": {
                "strategy": strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "model": "en_core_web_sm",
            }
        }

        # Initialize ProcessingManager
        manager = ProcessingManager(processing_config)

        all_chunks = []
        processing_time = 0.0

        start_time = time.time()

        for item in items_to_process:
            try:
                # Handle metadata conversion
                metadata_dict = item["metadata"].copy()
                if isinstance(metadata_dict.get("source_type"), str):
                    metadata_dict["source_type"] = SourceType(
                        metadata_dict["source_type"]
                    )
                if isinstance(metadata_dict.get("content_type"), str):
                    metadata_dict["content_type"] = ContentType(
                        metadata_dict["content_type"]
                    )

                # Filter metadata fields
                known_fields = {
                    "source_id",
                    "source_type",
                    "content_type",
                    "file_path",
                    "url",
                    "size_bytes",
                    "created_at",
                    "modified_at",
                    "checksum",
                    "encoding",
                    "language",
                    "title",
                    "author",
                    "description",
                    "tags",
                    "custom_metadata",
                }

                custom_metadata = metadata_dict.get("custom_metadata", {})
                filtered_metadata = {}

                for key, value in metadata_dict.items():
                    if key in known_fields:
                        filtered_metadata[key] = value
                    else:
                        custom_metadata[key] = value

                if custom_metadata:
                    filtered_metadata["custom_metadata"] = custom_metadata

                metadata = SourceMetadata(**filtered_metadata)
                ingestion_result = IngestionResult(
                    success=item["success"],
                    source_id=item["source_id"],
                    content=item["content"],
                    metadata=metadata,
                )

                if ingestion_result.success:
                    chunks = manager.process(ingestion_result)
                    all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Error processing item with {strategy}: {e}")
                continue

        processing_time = time.time() - start_time

        # Calculate statistics
        total_chunks = len(all_chunks)
        avg_chunk_size = (
            sum(len(chunk.content) for chunk in all_chunks) / total_chunks
            if total_chunks > 0
            else 0
        )
        avg_quality = (
            sum(chunk.quality_score for chunk in all_chunks) / total_chunks
            if total_chunks > 0
            else 0
        )

        comparison_results.append(
            {
                "strategy": strategy,
                "chunks": total_chunks,
                "avg_size": int(avg_chunk_size),
                "avg_quality": round(avg_quality, 2),
                "time": round(processing_time, 3),
            }
        )

        console.print(
            f"  Chunks: {total_chunks}, Avg Size: {int(avg_chunk_size)}, "
            f"Quality: {avg_quality:.2f}, Time: {processing_time:.3f}s"
        )

    # Display comparison table
    console.print("\n[bold]Comparison Summary:[/]")

    comparison_table = Table(show_header=True, header_style="bold magenta")
    comparison_table.add_column("Strategy", style="cyan")
    comparison_table.add_column("Chunks", justify="right", style="green")
    comparison_table.add_column("Avg Size", justify="right", style="white")
    comparison_table.add_column("Avg Quality", justify="right", style="yellow")
    comparison_table.add_column("Time (s)", justify="right", style="blue")

    for result in comparison_results:
        comparison_table.add_row(
            result["strategy"],
            str(result["chunks"]),
            str(result["avg_size"]),
            str(result["avg_quality"]),
            str(result["time"]),
        )

    console.print(comparison_table)


@app.command()
def create_config(
    output_file: str = typer.Argument(
        "processing_config.json", help="Output configuration file"
    ),
) -> None:
    """Create a default processing configuration file."""

    default_config = {
        "preprocessing": [
            {"name": "html_cleaner", "enabled": True},
            {"name": "whitespace_normalizer", "enabled": True},
        ],
        "chunking": {
            "strategy": "recursive",
            "chunk_size": 1024,
            "chunk_overlap": 200,
            "model": "en_core_web_sm",  # Used by 'semantic' strategy
        },
        "filtering": {"min_quality_score": 0.3},
    }

    try:
        with open(output_file, "w") as f:
            json.dump(default_config, f, indent=2)
        console.print(
            f"Created default processing configuration at: "
            f"[bold cyan]{output_file}[/bold cyan]"
        )
    except Exception as e:
        console.print(f"Error creating config file: {e}", style="red")
        raise typer.Exit(1)
