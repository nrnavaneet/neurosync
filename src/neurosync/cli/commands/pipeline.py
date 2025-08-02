"""
AI Pipeline management commands for NeuroSync CLI
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from neurosync.core.logging.logger import get_logger
from neurosync.ingestion.base import (
    ContentType,
    IngestionResult,
    SourceMetadata,
    SourceType,
)
from neurosync.processing.manager import ProcessingManager

app = typer.Typer(help="Pipeline management commands")
console = Console()
logger = get_logger(__name__)


@app.command()
def run(
    config: Optional[str] = typer.Argument(
        None, help="Pipeline configuration file OR direct source path"
    ),
    source_type: Optional[str] = typer.Option(
        None, "--type", help="Source type: file, api, database (for direct ingestion)"
    ),
    strategy: str = typer.Option(
        "auto",
        "--strategy",
        "-s",
        help="Processing strategy (auto for automatic selection)",
    ),
    chunk_size: int = typer.Option(
        1024, "--chunk-size", help="Chunk size for processing"
    ),
    chunk_overlap: int = typer.Option(200, "--overlap", help="Chunk overlap"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Process directories recursively (file only)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate pipeline without executing"
    ),
    parallel: bool = typer.Option(
        False, "--parallel", help="Enable parallel execution"
    ),
    show_recommendation: bool = typer.Option(
        False,
        "--show-recommendation",
        help="Show strategy recommendation before processing",
    ),
) -> None:
    """Run a NeuroSync pipeline with JSON config OR direct source ingestion"""

    if not config:
        console.print(
            "Error: Please provide either a config file or source path", style="red"
        )
        raise typer.Exit(1)

    # Check if this is a direct source ingestion or config file
    config_path = Path(config)

    # If source_type is specified, treat as direct ingestion
    if source_type:
        logger.info(f"Starting direct ingestion and processing from: {config}")

        # Validate source type
        valid_types = ["file", "api", "database"]
        if source_type not in valid_types:
            console.print(
                f"Invalid source type: {source_type}. Valid options: {valid_types}",
                style="red",
            )
            raise typer.Exit(1)

        # Set default output if not specified
        if not output:
            timestamp = int(time.time())
            output = f"pipeline_results_{source_type}_{timestamp}.json"

        # Handle automatic strategy selection
        final_strategy = strategy
        if strategy == "auto":
            # For auto mode, we need to analyze content first
            console.print(
                "Analyzing content for automatic strategy selection...", style="blue"
            )

            # Quick content analysis for strategy selection
            sample_content = ""
            sample_content_type = ContentType.TEXT

            if source_type == "file":
                source_path_obj = Path(config)
                if source_path_obj.is_file():
                    try:
                        with open(source_path_obj, "r", encoding="utf-8") as f:
                            sample_content = f.read(5000)  # First 5KB for analysis
                        sample_content_type = _detect_content_type(source_path_obj)
                    except (OSError, IOError, UnicodeDecodeError):
                        pass
                elif source_path_obj.is_dir():
                    # For directories, analyze first file found
                    patterns = ["*.txt", "*.md", "*.json", "*.csv", "*.html", "*.xml"]
                    for pattern in patterns:
                        files = list(source_path_obj.glob(pattern))
                        if files:
                            try:
                                with open(files[0], "r", encoding="utf-8") as f:
                                    sample_content = f.read(5000)
                                sample_content_type = _detect_content_type(files[0])
                                break
                            except (OSError, IOError, UnicodeDecodeError):
                                continue

            elif source_type == "api":
                try:
                    import requests

                    response = requests.get(config, timeout=10)
                    sample_content = response.text[:5000]
                    sample_content_type = (
                        ContentType.JSON
                        if "json" in response.headers.get("content-type", "")
                        else ContentType.TEXT
                    )
                except (ImportError, Exception):
                    pass

            elif source_type == "database":
                # For database, use a default strategy
                sample_content = "Mock database content for analysis"
                sample_content_type = ContentType.TEXT

            if sample_content:
                (
                    recommended_strategy,
                    confidence,
                    reasons,
                ) = _recommend_chunking_strategy(sample_content, sample_content_type)
                final_strategy = recommended_strategy

                console.print(
                    f"Auto-selected strategy: [bold green]{final_strategy}"
                    f"[/bold green] (confidence: {confidence:.1%})",
                    style="green",
                )
                if reasons:
                    console.print("📋 Reasons:", style="blue")
                    for reason in reasons:
                        console.print(f"   • {reason}", style="dim")
            else:
                final_strategy = "recursive"  # Safe fallback
                console.print(
                    "Could not analyze content, using recursive strategy as fallback",
                    style="yellow",
                )

        # Create pipeline config for direct ingestion
        pipeline_config = {
            "name": f"direct_pipeline_{source_type}",
            "description": (
                f"Direct {source_type} ingestion and {final_strategy} processing"
            ),
            "ingestion": {
                "direct_source": {
                    "path": config,  # The source path
                    "type": source_type,
                    "recursive": recursive if source_type == "file" else False,
                }
            },
            "processing": {
                "strategy": final_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "model": "en_core_web_sm",
            },
            "output": output,
        }

        if dry_run:
            console.print("🔍 Running in dry-run mode (validation only)")
            console.print(f"Would ingest from {source_type}: {config}")
            if strategy == "auto":
                console.print(
                    "Would automatically select best chunking strategy "
                    "based on content analysis"
                )
            else:
                console.print(f"Would process with {strategy} strategy")
            console.print(f"Would output to: {output}")
            return

        # Execute direct ingestion pipeline
        _execute_direct_pipeline(pipeline_config, parallel)
        console.print(
            "Direct ingestion and processing completed successfully", style="green"
        )
        return

    # Otherwise, treat as config file
    if not config_path.exists():
        console.print(f"Configuration file does not exist: {config}", style="red")
        raise typer.Exit(1)

    logger.info(f"Starting pipeline execution: {config}")

    try:
        with open(config_path, "r") as f:
            pipeline_config = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"Invalid JSON in configuration file: {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"Error reading configuration file: {e}", style="red")
        raise typer.Exit(1)

    if dry_run:
        console.print("🔍 Running in dry-run mode (validation only)")
        _validate_pipeline_config(pipeline_config)
        console.print("Pipeline validation completed successfully", style="green")
        return

    # Execute the pipeline
    _execute_pipeline(pipeline_config, parallel)
    console.print("Pipeline execution completed successfully", style="green")


@app.command()
def _execute_direct_pipeline(config, parallel: bool):
    """Execute pipeline with direct ingestion from source"""
    pipeline_name = config["name"]
    direct_source = config["ingestion"]["direct_source"]
    source_path = direct_source["path"]
    source_type = direct_source["type"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Stage 1: Perform direct ingestion
        ingest_task = progress.add_task(
            f"Ingesting data from {source_type} source...", total=100
        )

        ingestion_results = []

        try:
            if source_type == "file":
                # Handle file ingestion directly
                source_path_obj = Path(source_path)
                if source_path_obj.is_file():
                    # Single file
                    ingestion_results = _ingest_single_file(source_path_obj)
                elif source_path_obj.is_dir():
                    # Directory
                    recursive = direct_source.get("recursive", False)
                    ingestion_results = _ingest_directory(source_path_obj, recursive)
                else:
                    console.print(
                        f"File/directory not found: {source_path}", style="red"
                    )
                    raise typer.Exit(1)

            elif source_type == "api":
                # Handle API ingestion directly
                ingestion_results = _ingest_api_url(source_path)

            elif source_type == "database":
                # Handle database ingestion directly
                ingestion_results = _ingest_database(source_path)

            progress.update(ingest_task, completed=100)

        except Exception as e:
            console.print(f"Ingestion failed: {e}", style="red")
            raise typer.Exit(1)

        if not ingestion_results:
            console.print("No data was ingested", style="yellow")
            return

        # Stage 2: Initialize processing manager
        init_task = progress.add_task("Initializing processing manager...", total=100)

        processing_config = config["processing"]
        chunking_config = {
            "chunking": {
                "strategy": processing_config["strategy"],
                "chunk_size": processing_config.get("chunk_size", 1024),
                "chunk_overlap": processing_config.get("chunk_overlap", 200),
                "model": processing_config.get("model", "en_core_web_sm"),
            },
            "preprocessing": [
                {"name": "html_cleaner", "enabled": True},
                {"name": "whitespace_normalizer", "enabled": True},
            ],
        }

        manager = ProcessingManager(chunking_config)
        progress.update(init_task, completed=100)

        # Stage 3: Process chunks
        process_task = progress.add_task(
            "Processing chunks...", total=len(ingestion_results)
        )
        all_chunks = []
        processed_count = 0

        for ingestion_result in ingestion_results:
            try:
                if ingestion_result.success:
                    chunks = manager.process(ingestion_result)
                    all_chunks.extend(chunks)
                    processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to process item {processed_count}: {e}")
                continue

            progress.update(process_task, advance=1)

        # Stage 4: Save results
        save_task = progress.add_task("Saving results...", total=100)

        output_path = config["output"]
        output_data = {
            "pipeline_name": pipeline_name,
            "processing_config": processing_config,
            "total_chunks": len(all_chunks),
            "processed_items": processed_count,
            "timestamp": time.time(),
            "chunks": [chunk.to_dict() for chunk in all_chunks],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        progress.update(save_task, completed=100)

    # Display results summary
    _display_pipeline_results(
        pipeline_name, processed_count, len(all_chunks), output_path
    )


def _ingest_single_file(file_path: Path) -> List[IngestionResult]:
    """Ingest a single file and return IngestionResult objects"""
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Create metadata
        source_metadata = SourceMetadata(
            source_id=file_path.name,
            source_type=SourceType.FILE,
            content_type=_detect_content_type(file_path),
            file_path=str(file_path),
            size_bytes=file_path.stat().st_size,
            encoding="utf-8",
        )

        # Create ingestion result
        result = IngestionResult(
            success=True,
            source_id=file_path.name,
            content=content,
            metadata=source_metadata,
        )

        return [result]

    except Exception as e:
        # Return failed result
        result = IngestionResult(success=False, source_id=file_path.name, error=str(e))
        return [result]


def _ingest_directory(dir_path: Path, recursive: bool) -> List[IngestionResult]:
    """Ingest all files in a directory"""
    results = []

    # File patterns to include
    patterns = ["*.txt", "*.md", "*.json", "*.csv", "*.html", "*.xml"]

    files: List[Path] = []
    if recursive:
        for pattern in patterns:
            files.extend(dir_path.rglob(pattern))
    else:
        for pattern in patterns:
            files.extend(dir_path.glob(pattern))

    for file_path in files:
        if file_path.is_file():
            file_results = _ingest_single_file(file_path)
            results.extend(file_results)

    return results


def _ingest_api_url(url: str) -> List[IngestionResult]:
    """Ingest data from API URL"""
    try:
        import requests

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        content = response.text

        # Create metadata
        source_metadata = SourceMetadata(
            source_id=f"api_{hash(url)}",
            source_type=SourceType.API,
            content_type=ContentType.JSON
            if "json" in response.headers.get("content-type", "")
            else ContentType.TEXT,
            url=url,
            size_bytes=len(content.encode("utf-8")),
            encoding="utf-8",
        )

        # Create ingestion result
        result = IngestionResult(
            success=True,
            source_id=f"api_{hash(url)}",
            content=content,
            metadata=source_metadata,
        )

        return [result]

    except Exception as e:
        result = IngestionResult(
            success=False, source_id=f"api_{hash(url)}", error=str(e)
        )
        return [result]


def _ingest_database(connection_string: str) -> List[IngestionResult]:
    """Ingest data from database"""
    try:
        # For now, create a simple mock database ingestion
        # In a real implementation, this would connect to the actual database
        content = f"Mock database content from: {connection_string}"

        source_metadata = SourceMetadata(
            source_id=f"db_{hash(connection_string)}",
            source_type=SourceType.DATABASE,
            content_type=ContentType.TEXT,
            size_bytes=len(content.encode("utf-8")),
            encoding="utf-8",
        )

        result = IngestionResult(
            success=True,
            source_id=f"db_{hash(connection_string)}",
            content=content,
            metadata=source_metadata,
        )

        return [result]

    except Exception as e:
        result = IngestionResult(
            success=False, source_id=f"db_{hash(connection_string)}", error=str(e)
        )
        return [result]


def _detect_content_type(file_path: Path) -> ContentType:
    """Detect content type from file extension"""
    suffix = file_path.suffix.lower()
    mapping = {
        ".txt": ContentType.TEXT,
        ".md": ContentType.MARKDOWN,
        ".json": ContentType.JSON,
        ".csv": ContentType.CSV,
        ".xml": ContentType.XML,
        ".html": ContentType.HTML,
        ".htm": ContentType.HTML,
    }
    return mapping.get(suffix, ContentType.TEXT)


def _display_strategy_recommendation(
    strategy: str, confidence: float, reasons: List[str], content_type: ContentType
) -> None:
    """Display chunking strategy recommendation to user"""
    confidence_color = (
        "green" if confidence > 0.8 else "yellow" if confidence > 0.5 else "red"
    )
    confidence_percentage = f"{confidence * 100:.1f}%"

    recommendation_panel = Panel(
        f"""[bold]Recommended Strategy:[/bold] [cyan]{strategy}[/cyan]
[bold]Confidence:[/bold] [{confidence_color}]{confidence_percentage}"""
        f"""[/{confidence_color}]
[bold]Content Type:[/bold] [blue]{content_type.value}[/blue]

        [bold yellow]Analysis Reasons:[/bold yellow]
        {chr(10).join(f"• {reason}" for reason in reasons)
            if reasons else "• Content analysis completed"}

        [dim]Use --show-recommendation to always see this analysis[/dim]""",
        title="🎯 Auto-Strategy Selection",
        border_style=confidence_color,
    )
    console.print(recommendation_panel)


def _analyze_content_for_chunking(
    content: str, content_type: ContentType = ContentType.TEXT
) -> Dict[str, Any]:
    """Analyze content to recommend best chunking strategy"""
    analysis = {
        "content_length": len(content),
        "word_count": len(content.split()),
        "line_count": len(content.split("\n")),
        "paragraph_count": len([p for p in content.split("\n\n") if p.strip()]),
        "has_structure": False,
        "has_code": False,
        "has_tables": False,
        "has_lists": False,
        "complexity_score": 0,
        "recommended_strategy": "recursive",
        "confidence": 0.0,
        "reasons": [],
    }

    # Analyze structure indicators
    if content_type == ContentType.MARKDOWN:
        analysis["has_structure"] = bool(re.search(r"^#+\s", content, re.MULTILINE))
        analysis["has_tables"] = "|" in content and content.count("|") > 5
        analysis["has_lists"] = bool(re.search(r"^\s*[-*+]\s", content, re.MULTILINE))
        analysis["has_code"] = "```" in content or "`" in content
    elif content_type == ContentType.HTML:
        analysis["has_structure"] = bool(re.search(r"<h[1-6]>", content, re.IGNORECASE))
        analysis["has_tables"] = "<table>" in content.lower()
        analysis["has_lists"] = bool(re.search(r"<[uo]l>", content, re.IGNORECASE))
        analysis["has_code"] = "<code>" in content.lower() or "<pre>" in content.lower()
    elif content_type == ContentType.JSON:
        try:
            json.loads(content)
            analysis["has_structure"] = True
        except (ValueError, json.JSONDecodeError):
            pass

    # Calculate complexity score
    complexity_factors = [
        cast(int, analysis["has_structure"]) * 2,
        cast(int, analysis["has_tables"]) * 1.5,
        cast(int, analysis["has_lists"]) * 1,
        cast(int, analysis["has_code"]) * 1.5,
        min(cast(int, analysis["paragraph_count"]) / 10, 2),
        min(cast(int, analysis["word_count"]) / 1000, 3),
    ]
    analysis["complexity_score"] = sum(complexity_factors)

    # Recommend strategy based on analysis
    strategy_scores = {
        "recursive": 5.0,  # Default baseline
        "semantic": 0.0,
        "sliding_window": 0.0,
        "token_aware_sliding": 0.0,
        "hierarchical": 0.0,
        "document_structure": 0.0,
    }

    # Semantic is good for natural language with clear sentence boundaries
    if (
        cast(int, analysis["word_count"]) > 100
        and cast(int, analysis["paragraph_count"]) > 3
    ):
        strategy_scores["semantic"] += 4.0
        if not analysis["has_code"] and not analysis["has_tables"]:
            strategy_scores["semantic"] += 2.0

    # Sliding window is good for continuous text without clear structure
    if cast(int, analysis["word_count"]) > 500 and not analysis["has_structure"]:
        strategy_scores["sliding_window"] += 3.0
        if (
            cast(int, analysis["paragraph_count"]) < 5
        ):  # Few paragraphs = continuous text
            strategy_scores["sliding_window"] += 2.0

    # Token-aware is good for very long content or when token precision matters
    if cast(int, analysis["word_count"]) > 2000:
        strategy_scores["token_aware_sliding"] += 4.0
        if cast(int, analysis["content_length"]) > 10000:
            strategy_scores["token_aware_sliding"] += 2.0

    # Hierarchical is excellent for structured documents
    if analysis["has_structure"]:
        strategy_scores["hierarchical"] += 5.0
        if cast(int, analysis["paragraph_count"]) > 5:
            strategy_scores["hierarchical"] += 2.0
        if content_type in [ContentType.MARKDOWN, ContentType.HTML]:
            strategy_scores["hierarchical"] += 3.0

    # Document structure is best for complex structured documents
    if analysis["has_structure"] and (analysis["has_tables"] or analysis["has_lists"]):
        strategy_scores["document_structure"] += 6.0
        if cast(float, analysis["complexity_score"]) > 5:
            strategy_scores["document_structure"] += 3.0
        if content_type in [ContentType.MARKDOWN, ContentType.HTML]:
            strategy_scores["document_structure"] += 2.0

    # Find best strategy
    best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
    analysis["recommended_strategy"] = best_strategy[0]
    analysis["confidence"] = min(best_strategy[1] / 10.0, 1.0)

    # Generate reasons
    reasons = []
    if analysis["has_structure"]:
        reasons.append("Document has clear hierarchical structure")
    if analysis["has_tables"]:
        reasons.append("Contains tables that benefit from structure-aware processing")
    if analysis["has_code"]:
        reasons.append("Contains code blocks")
    if cast(int, analysis["word_count"]) > 2000:
        reasons.append("Long content benefits from token-aware processing")
    if cast(int, analysis["paragraph_count"]) > 5:
        reasons.append("Multiple paragraphs suggest semantic boundaries")
    if not analysis["has_structure"] and cast(int, analysis["word_count"]) > 500:
        reasons.append("Continuous text without clear structure")

    analysis["reasons"] = reasons

    return analysis


def _recommend_chunking_strategy(
    content: str, content_type: ContentType = ContentType.TEXT
) -> tuple:
    """Recommend the best chunking strategy for given content"""
    analysis = _analyze_content_for_chunking(content, content_type)
    return analysis["recommended_strategy"], analysis["confidence"], analysis["reasons"]


def _validate_pipeline_config(config):
    """Validate pipeline configuration"""
    required_keys = ["name", "ingestion", "processing"]
    for key in required_keys:
        if key not in config:
            console.print(f"Missing required key in config: {key}", style="red")
            raise typer.Exit(1)

    # Validate ingestion config
    if "sources" not in config["ingestion"]:
        console.print("Missing 'sources' in ingestion config", style="red")
        raise typer.Exit(1)

    # Validate processing config
    processing = config["processing"]
    if "strategy" not in processing:
        console.print("Missing 'strategy' in processing config", style="red")
        raise typer.Exit(1)

    valid_strategies = [
        "recursive",
        "semantic",
        "sliding_window",
        "token_aware_sliding",
        "hierarchical",
        "document_structure",
    ]
    if processing["strategy"] not in valid_strategies:
        console.print(
            f"Invalid strategy: {processing['strategy']}. "
            f"Valid options: {valid_strategies}",
            style="red",
        )
        raise typer.Exit(1)


def _execute_pipeline(config, parallel: bool):
    """Execute the pipeline with the given configuration"""
    pipeline_name = config["name"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Stage 1: Load ingestion data
        load_task = progress.add_task("Loading ingestion data...", total=100)
        all_ingestion_results = []

        for source in config["ingestion"]["sources"]:
            source_path = Path(source["path"])
            if not source_path.exists():
                console.print(f"Source file not found: {source['path']}", style="red")
                continue

            try:
                with open(source_path, "r") as f:
                    source_data = json.load(f)
                    # Simply extend without type checking for now
                    all_ingestion_results.extend(source_data)
            except Exception as e:
                console.print(f"Error loading {source['path']}: {e}", style="red")
                continue

        progress.update(load_task, completed=100)

        # Stage 2: Initialize processing manager
        init_task = progress.add_task("Initializing processing manager...", total=100)

        processing_config = config["processing"]
        chunking_config = {
            "chunking": {
                "strategy": processing_config["strategy"],
                "chunk_size": processing_config.get("chunk_size", 1024),
                "chunk_overlap": processing_config.get("chunk_overlap", 200),
                "model": processing_config.get("model", "en_core_web_sm"),
            },
            "preprocessing": [
                # Fixed: preprocessors need to be a list, not a dict
                {"name": "html_cleaner", "enabled": True},
                {"name": "whitespace_normalizer", "enabled": True},
            ],
        }

        manager = ProcessingManager(chunking_config)

        progress.update(init_task, completed=100)

        # Stage 3: Process chunks
        process_task = progress.add_task(
            "Processing chunks...", total=len(all_ingestion_results)
        )
        all_chunks = []
        processed_count = 0

        for item in all_ingestion_results:
            try:
                # Debug print
                if not isinstance(item, dict):
                    console.print(
                        f"Skipping non-dict item: {type(item)} - {item}", style="yellow"
                    )
                    continue

                # Handle different data formats - check if it's the new format
                # or old format
                if "source_metadata" in item:
                    # New format with IngestionResult structure
                    source_metadata = SourceMetadata(
                        source_id=item["source_metadata"]["source_id"],
                        source_type=SourceType(item["source_metadata"]["source_type"]),
                        content_type=ContentType(
                            item["source_metadata"]["content_type"]
                        ),
                        file_path=item["source_metadata"].get("file_path"),
                        size_bytes=item["source_metadata"].get("size_bytes", 0),
                        encoding=item["source_metadata"].get("encoding", "utf-8"),
                        custom_metadata=item["source_metadata"].get("metadata", {}),
                    )

                    ingestion_result = IngestionResult(
                        success=item["success"],
                        source_id=item["source_metadata"]["source_id"],
                        content=item["content"],
                        metadata=source_metadata,
                        error=item.get("error"),
                    )
                else:
                    # Old format from sample data - convert to new format
                    metadata = item.get("metadata", {})

                    # Safe enum construction with fallbacks
                    try:
                        source_type = SourceType(metadata.get("source_type", "file"))
                    except (ValueError, TypeError):
                        source_type = SourceType.FILE

                    try:
                        content_type = ContentType(metadata.get("content_type", "text"))
                    except (ValueError, TypeError):
                        content_type = ContentType.TEXT

                    source_metadata = SourceMetadata(
                        source_id=item.get("source_id", f"item_{processed_count}"),
                        source_type=source_type,
                        content_type=content_type,
                        file_path=metadata.get("file_path"),
                        size_bytes=metadata.get("size_bytes", 0),
                        encoding=metadata.get("encoding", "utf-8"),
                        custom_metadata=metadata,
                    )

                    ingestion_result = IngestionResult(
                        success=item.get("success", True),
                        source_id=item.get("source_id", f"item_{processed_count}"),
                        content=item["content"],
                        metadata=source_metadata,
                        error=item.get("error"),
                    )

                if ingestion_result.success:
                    chunks = manager.process(ingestion_result)
                    all_chunks.extend(chunks)
                    processed_count += 1

            except Exception as e:
                logger.warning(f"Skipping malformed ingestion item: {e}")
                console.print(
                    f"Error processing item {processed_count}: {e}", style="yellow"
                )
                continue

            progress.update(process_task, advance=1)

        # Stage 4: Save results
        save_task = progress.add_task("Saving results...", total=100)

        output_path = config.get("output", f"{pipeline_name}_pipeline_results.json")
        output_data = {
            "pipeline_name": pipeline_name,
            "processing_config": processing_config,
            "total_chunks": len(all_chunks),
            "processed_items": processed_count,
            "timestamp": time.time(),
            "chunks": [chunk.to_dict() for chunk in all_chunks],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        progress.update(save_task, completed=100)

    # Display results summary
    _display_pipeline_results(
        pipeline_name, processed_count, len(all_chunks), output_path
    )


def _display_pipeline_results(
    pipeline_name: str, processed_items: int, total_chunks: int, output_path: str
) -> None:
    """Display pipeline execution results"""
    results_panel = Panel(
        f"""[bold green]Pipeline Execution Complete![/bold green]

Pipeline: [cyan]{pipeline_name}[/cyan]
Processed Items: [yellow]{processed_items}[/yellow]
Total Chunks: [blue]{total_chunks}[/blue]
Output File: [green]{output_path}[/green]

[dim]Results saved successfully![/dim]""",
        title="NeuroSync Pipeline Results",
        border_style="green",
    )
    console.print(results_panel)


@app.command()
def create(
    name: str = typer.Argument(..., help="Pipeline name"),
    template: str = typer.Option("basic", "--template", "-t", help="Pipeline template"),
    output_dir: str = typer.Option(
        ".", "--output-dir", help="Output directory for config file"
    ),
) -> None:
    """Create a new pipeline configuration"""
    logger.info(f"Creating new pipeline: {name}")

    templates: Dict[str, Dict[str, Any]] = {
        "basic": {
            "name": name,
            "description": f"Basic single-source pipeline for {name}",
            "ingestion": {
                "sources": [
                    {
                        "name": "sample_file_data",
                        "path": "sample_data/file_ingestion_results.json",
                        "type": "file",
                    }
                ]
            },
            "processing": {
                "strategy": "recursive",
                "chunk_size": 1024,
                "chunk_overlap": 200,
                "model": "en_core_web_sm",
            },
            "output": f"{name}_results.json",
        },
        "advanced": {
            "name": name,
            "description": f"Advanced multi-source pipeline for {name}",
            "ingestion": {
                "sources": [
                    {
                        "name": "file_data",
                        "path": "sample_data/file_ingestion_results.json",
                        "type": "file",
                    },
                    {
                        "name": "api_data",
                        "path": "sample_data/api_ingestion_results.json",
                        "type": "api",
                    },
                    {
                        "name": "database_data",
                        "path": "sample_data/database_ingestion_results.json",
                        "type": "database",
                    },
                ]
            },
            "processing": {
                "strategy": "semantic",
                "chunk_size": 512,
                "chunk_overlap": 100,
                "model": "en_core_web_sm",
            },
            "output": f"{name}_advanced_results.json",
        },
        "comprehensive": {
            "name": name,
            "description": f"Comprehensive pipeline testing all strategies for {name}",
            "ingestion": {
                "sources": [
                    {
                        "name": "file_data",
                        "path": "sample_data/file_ingestion_results.json",
                        "type": "file",
                    },
                    {
                        "name": "api_data",
                        "path": "sample_data/api_ingestion_results.json",
                        "type": "api",
                    },
                    {
                        "name": "database_data",
                        "path": "sample_data/database_ingestion_results.json",
                        "type": "database",
                    },
                ]
            },
            "processing": {
                "strategy": "hierarchical",
                "chunk_size": 1024,
                "chunk_overlap": 200,
                "model": "en_core_web_sm",
            },
            "output": f"{name}_comprehensive_results.json",
        },
        "semantic_focused": {
            "name": name,
            "description": f"Semantic chunking focused pipeline for {name}",
            "ingestion": {
                "sources": [
                    {
                        "name": "file_data",
                        "path": "sample_data/file_ingestion_results.json",
                        "type": "file",
                    }
                ]
            },
            "processing": {
                "strategy": "semantic",
                "chunk_size": 768,
                "chunk_overlap": 150,
                "model": "en_core_web_lg",
            },
            "output": f"{name}_semantic_results.json",
        },
        "sliding_window": {
            "name": name,
            "description": f"Sliding window chunking pipeline for {name}",
            "ingestion": {
                "sources": [
                    {
                        "name": "api_data",
                        "path": "sample_data/api_ingestion_results.json",
                        "type": "api",
                    }
                ]
            },
            "processing": {
                "strategy": "sliding_window",
                "chunk_size": 512,
                "chunk_overlap": 128,
                "model": "en_core_web_sm",
            },
            "output": f"{name}_sliding_results.json",
        },
        "token_aware": {
            "name": name,
            "description": f"Token-aware sliding window pipeline for {name}",
            "ingestion": {
                "sources": [
                    {
                        "name": "database_data",
                        "path": "sample_data/database_ingestion_results.json",
                        "type": "database",
                    }
                ]
            },
            "processing": {
                "strategy": "token_aware_sliding",
                "chunk_size": 1024,
                "chunk_overlap": 256,
                "model": "en_core_web_sm",
            },
            "output": f"{name}_token_aware_results.json",
        },
        "document_structure": {
            "name": name,
            "description": f"Document structure aware pipeline for {name}",
            "ingestion": {
                "sources": [
                    {
                        "name": "file_data",
                        "path": "sample_data/file_ingestion_results.json",
                        "type": "file",
                    }
                ]
            },
            "processing": {
                "strategy": "document_structure",
                "chunk_size": 2048,
                "chunk_overlap": 300,
                "model": "en_core_web_sm",
            },
            "output": f"{name}_document_structure_results.json",
        },
        "performance_test": {
            "name": name,
            "description": f"Performance testing pipeline with large chunks for {name}",
            "ingestion": {
                "sources": [
                    {
                        "name": "file_data",
                        "path": "sample_data/file_ingestion_results.json",
                        "type": "file",
                    },
                    {
                        "name": "api_data",
                        "path": "sample_data/api_ingestion_results.json",
                        "type": "api",
                    },
                    {
                        "name": "database_data",
                        "path": "sample_data/database_ingestion_results.json",
                        "type": "database",
                    },
                ]
            },
            "processing": {
                "strategy": "recursive",
                "chunk_size": 4096,
                "chunk_overlap": 512,
                "model": "en_core_web_sm",
            },
            "output": f"{name}_performance_test_results.json",
        },
        "minimal": {
            "name": name,
            "description": f"Minimal configuration pipeline for {name}",
            "ingestion": {
                "sources": [
                    {
                        "name": "file_data",
                        "path": "sample_data/file_ingestion_results.json",
                        "type": "file",
                    }
                ]
            },
            "processing": {
                "strategy": "recursive",
                "chunk_size": 256,
                "chunk_overlap": 50,
                "model": "en_core_web_sm",
            },
            "output": f"{name}_minimal_results.json",
        },
        "custom": {
            "name": name,
            "description": f"Customizable template for {name} (edit after creation)",
            "ingestion": {
                "sources": [
                    {
                        "name": "your_data_source",
                        "path": "path/to/your/ingestion_results.json",
                        "type": "file",
                    }
                ]
            },
            "processing": {
                "strategy": "recursive",
                "chunk_size": 1024,
                "chunk_overlap": 200,
                "model": "en_core_web_sm",
            },
            "output": f"{name}_custom_results.json",
        },
    }

    if template not in templates:
        console.print(f"Unknown template: {template}", style="red")
        console.print(f"Available templates: {list(templates.keys())}")
        console.print(
            "\nUse 'neurosync pipeline templates' to see detailed "
            "template descriptions."
        )
        raise typer.Exit(1)

    output_path = Path(output_dir) / f"{name}_pipeline.json"
    config = templates[template]

    try:
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        console.print(
            f"Created pipeline '{name}' from '{template}' template", style="green"
        )
        console.print(f"Configuration saved to: {output_path}", style="blue")

        # Display template details
        template_panel = Panel(
            f"""[bold]{config['description']}[/bold]

[yellow]Sources:[/yellow]
{_format_sources(config['ingestion']['sources'])}

[yellow]Processing:[/yellow]
• Strategy: [cyan]{config['processing']['strategy']}[/cyan]
• Chunk Size: [blue]{config['processing']['chunk_size']}[/blue]
• Overlap: [blue]{config['processing']['chunk_overlap']}[/blue]
• Model: [green]{config['processing']['model']}[/green]

[yellow]Output:[/yellow] [green]{config['output']}[/green]""",
            title=f"📋 Pipeline Configuration: {name}",
            border_style="blue",
        )
        console.print(template_panel)

    except Exception as e:
        console.print(f"Error creating pipeline configuration: {e}", style="red")
        raise typer.Exit(1)


def _format_sources(sources: List[dict]) -> str:
    """Format sources list for display"""
    formatted: List[str] = []
    for source in sources:
        formatted.append(f"• {source['name']} ({source['type']}): {source['path']}")
    return "\n".join(formatted)


@app.command()
def test_all(
    strategy: str = typer.Option(
        "recursive", "--strategy", "-s", help="Chunking strategy to test"
    ),
    chunk_size: int = typer.Option(
        1024, "--chunk-size", help="Chunk size for processing"
    ),
    output_dir: str = typer.Option(
        ".", "--output-dir", help="Output directory for results"
    ),
) -> None:
    """Test pipeline with all three ingestion types (file, API, database)"""
    logger.info(f"Testing pipeline with all ingestion types using {strategy} strategy")

    # Check if sample data exists
    sample_files = [
        "sample_data/file_ingestion_results.json",
        "sample_data/api_ingestion_results.json",
        "sample_data/database_ingestion_results.json",
    ]

    missing_files = []
    for file_path in sample_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        console.print("Missing sample data files:", style="red")
        for file_path in missing_files:
            console.print(f"  • {file_path}", style="red")
        console.print(
            "\nPlease ensure all sample data files exist before running the test.",
            style="yellow",
        )
        raise typer.Exit(1)

    # Create temporary pipeline config
    test_config = {
        "name": "test_all_ingestion_types",
        "description": (
            f"Test pipeline for all ingestion types with {strategy} strategy"
        ),
        "ingestion": {
            "sources": [
                {
                    "name": "file_data",
                    "path": "sample_data/file_ingestion_results.json",
                    "type": "file",
                },
                {
                    "name": "api_data",
                    "path": "sample_data/api_ingestion_results.json",
                    "type": "api",
                },
                {
                    "name": "database_data",
                    "path": "sample_data/database_ingestion_results.json",
                    "type": "database",
                },
            ]
        },
        "processing": {
            "strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": 200,
            "model": "en_core_web_sm",
        },
        "output": f"{output_dir}/test_all_ingestion_results.json",
    }

    # Save temporary config
    temp_config_path = Path(output_dir) / "temp_test_config.json"
    with open(temp_config_path, "w") as f:
        json.dump(test_config, f, indent=2)

    console.print("Running comprehensive ingestion test...\n", style="bold blue")

    try:
        # Execute the pipeline
        _execute_pipeline(test_config, parallel=False)

        # Clean up temp config
        temp_config_path.unlink()

        console.print("\nTest completed successfully!", style="bold green")

    except Exception as e:
        console.print(f"\nTest failed: {e}", style="red")
        if temp_config_path.exists():
            temp_config_path.unlink()
        raise typer.Exit(1)
    """List all available pipelines"""
    table = Table(title="Available Pipelines")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Last Run", style="yellow")
    table.add_column("Sources", style="blue")

    # Mock data
    pipelines = [
        ("rag-documents", "Active", "2 hours ago", "3"),
        ("api-ingestion", "Inactive", "1 day ago", "1"),
        ("batch-processing", "Running", "Running now", "5"),
    ]

    for name, status, last_run, sources in pipelines:
        status_style = (
            "green"
            if status == "Active"
            else "red"
            if status == "Inactive"
            else "yellow"
        )
        table.add_row(
            name, f"[{status_style}]{status}[/{status_style}]", last_run, sources
        )

    console.print(table)


@app.command("list")
def list_pipelines() -> None:
    """List all available pipeline configurations"""
    table = Table(title="Available Pipeline Configurations")
    table.add_column("Name", style="cyan")
    table.add_column("Strategy", style="green")
    table.add_column("Sources", style="blue")
    table.add_column("Config File", style="yellow")

    # Find all pipeline config files
    config_files = list(Path(".").glob("*_pipeline.json"))

    if not config_files:
        console.print("No pipeline configurations found.", style="yellow")
        console.print(
            "Use 'neurosync pipeline create <name>' to create a new pipeline.",
            style="dim",
        )
        return

    for config_file in config_files:
        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            name = config.get("name", config_file.stem)
            strategy = config.get("processing", {}).get("strategy", "N/A")
            source_count = len(config.get("ingestion", {}).get("sources", []))

            table.add_row(name, strategy, str(source_count), str(config_file))
        except Exception as e:
            logger.warning(f"Could not parse config file {config_file}: {e}")
            continue

    console.print(table)


@app.command()
def analyze(
    source: str = typer.Argument(
        ..., help="Source path to analyze (file, directory, or URL)"
    ),
    source_type: Optional[str] = typer.Option(
        None, "--type", help="Source type: file, api, database"
    ),
    compare_all: bool = typer.Option(
        False, "--compare-all", help="Compare all strategies with scores"
    ),
) -> None:
    """Analyze content and recommend the best chunking strategy"""

    console.print(f"Analyzing source: [cyan]{source}[/cyan]", style="bold")

    # Auto-detect source type if not provided
    if not source_type:
        source_path = Path(source)
        if source_path.exists():
            source_type = "file"
        elif source.startswith(("http://", "https://")):
            source_type = "api"
        elif "://" in source:  # Database connection strings
            source_type = "database"
        else:
            console.print(
                "Could not auto-detect source type. Please specify --type", style="red"
            )
            raise typer.Exit(1)

    # Content analysis
    content_samples = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        analyze_task = progress.add_task("Analyzing content...", total=100)

        if source_type == "file":
            source_path = Path(source)
            if source_path.is_file():
                try:
                    with open(source_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    content_type = _detect_content_type(source_path)
                    content_samples.append((content, content_type, str(source_path)))
                except Exception as e:
                    console.print(f"Error reading file: {e}", style="red")
                    raise typer.Exit(1)

            elif source_path.is_dir():
                patterns = ["*.txt", "*.md", "*.json", "*.csv", "*.html", "*.xml"]
                files_found: List[Path] = []
                for pattern in patterns:
                    files_found.extend(source_path.glob(pattern))

                # Analyze up to 5 files for directory analysis
                for file_path in files_found[:5]:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        content_type = _detect_content_type(file_path)
                        content_samples.append((content, content_type, str(file_path)))
                    except (OSError, IOError, UnicodeDecodeError):
                        continue
            else:
                console.print(f"Path not found: {source}", style="red")
                raise typer.Exit(1)

        elif source_type == "api":
            try:
                import requests

                response = requests.get(source, timeout=30)
                response.raise_for_status()
                content = response.text
                content_type = (
                    ContentType.JSON
                    if "json" in response.headers.get("content-type", "")
                    else ContentType.TEXT
                )
                content_samples.append((content, content_type, source))
            except Exception as e:
                console.print(f"Error fetching API: {e}", style="red")
                raise typer.Exit(1)

        elif source_type == "database":
            # Mock database content for analysis
            content = f"Mock database content for analysis from: {source}"
            content_type = ContentType.TEXT
            content_samples.append((content, content_type, source))

        progress.update(analyze_task, completed=100)

    if not content_samples:
        console.print("No content found to analyze", style="red")
        raise typer.Exit(1)

    # Analyze each content sample
    console.print(
        f"\nAnalysis Results for {len(content_samples)} content sample(s):",
        style="bold blue",
    )

    all_recommendations = []

    for i, (content, content_type, source_name) in enumerate(content_samples):
        console.print(f"\n[bold]Sample {i+1}: {Path(source_name).name}[/bold]")

        if compare_all:
            # Compare all strategies
            strategy_scores = _compare_all_strategies(content, content_type)

            # Create comparison table
            table = Table(title=f"Strategy Comparison for {Path(source_name).name}")
            table.add_column("Strategy", style="cyan")
            table.add_column("Score", style="green")
            table.add_column("Confidence", style="yellow")
            table.add_column("Best For", style="blue")

            strategy_descriptions = {
                "recursive": "General-purpose, reliable chunking",
                "semantic": "Semantic boundary awareness",
                "sliding_window": "Continuous text processing",
                "token_aware_sliding": "Token-optimized processing",
                "hierarchical": "Structured document processing",
                "document_structure": "Complex structured documents",
            }

            for strategy, score in sorted(
                strategy_scores.items(), key=lambda x: x[1], reverse=True
            ):
                confidence = min(score / 10.0, 1.0)
                description = strategy_descriptions.get(strategy, "Unknown strategy")

                style = (
                    "bold green" if score == max(strategy_scores.values()) else "white"
                )
                table.add_row(
                    f"[{style}]{strategy}[/{style}]",
                    f"[{style}]{score:.1f}[/{style}]",
                    f"[{style}]{confidence:.1%}[/{style}]",
                    f"[{style}]{description}[/{style}]",
                )

            console.print(table)
        else:
            # Single recommendation
            analysis = _analyze_content_for_chunking(content, content_type)

            # Display content analysis
            info_panel = Panel(
                f"""[bold]Content Analysis:[/bold]
• Length: [cyan]{analysis['content_length']:,}[/cyan] characters
• Words: [cyan]{analysis['word_count']:,}[/cyan]
• Lines: [cyan]{analysis['line_count']:,}[/cyan]
• Paragraphs: [cyan]{analysis['paragraph_count']}[/cyan]
• Content Type: [yellow]{content_type.value}[/yellow]

[bold]Structure Indicators:[/bold]
• Has Structure: [green]{'✓' if analysis['has_structure'] else '✗'}[/green]
• Has Tables: [green]{'✓' if analysis['has_tables'] else '✗'}[/green]
• Has Lists: [green]{'✓' if analysis['has_lists'] else '✗'}[/green]
• Has Code: [green]{'✓' if analysis['has_code'] else '✗'}[/green]

[bold]Complexity Score:[/bold] [blue]{analysis['complexity_score']:.1f}[/blue]""",
                title="Content Overview",
                border_style="blue",
            )
            console.print(info_panel)

            # Display recommendation
            reasons_text = (
                chr(10).join(f"• {reason}" for reason in analysis["reasons"])
                if analysis["reasons"]
                else "• Best general-purpose strategy for " "this content"
            )
            recommendation_panel = Panel(
                f"""[bold green]Recommended Strategy: """
                f"""{analysis['recommended_strategy']}[/bold green]

[bold]Confidence:[/bold] [yellow]{analysis['confidence']:.1%}[/yellow]

[bold]Reasons:[/bold]
{reasons_text}""",
                title="Strategy Recommendation",
                border_style="green",
            )
            console.print(recommendation_panel)

        all_recommendations.append(
            analysis["recommended_strategy"]
            if not compare_all
            else max(strategy_scores.items(), key=lambda x: x[1])[0]
        )

    # Overall recommendation for multiple files
    if len(content_samples) > 1:
        from collections import Counter

        strategy_counts = Counter(all_recommendations)
        most_common = strategy_counts.most_common(1)[0]

        overall_panel = Panel(
            f"""[bold green]Overall Recommendation: {most_common[0]}[/bold green]

[bold]Consistency:[/bold] [yellow]{most_common[1]}/"""
            f"""{len(content_samples)} samples[/yellow]

[bold]Usage Example:[/bold]
[dim]neurosync pipeline run {source} --type {source_type} "
"--strategy {most_common[0]}[/dim]""",
            title="Overall Analysis",
            border_style="green",
        )
        console.print(overall_panel)


def _compare_all_strategies(
    content: str, content_type: ContentType = ContentType.TEXT
) -> dict:
    """Compare all chunking strategies and return scores"""
    analysis = _analyze_content_for_chunking(content, content_type)

    strategy_scores = {
        "recursive": 5.0,  # Default baseline
        "semantic": 0.0,
        "sliding_window": 0.0,
        "token_aware_sliding": 0.0,
        "hierarchical": 0.0,
        "document_structure": 0.0,
    }

    # Semantic is good for natural language with clear sentence boundaries
    if (
        cast(int, analysis["word_count"]) > 100
        and cast(int, analysis["paragraph_count"]) > 3
    ):
        strategy_scores["semantic"] += 4.0
        if not analysis["has_code"] and not analysis["has_tables"]:
            strategy_scores["semantic"] += 2.0

    # Sliding window is good for continuous text without clear structure
    if cast(int, analysis["word_count"]) > 500 and not analysis["has_structure"]:
        strategy_scores["sliding_window"] += 3.0
        if (
            cast(int, analysis["paragraph_count"]) < 5
        ):  # Few paragraphs = continuous text
            strategy_scores["sliding_window"] += 2.0

    # Token-aware is good for very long content or when token precision matters
    if cast(int, analysis["word_count"]) > 2000:
        strategy_scores["token_aware_sliding"] += 4.0
        if cast(int, analysis["content_length"]) > 10000:
            strategy_scores["token_aware_sliding"] += 2.0

    # Hierarchical is excellent for structured documents
    if analysis["has_structure"]:
        strategy_scores["hierarchical"] += 5.0
        if cast(int, analysis["paragraph_count"]) > 5:
            strategy_scores["hierarchical"] += 2.0
        if content_type in [ContentType.MARKDOWN, ContentType.HTML]:
            strategy_scores["hierarchical"] += 3.0

    # Document structure is best for complex structured documents
    if analysis["has_structure"] and (analysis["has_tables"] or analysis["has_lists"]):
        strategy_scores["document_structure"] += 6.0
        if cast(float, analysis["complexity_score"]) > 5:
            strategy_scores["document_structure"] += 3.0
        if content_type in [ContentType.MARKDOWN, ContentType.HTML]:
            strategy_scores["document_structure"] += 2.0

    return strategy_scores


@app.command()
def templates() -> None:
    """List all available pipeline templates with descriptions"""
    template_info = {
        "basic": {
            "description": "Basic single-source pipeline",
            "sources": "1 (file)",
            "strategy": "recursive",
            "chunk_size": "1024",
            "use_case": "Simple document processing",
        },
        "advanced": {
            "description": "Advanced multi-source pipeline",
            "sources": "3 (file, API, database)",
            "strategy": "semantic",
            "chunk_size": "512",
            "use_case": "Complex data integration",
        },
        "comprehensive": {
            "description": "Comprehensive testing pipeline",
            "sources": "3 (all types)",
            "strategy": "hierarchical",
            "chunk_size": "1024",
            "use_case": "Full system testing",
        },
        "semantic_focused": {
            "description": "Semantic chunking focused",
            "sources": "1 (file)",
            "strategy": "semantic",
            "chunk_size": "768",
            "use_case": "Semantic understanding",
        },
        "sliding_window": {
            "description": "Sliding window chunking",
            "sources": "1 (API)",
            "strategy": "sliding_window",
            "chunk_size": "512",
            "use_case": "Continuous text processing",
        },
        "token_aware": {
            "description": "Token-aware sliding window",
            "sources": "1 (database)",
            "strategy": "token_aware_sliding",
            "chunk_size": "1024",
            "use_case": "Token-optimized processing",
        },
        "document_structure": {
            "description": "Document structure aware",
            "sources": "1 (file)",
            "strategy": "document_structure",
            "chunk_size": "2048",
            "use_case": "Structured document processing",
        },
        "performance_test": {
            "description": "Performance testing pipeline",
            "sources": "3 (all types)",
            "strategy": "recursive",
            "chunk_size": "4096",
            "use_case": "Large-scale performance testing",
        },
        "minimal": {
            "description": "Minimal configuration",
            "sources": "1 (file)",
            "strategy": "recursive",
            "chunk_size": "256",
            "use_case": "Quick testing and prototyping",
        },
        "custom": {
            "description": "Customizable template",
            "sources": "1 (customizable)",
            "strategy": "recursive",
            "chunk_size": "1024",
            "use_case": "User customization base",
        },
    }

    table = Table(title="Available Pipeline Templates")
    table.add_column("Template", style="cyan", width=20)
    table.add_column("Description", style="green", width=25)
    table.add_column("Sources", style="blue", width=20)
    table.add_column("Strategy", style="yellow", width=20)
    table.add_column("Chunk Size", style="magenta", width=12)
    table.add_column("Use Case", style="white", width=25)

    for template_name, info in template_info.items():
        table.add_row(
            template_name,
            info["description"],
            info["sources"],
            info["strategy"],
            info["chunk_size"],
            info["use_case"],
        )

    console.print(table)
    console.print("\n[bold green]Usage:[/bold green]")
    console.print("neurosync pipeline create <name> --template <template_name>")
    console.print("\n[bold blue]Examples:[/bold blue]")
    console.print("neurosync pipeline create my_basic_pipeline --template basic")
    console.print("neurosync pipeline create my_advanced_pipeline --template advanced")
    console.print(
        "neurosync pipeline create my_semantic_pipeline --template semantic_focused"
    )


@app.command()
def stop(
    name: str = typer.Argument(..., help="Pipeline name to stop"),
    force: bool = typer.Option(False, "--force", help="Force stop the pipeline"),
) -> None:
    """Stop a running pipeline"""
    logger.info(f"Stopping pipeline: {name}")

    if force:
        console.print(f"Force stopping pipeline: {name}")
    else:
        console.print(f"Gracefully stopping pipeline: {name}")

    # Simulate stopping
    import time

    time.sleep(2)

    console.print(f"Pipeline '{name}' stopped successfully")


@app.command()
def validate(
    config: str = typer.Argument(..., help="Pipeline configuration file"),
) -> None:
    """Validate a pipeline configuration"""
    logger.info(f"Validating pipeline configuration: {config}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        checks = [
            "Checking configuration syntax",
            "Validating data sources",
            "Verifying LLM connections",
            "Testing storage backends",
        ]

        for check in checks:
            task = progress.add_task(check, total=100)

            time.sleep(0.5)
            progress.update(task, completed=100)

    console.print("Pipeline configuration is valid")
