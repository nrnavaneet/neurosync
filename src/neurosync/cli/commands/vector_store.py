"""
CLI commands for managing and interacting with vector stores.

This module provides comprehensive command-line tools for vector store
management, including building indexes, querying embeddings, configuration
management, and performance optimization. It supports multiple vector
store backends and provides utilities for backup, restoration, and monitoring.

Key Commands:
    build: Create vector indexes from processed text chunks
    search: Semantic search with configurable parameters
    config: Generate and manage vector store configurations
    backup: Create versioned backups of vector indexes
    restore: Restore from backup versions
    optimize: Performance optimization and index maintenance
    stats: Index statistics and performance metrics

Features:
    - Multi-backend support (FAISS, Qdrant, Chroma)
    - Batch processing with progress monitoring
    - Semantic search with similarity scoring
    - Configuration templating and validation
    - Backup and versioning system
    - Performance optimization tools
    - Index statistics and monitoring
    - Rich CLI interface with formatted output

Supported Vector Stores:
    - FAISS: High-performance similarity search
    - Qdrant: Production-scale vector database
    - Chroma: Developer-friendly vector store
    - In-memory: Fast development and testing

Vector Store Operations:
    - Index creation and updates
    - Similarity search and retrieval
    - Metadata filtering and faceted search
    - Batch operations for large datasets
    - Index optimization and compression
    - Backup and disaster recovery

Example Usage:
    # Build vector index from chunks
    $ neurosync vector-store build chunks.json embedding.yaml store.yaml

    # Semantic search
    $ neurosync vector-store search "machine learning concepts" --top-k 10

    # Generate configuration
    $ neurosync vector-store config --type faiss --output store.yaml

    # Create backup
    $ neurosync vector-store backup --version v1.0

    # Performance optimization
    $ neurosync vector-store optimize --method compress

For configuration examples and performance tuning, see:
    - docs/vector-stores.md
    - docs/search-optimization.md
    - examples/vector-store-configs.yaml
"""
import json
import warnings
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

from neurosync.core.logging.logger import get_logger
from neurosync.ingestion.base import (  # Needed to reconstruct Chunk
    ContentType,
    SourceMetadata,
    SourceType,
)
from neurosync.pipelines.embedding_pipeline import EmbeddingPipeline
from neurosync.processing.base import Chunk

# Suppress torch warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

app = typer.Typer(help="Manage and query the vector store")
console = Console()
logger = get_logger(__name__)


@app.command()
def build(
    chunks_file: str = typer.Argument(..., help="Path to the processed chunks file."),
    embedding_config_file: str = typer.Argument(
        ..., help="Path to the embedding config."
    ),
    vector_store_config_file: str = typer.Argument(
        ..., help="Path to the vector store config."
    ),
    batch_size: int = typer.Option(32, help="Batch size for processing."),
    enable_hybrid: bool = typer.Option(False, help="Enable hybrid search indexing."),
    create_backup: bool = typer.Option(False, help="Create backup before building."),
    hybrid_config_file: str = typer.Option(None, help="Path to hybrid search config."),
):
    """Build or update the vector store from processed chunks."""

    with console.status("[bold blue]Loading configurations...") as status:
        # Load configs
        with open(embedding_config_file, "r") as f:
            embedding_config = json.load(f)
        with open(vector_store_config_file, "r") as f:
            vector_store_config = json.load(f)

        hybrid_config = {}
        if hybrid_config_file and Path(hybrid_config_file).exists():
            with open(hybrid_config_file, "r") as f:
                hybrid_config = json.load(f)

        status.update("[bold blue]Loading chunks...")
        # Load chunks
        with open(chunks_file, "r") as f:
            chunk_data = json.load(f)

    # Reconstruct Chunk objects
    chunks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[blue]Loading chunks...", total=len(chunk_data))

        for data in chunk_data:
            # A bit complex due to nested dataclasses
            source_metadata_data = data["source_metadata"]

            # Handle enum conversion
            source_metadata_data["source_type"] = SourceType(
                source_metadata_data["source_type"]
            )
            source_metadata_data["content_type"] = ContentType(
                source_metadata_data["content_type"]
            )

            metadata = SourceMetadata(**source_metadata_data)
            chunk = Chunk(
                chunk_id=data["chunk_id"],
                content=data["content"],
                sequence_num=data["sequence_num"],
                source_metadata=metadata,
                quality_score=data["quality_score"],
                processing_metadata=data["processing_metadata"],
            )
            chunks.append(chunk)
            progress.advance(task)

    rprint(
        f"\n[bold green]Building vector store with {len(chunks)} chunks...[/bold green]"
    )

    pipeline = EmbeddingPipeline(
        embedding_config,
        vector_store_config,
        enable_hybrid_search=enable_hybrid,
        hybrid_search_config=hybrid_config,
    )

    try:
        with console.status("[bold yellow]Building vector store...") as status:
            pipeline.run(chunks, batch_size=batch_size, create_backup=create_backup)

        rprint("[bold green]Vector store built successfully![/bold green]")

        # Display metrics
        metrics = pipeline.get_metrics()
        _display_metrics(metrics)

    except Exception as e:
        rprint(f"[bold red]Failed to build vector store: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Option(..., help="The search query text."),
    embedding_config_file: str = typer.Option(
        ..., help="Path to the embedding config."
    ),
    vector_store_config_file: str = typer.Option(
        ..., help="Path to the vector store config."
    ),
    top_k: int = typer.Option(5, help="Number of results to return."),
    use_hybrid: bool = typer.Option(False, help="Use hybrid search if available."),
    hybrid_config_file: str = typer.Option(None, help="Path to hybrid search config."),
    show_scores: bool = typer.Option(True, help="Show detailed scores."),
):
    """Perform a similarity search in the vector store."""

    with console.status("[bold blue]Loading configurations...") as status:
        # Load configs
        with open(embedding_config_file, "r") as f:
            embedding_config = json.load(f)
        with open(vector_store_config_file, "r") as f:
            vector_store_config = json.load(f)

        hybrid_config = {}
        if hybrid_config_file and Path(hybrid_config_file).exists():
            with open(hybrid_config_file, "r") as f:
                hybrid_config = json.load(f)

        status.update("[bold blue]Initializing pipeline...")
        pipeline = EmbeddingPipeline(
            embedding_config,
            vector_store_config,
            enable_hybrid_search=use_hybrid,
            hybrid_search_config=hybrid_config,
        )

    # Create search query panel
    query_panel = Panel(
        Text(query, style="bold white"), title=" Search Query", border_style="cyan"
    )
    console.print(query_panel)

    # Perform search with spinner
    with console.status("[bold yellow]Searching vector store...") as status:
        results = pipeline.search(query, top_k, use_hybrid=use_hybrid)

    if not results:
        rprint("[bold yellow]No results found.[/bold yellow]")
        return

    # Display results
    rprint(f"\n[bold green]Found {len(results)} results[/bold green]")

    table = Table(
        title=f" Top {len(results)} Search Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", style="cyan", width=4)
    table.add_column("Score", style="magenta", width=8)

    if show_scores and use_hybrid and len(results) > 0 and "dense_score" in results[0]:
        table.add_column("Dense", style="blue", width=7)
        table.add_column("Sparse", style="green", width=7)
        table.add_column("Hybrid", style="yellow", width=7)

    table.add_column("Source", style="cyan", width=12)
    table.add_column("Text Snippet", style="white", max_width=50)

    for i, result in enumerate(results):
        row = [
            str(i + 1),
            f"{result['score']:.4f}",
        ]

        if show_scores and use_hybrid and "dense_score" in result:
            row.extend(
                [
                    f"{result['dense_score']:.3f}",
                    f"{result['sparse_score']:.3f}",
                    f"{result['hybrid_score']:.3f}",
                ]
            )

        metadata = result["metadata"]
        source_text = metadata.get("source_id", "N/A")
        snippet = metadata.get("text", "N/A")

        # Truncate snippet for better display
        if len(snippet) > 100:
            snippet = snippet[:97] + "..."

        row.extend([source_text, snippet])

        table.add_row(*row)

    console.print(table)


@app.command()
def create_config(
    type: str = typer.Argument(
        ..., help="Config type: 'embedding', 'vector-store', or 'hybrid'"
    ),
    output_file: str = typer.Argument("config.json"),
    model_type: str = typer.Option(
        "huggingface", help="Model type for embedding config"
    ),
    store_type: str = typer.Option("faiss", help="Store type for vector store config"),
    index_type: str = typer.Option("flat", help="FAISS index type"),
    use_defaults: bool = typer.Option(True, help="Use default configuration templates"),
):
    """Create configuration files with optional default templates."""

    config_dir = (
        Path(__file__).parent.parent.parent.parent.parent / "config" / "defaults"
    )

    with console.status("[bold blue]Creating configuration..."):
        if type == "embedding":
            if use_defaults and model_type == "huggingface":
                default_file = config_dir / "embedding_huggingface.json"
                if default_file.exists():
                    import shutil

                    shutil.copy(default_file, output_file)
                    rprint(
                        "[bold green] Created embedding config from template: "
                        f"{output_file}[/bold green]"
                    )
                    return
            elif use_defaults and model_type == "openai":
                default_file = config_dir / "embedding_openai.json"
                if default_file.exists():
                    import shutil

                    shutil.copy(default_file, output_file)
                    rprint(
                        "[bold green] Created embedding config from template: "
                        f"{output_file}[/bold green]"
                    )
                    rprint(
                        "[bold yellow]  Don't forget to update your OpenAI "
                        "API key![/bold yellow]"
                    )
                    return

            # Fallback to manual creation
            if model_type == "huggingface":
                config = {
                    "type": "huggingface",
                    "model_name": "all-MiniLM-L6-v2",
                    "enable_monitoring": True,
                }
            elif model_type == "openai":
                config = {
                    "type": "openai",
                    "model_name": "text-embedding-3-small",
                    "api_key": "your-openai-api-key",
                    "max_batch_size": 2048,
                    "enable_monitoring": True,
                }
            else:
                rprint(
                    "[bold red] Invalid model type. Use 'huggingface' or "
                    "'openai'.[/bold red]"
                )
                raise typer.Exit(1)

        elif type == "vector-store":
            if use_defaults and store_type == "faiss":
                default_file = config_dir / f"vector_store_faiss_{index_type}.json"
                if default_file.exists():
                    import shutil

                    shutil.copy(default_file, output_file)
                    rprint(
                        "[bold green] Created vector store config from "
                        f"template: {output_file}[/bold green]"
                    )
                    return
            elif use_defaults and store_type == "qdrant":
                default_file = config_dir / "vector_store_qdrant.json"
                if default_file.exists():
                    import shutil

                    shutil.copy(default_file, output_file)
                    rprint(
                        "[bold green] Created vector store config from "
                        f"template: {output_file}[/bold green]"
                    )
                    return

            # Fallback to manual creation
            if store_type == "faiss":
                config = {
                    "type": "faiss",
                    "path": "./vector_store",
                    "index_type": index_type,
                    "index_params": {
                        "nlist": 100,
                        "m": 16,
                        "ef_construction": 200,
                        "ef_search": 100,
                    },
                    "enable_versioning": True,
                }
            elif store_type == "qdrant":
                config = {
                    "type": "qdrant",
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "neurosync_vectors",
                    "distance_metric": "cosine",
                    "enable_versioning": False,
                }
            else:
                rprint(
                    "[bold red] Invalid store type. Use 'faiss' or "
                    "'qdrant'.[/bold red]"
                )
                raise typer.Exit(1)

        elif type == "hybrid":
            config = {
                "sparse_weight": 0.3,
                "dense_weight": 0.7,
                "use_rank_fusion": True,
                "sparse_store_path": "./sparse_store",
            }
        else:
            rprint(
                "[bold red] Invalid config type. Use 'embedding', "
                "'vector-store', or 'hybrid'.[/bold red]"
            )
            raise typer.Exit(1)

        with open(output_file, "w") as f:
            json.dump(config, f, indent=2)
        rprint(f"[bold green] Created {type} config at {output_file}[/bold green]")


@app.command()
def list_defaults():
    """List available default configuration templates."""

    config_dir = (
        Path(__file__).parent.parent.parent.parent.parent / "config" / "defaults"
    )

    if not config_dir.exists():
        rprint("[bold red] Default configuration directory not found[/bold red]")
        return

    defaults = list(config_dir.glob("*.json"))

    if not defaults:
        rprint("[bold yellow]  No default configurations found[/bold yellow]")
        return

    table = Table(
        title=" Available Default Configurations",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Template", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Description", style="white")

    for default_file in sorted(defaults):
        name = default_file.stem
        if "embedding" in name:
            config_type = "Embedding"
            if "huggingface" in name:
                desc = "HuggingFace sentence-transformers model"
            elif "openai" in name:
                desc = "OpenAI text embedding API"
            else:
                desc = "Custom embedding configuration"
        elif "vector_store" in name:
            config_type = "Vector Store"
            if "faiss_flat" in name:
                desc = "FAISS flat index - simple & fast"
            elif "faiss_hnsw" in name:
                desc = "FAISS HNSW index - high performance"
            elif "faiss_ivf" in name:
                desc = "FAISS IVF index - large scale"
            elif "qdrant" in name:
                desc = "Qdrant vector database"
            else:
                desc = "Custom vector store configuration"
        else:
            config_type = "Other"
            desc = "Custom configuration"

        table.add_row(name, config_type, desc)

    console.print(table)

    # Add usage instructions
    usage_panel = Panel(
        " [bold]Usage Tips:[/bold]\n\n"
        "• Use [cyan]neurosync vector-store create-config embedding[/cyan] "
        "with [green]--use-defaults[/green]\n"
        "• Use [cyan]neurosync vector-store create-config vector-store[/cyan] "
        "with [green]--use-defaults[/green]\n"
        "• Specify [green]--model-type[/green] or [green]--store-type[/green] "
        "and [green]--index-type[/green] for specific templates\n"
        "• Default configs are copied to your specified output file",
        title="How to Use Default Configs",
        border_style="blue",
    )
    console.print(usage_panel)


@app.command()
def info(
    embedding_config_file: str = typer.Argument(
        ..., help="Path to the embedding config."
    ),
    vector_store_config_file: str = typer.Argument(
        ..., help="Path to the vector store config."
    ),
):
    """Display information about the vector store."""

    # Load configs
    with open(embedding_config_file, "r") as f:
        embedding_config = json.load(f)
    with open(vector_store_config_file, "r") as f:
        vector_store_config = json.load(f)

    pipeline = EmbeddingPipeline(embedding_config, vector_store_config)

    # Get metrics
    metrics = pipeline.get_metrics()
    _display_metrics(metrics)


@app.command()
def backup(
    embedding_config_file: str = typer.Argument(
        ..., help="Path to the embedding config."
    ),
    vector_store_config_file: str = typer.Argument(
        ..., help="Path to the vector store config."
    ),
    description: str = typer.Option("Manual backup", help="Backup description"),
):
    """Create a backup of the vector store."""

    # Load configs
    with open(embedding_config_file, "r") as f:
        embedding_config = json.load(f)
    with open(vector_store_config_file, "r") as f:
        vector_store_config = json.load(f)

    pipeline = EmbeddingPipeline(embedding_config, vector_store_config)

    try:
        backup_id = pipeline.vector_store_manager.create_backup(description)
        if backup_id:
            console.print(f" Created backup: {backup_id}", style="green")
        else:
            console.print(" Versioning not enabled", style="yellow")
    except Exception as e:
        console.print(f" Failed to create backup: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def restore(
    version_id: str = typer.Argument(..., help="Version ID to restore"),
    embedding_config_file: str = typer.Argument(
        ..., help="Path to the embedding config."
    ),
    vector_store_config_file: str = typer.Argument(
        ..., help="Path to the vector store config."
    ),
):
    """Restore from a backup version."""

    # Load configs
    with open(embedding_config_file, "r") as f:
        embedding_config = json.load(f)
    with open(vector_store_config_file, "r") as f:
        vector_store_config = json.load(f)

    pipeline = EmbeddingPipeline(embedding_config, vector_store_config)

    try:
        pipeline.vector_store_manager.restore_backup(version_id)
        console.print(f" Restored from backup: {version_id}", style="green")
    except Exception as e:
        console.print(f" Failed to restore backup: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def list_backups(
    embedding_config_file: str = typer.Argument(
        ..., help="Path to the embedding config."
    ),
    vector_store_config_file: str = typer.Argument(
        ..., help="Path to the vector store config."
    ),
):
    """List all available backups."""

    # Load configs
    with open(embedding_config_file, "r") as f:
        embedding_config = json.load(f)
    with open(vector_store_config_file, "r") as f:
        vector_store_config = json.load(f)

    pipeline = EmbeddingPipeline(embedding_config, vector_store_config)

    backups = pipeline.vector_store_manager.list_backups()

    if not backups:
        console.print(" No backups found", style="yellow")
        return

    table = Table(title="Available Backups")
    table.add_column("Version ID", style="cyan")
    table.add_column("Timestamp", style="green")
    table.add_column("Description", style="white")

    for backup in backups:
        table.add_row(backup["version_id"], backup["timestamp"], backup["description"])

    console.print(table)


@app.command()
def optimize(
    embedding_config_file: str = typer.Argument(
        ..., help="Path to the embedding config."
    ),
    vector_store_config_file: str = typer.Argument(
        ..., help="Path to the vector store config."
    ),
):
    """Optimize the vector store for better performance."""

    # Load configs
    with open(embedding_config_file, "r") as f:
        embedding_config = json.load(f)
    with open(vector_store_config_file, "r") as f:
        vector_store_config = json.load(f)

    pipeline = EmbeddingPipeline(embedding_config, vector_store_config)

    try:
        console.print(" Optimizing vector store...", style="blue")
        pipeline.vector_store_manager.optimize()
        console.print(" Vector store optimized", style="green")
    except Exception as e:
        console.print(f" Failed to optimize: {e}", style="red")
        raise typer.Exit(1)


def _display_metrics(metrics: dict) -> None:
    """Display metrics in a formatted way."""

    # Vector store metrics
    if "vector_store" in metrics:
        store_info = metrics["vector_store"]
        store_panel = Panel(
            f"Type: {store_info.get('type', 'N/A')}\n"
            f"Count: {store_info.get('count', 0):,} vectors\n"
            f"Dimension: {store_info.get('dimension', 0)}\n"
            f"Index Type: {store_info.get('index_type', 'N/A')}\n"
            f"Trained: {store_info.get('is_trained', 'N/A')}",
            title="Vector Store Info",
            border_style="blue",
        )
        console.print(store_panel)

    # Embedding metrics
    if "embedding" in metrics:
        embedding_info = metrics["embedding"]
        embedding_panel = Panel(
            f"Model: {embedding_info.get('model_name', 'N/A')}\n"
            f"Total Texts: {embedding_info.get('total_texts', 0):,}\n"
            f"Total Time: {embedding_info.get('total_time_seconds', 0):.2f}s\n"
            "Avg Time per Text: "
            f"{embedding_info.get('average_time_per_text', 0):.3f}s\n"
            f"Errors: {embedding_info.get('error_count', 0)}",
            title="Embedding Metrics",
            border_style="green",
        )
        console.print(embedding_panel)

    # Hybrid search metrics
    if "hybrid_search" in metrics:
        hybrid_info = metrics["hybrid_search"]
        hybrid_panel = Panel(
            f"Sparse Enabled: {hybrid_info.get('sparse_enabled', False)}\n"
            f"Sparse Documents: {hybrid_info.get('sparse_documents', 0):,}\n"
            f"Dense Weight: {hybrid_info.get('dense_weight', 0)}\n"
            f"Sparse Weight: {hybrid_info.get('sparse_weight', 0)}\n"
            f"Use Rank Fusion: {hybrid_info.get('use_rank_fusion', False)}",
            title="Hybrid Search Info",
            border_style="yellow",
        )
        console.print(hybrid_panel)
