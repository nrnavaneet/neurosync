"""
NeuroSync CLI - Main entry point
"""

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from neurosync.cli.commands import ingest, process, serve, status, vector_store
from neurosync.core.config.settings import settings
from neurosync.core.logging.logger import get_logger

# Initialize CLI app
app = typer.Typer(
    name="neurosync",
    help="AI-Native ETL Pipeline for RAG and LLM Applications",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

# Initialize console and logger
console = Console()
logger = get_logger(__name__)

# Add subcommands
app.add_typer(ingest.app, name="ingest", help="Data ingestion commands")
app.add_typer(status.app, name="status", help="System status commands")
app.add_typer(process.app, name="process", help="Intelligent processing and chunking")
app.add_typer(
    vector_store.app, name="vector-store", help="Vector store management commands"
)
app.add_typer(serve.app, name="serve", help="API serving and LLM integration commands")


@app.command(name="run")
def run_pipeline_command(
    input_path: str = typer.Argument(
        ..., help="Input path (file, directory, URL, or database connection)"
    ),
    auto: bool = typer.Option(
        False, "--auto", help="Run in automatic mode with smart defaults"
    ),
    output_dir: str = typer.Option(
        "./", "--output-dir", "-o", help="Output directory for generated files"
    ),
) -> None:
    """
    Run the complete NeuroSync pipeline from ingestion to chat.

    This command handles everything:
    - Data ingestion from any source
    - Intelligent processing and chunking
    - Embedding generation
    - Vector store creation
    - LLM setup and interactive chat

    Just provide your input path and let NeuroSync do the rest!
    """
    import os
    from pathlib import Path

    from neurosync.pipelines.pipeline import FullPipeline

    # Change to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    os.chdir(output_path)

    # Initialize and run pipeline
    pipeline = FullPipeline()

    # Show welcome message
    if not auto:
        pipeline.show_welcome()

    # Run the full pipeline
    pipeline.run_full_pipeline(input_path, auto_mode=auto)


def version_callback(ctx, param, value: bool) -> None:
    """Handle version callback"""
    if value:
        table = Table(title="NeuroSync Version Information")
        table.add_column("Component", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Environment", style="yellow")

        # Add core system information rows
        table.add_row("NeuroSync", settings.APP_VERSION, settings.ENVIRONMENT)
        table.add_row("Python", "3.11+", "Required")
        table.add_row("API Port", str(settings.API_PORT), "Default")

        console.print(table)
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show NeuroSync version and exit",
    ),
) -> None:
    """
    NeuroSync CLI - AI-Native ETL Pipeline for RAG and LLM Applications

    Run 'neurosync --help' for available commands.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

    if config_file:
        logger.info(f"Using configuration file: {config_file}")


@app.command()
def version() -> None:
    """Show NeuroSync version information"""
    table = Table(title="NeuroSync Version Information")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Environment", style="yellow")

    # Core version and environment information
    table.add_row("NeuroSync", settings.APP_VERSION, settings.ENVIRONMENT)
    table.add_row("Python", "3.11+", "Required")
    table.add_row("API Port", str(settings.API_PORT), "Default")

    console.print(table)


@app.command()
def init(
    name: str = typer.Argument(..., help="Project name"),
    directory: Optional[str] = typer.Option(
        None, "--dir", "-d", help="Target directory"
    ),
    with_defaults: bool = typer.Option(
        True,
        "--with-defaults/--no-defaults",
        help="Include default configuration files",
    ),
) -> None:
    """Initialize a new NeuroSync project"""
    import json
    import shutil
    from pathlib import Path

    if directory:
        base_dir = Path(directory)
        target_dir = base_dir / name
    else:
        target_dir = Path(f"./{name}")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    config_dir = target_dir / "config"
    data_dir = target_dir / "data"
    logs_dir = target_dir / "logs"

    config_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Create main pipeline configuration
    config_file = config_dir / "pipeline.yaml"
    config_content = f"""
# NeuroSync Pipeline Configuration
name: {name}
version: "1.0.0"

ingestion:
  sources: []

processing:
  chunk_size: 512
  overlap: 50

storage:
  vector_store: faiss
  metadata_store: postgres

llm:
  providers: []
"""

    config_file.write_text(config_content.strip())

    # Create default configuration files if requested
    if with_defaults:
        # Get the source defaults directory
        neurosync_root = Path(__file__).parent.parent.parent.parent
        source_defaults_dir = neurosync_root / "config" / "defaults"

        if source_defaults_dir.exists():
            # Create defaults subdirectory in project config
            defaults_dir = config_dir / "defaults"
            defaults_dir.mkdir(exist_ok=True)

            # Copy all default configuration files
            for default_file in source_defaults_dir.glob("*.json"):
                target_file = defaults_dir / default_file.name
                shutil.copy2(default_file, target_file)
                console.print(f"  Created: {target_file.relative_to(target_dir)}")
        else:
            # Create default configurations manually if source directory doesn't exist
            defaults_dir = config_dir / "defaults"
            defaults_dir.mkdir(exist_ok=True)

            # Create embedding configurations
            embedding_hf_config = {
                "type": "huggingface",
                "model_name": "all-MiniLM-L6-v2",
                "enable_monitoring": True,
            }
            (defaults_dir / "embedding_huggingface.json").write_text(
                json.dumps(embedding_hf_config, indent=2)
            )

            embedding_openai_config = {
                "type": "openai",
                "model_name": "text-embedding-3-small",
                "api_key": "your-openai-api-key",
                "max_batch_size": 2048,
                "enable_monitoring": True,
            }
            (defaults_dir / "embedding_openai.json").write_text(
                json.dumps(embedding_openai_config, indent=2)
            )

            # Create vector store configurations
            vector_store_faiss_flat = {
                "type": "faiss",
                "path": "./vector_store",
                "index_type": "flat",
                "index_params": {
                    "nlist": 100,
                    "m": 16,
                    "ef_construction": 200,
                    "ef_search": 100,
                },
                "enable_versioning": True,
            }
            (defaults_dir / "vector_store_faiss_flat.json").write_text(
                json.dumps(vector_store_faiss_flat, indent=2)
            )

            vector_store_faiss_hnsw = {
                "type": "faiss",
                "path": "./vector_store",
                "index_type": "hnsw",
                "index_params": {"M": 16, "ef_construction": 200, "ef_search": 100},
                "enable_versioning": True,
            }
            (defaults_dir / "vector_store_faiss_hnsw.json").write_text(
                json.dumps(vector_store_faiss_hnsw, indent=2)
            )

            vector_store_faiss_ivf = {
                "type": "faiss",
                "path": "./vector_store",
                "index_type": "ivf",
                "index_params": {"nlist": 100, "nprobe": 10},
                "enable_versioning": True,
            }
            (defaults_dir / "vector_store_faiss_ivf.json").write_text(
                json.dumps(vector_store_faiss_ivf, indent=2)
            )

            vector_store_qdrant = {
                "type": "qdrant",
                "host": "localhost",
                "port": 6333,
                "collection_name": "neurosync_vectors",
                "distance_metric": "cosine",
                "enable_versioning": False,
            }
            (defaults_dir / "vector_store_qdrant.json").write_text(
                json.dumps(vector_store_qdrant, indent=2)
            )

            console.print(
                "  Created default configuration files in "
                f"{defaults_dir.relative_to(target_dir)}"
            )

    # Create sample data structure
    sample_dir = data_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    # Create a sample text file
    sample_file = sample_dir / "sample.txt"
    sample_content = """Welcome to NeuroSync!

This is a sample document that demonstrates the capabilities of the NeuroSync
AI-Native ETL Pipeline.

NeuroSync is designed for Retrieval-Augmented Generation (RAG) and Large
Language Model (LLM) applications. It provides:

1. Intelligent data ingestion from multiple sources
2. Advanced text processing and chunking strategies
3. Vector embeddings with multiple provider support
4. Scalable vector storage solutions
5. Hybrid search capabilities

You can use this sample file to test your pipeline:
1. Run: neurosync ingest file samples/sample.txt --output data/chunks.json
2. Run: neurosync process file data/chunks.json --strategy recursive
   --output data/processed.json
3. Run: neurosync vector-store build data/processed.json \\
   config/defaults/embedding_huggingface.json \\
   config/defaults/vector_store_faiss_flat.json

Happy processing with NeuroSync!
"""
    sample_file.write_text(sample_content.strip())

    console.print(
        f"\n[bold green]Initialized NeuroSync project '{name}' "
        f"in {target_dir}[/bold green]"
    )
    console.print("Project structure:")
    console.print("  ├── config/")
    console.print("  │   ├── pipeline.yaml")
    if with_defaults:
        console.print("  │   └── defaults/")
        console.print("  │       ├── embedding_*.json")
        console.print("  │       └── vector_store_*.json")
    console.print("  ├── data/")
    console.print("  │   └── samples/")
    console.print("  │       └── sample.txt")
    console.print("  └── logs/")
    console.print("\nNext steps:")
    console.print(f"  1. cd {name}")
    console.print("  2. Edit config/pipeline.yaml to configure your pipeline")
    console.print(
        "  3. Test with: neurosync ingest file data/samples/sample.txt "
        "--output data/chunks.json"
    )


if __name__ == "__main__":
    app()
