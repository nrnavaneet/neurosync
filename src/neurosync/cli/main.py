"""
NeuroSync CLI - Main entry point
"""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from neurosync.cli.commands import ingest, pipeline, status
from neurosync.core.config.settings import settings
from neurosync.core.logging.logger import get_logger

# Initialize CLI app
app = typer.Typer(
    name="neurosync",
    help="AI-Native ETL Pipeline for RAG and LLM Applications",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Initialize console and logger
console = Console()
logger = get_logger(__name__)

# Add subcommands
app.add_typer(ingest.app, name="ingest", help="Data ingestion commands")
app.add_typer(pipeline.app, name="pipeline", help="Pipeline management commands")
app.add_typer(status.app, name="status", help="System status commands")


def version_callback(value: bool) -> None:
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
        import logging

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
) -> None:
    """Initialize a new NeuroSync project"""
    from pathlib import Path

    if directory:
        base_dir = Path(directory)
        target_dir = base_dir / name
    else:
        target_dir = Path(f"./{name}")

    target_dir.mkdir(parents=True, exist_ok=True)

    config_dir = target_dir / "config"
    data_dir = target_dir / "data"
    logs_dir = target_dir / "logs"

    config_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

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

    console.print(f"Initialized NeuroSync project '{name}' in {target_dir}")
    console.print(f"Edit {config_file} to configure your pipeline")


if __name__ == "__main__":
    app()
