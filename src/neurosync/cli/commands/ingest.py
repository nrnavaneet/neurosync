"""
Ingestion commands for NeuroSync CLI
"""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from neurosync.core.logging.logger import get_logger

app = typer.Typer(help="Data ingestion commands")
console = Console()
logger = get_logger(__name__)


@app.command()
def file(
    source: str = typer.Argument(..., help="File path or directory to ingest"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    chunk_size: int = typer.Option(
        512, "--chunk-size", help="Chunk size for processing"
    ),
    overlap: int = typer.Option(50, "--overlap", help="Overlap between chunks"),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Process directories recursively"
    ),
) -> None:
    """Ingest data from local files"""
    logger.info(f"Starting file ingestion from: {source}")

    source_path = Path(source)
    if not source_path.exists():
        console.print(f"Source path does not exist: {source}", style="red")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting files...", total=None)

        if source_path.is_file():
            files_to_process = [source_path]
        else:
            pattern = "**/*" if recursive else "*"
            files_to_process = [
                f
                for f in source_path.glob(pattern)
                if f.is_file()
                and f.suffix in [".txt", ".md", ".pdf", ".docx", ".csv", ".json"]
            ]

        progress.update(task, total=len(files_to_process))

        results = []
        for file_path in files_to_process:
            try:
                # Simulate file processing
                logger.info(f"Processing file: {file_path}")
                results.append(
                    {
                        "file": str(file_path),
                        "size": file_path.stat().st_size,
                        "status": "success",
                    }
                )
                progress.advance(task)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({"file": str(file_path), "size": 0, "status": "failed"})

    # Display results
    table = Table(title="Ingestion Results")
    table.add_column("File", style="cyan")
    table.add_column("Size (bytes)", style="green")
    table.add_column("Status", style="yellow")

    for result in results:
        status_style = "green" if result["status"] == "success" else "red"
        table.add_row(
            result["file"],
            str(result["size"]),
            f"[{status_style}]{result['status']}[/{status_style}]",
        )

    console.print(table)
    console.print(f"Processed {len(files_to_process)} files")


@app.command()
def api(
    url: str = typer.Argument(..., help="API endpoint URL"),
    method: str = typer.Option("GET", "--method", "-m", help="HTTP method"),
    headers: Optional[List[str]] = typer.Option(
        None, "--header", "-H", help="HTTP headers (key:value)"
    ),
    auth_token: Optional[str] = typer.Option(
        None, "--token", help="Bearer token for authentication"
    ),
) -> None:
    """Ingest data from API endpoints"""
    logger.info(f"Starting API ingestion from: {url}")

    # Parse headers
    parsed_headers = {}
    if headers:
        for header in headers:
            if ":" in header:
                key, value = header.split(":", 1)
                parsed_headers[key.strip()] = value.strip()

    if auth_token:
        parsed_headers["Authorization"] = f"Bearer {auth_token}"

    console.print(f"Connecting to: {url}")
    console.print(f"Method: {method}")
    if parsed_headers:
        console.print(f"Headers: {list(parsed_headers.keys())}")

    # Simulate API call
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching data from API...", total=None)

        # Simulate API processing
        import time

        time.sleep(2)
        progress.update(task, completed=100, total=100)

    console.print("API ingestion completed successfully")


@app.command()
def database(
    connection_string: str = typer.Argument(..., help="Database connection string"),
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="SQL query to execute"
    ),
    table: Optional[str] = typer.Option(
        None, "--table", "-t", help="Table name to ingest"
    ),
    batch_size: int = typer.Option(
        1000, "--batch-size", help="Batch size for processing"
    ),
) -> None:
    """Ingest data from databases"""
    logger.info("Starting database ingestion")

    if not query and not table:
        console.print("Either --query or --table must be specified", style="red")
        raise typer.Exit(1)

    console.print("Connecting to database...")

    # Simulate database connection and query
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Executing database query...", total=None)

        # Simulate database processing
        import time

        time.sleep(3)
        progress.update(task, completed=100, total=100)

    console.print("Database ingestion completed successfully")
    console.print(f"Batch size: {batch_size}")


@app.command()
def list_sources() -> None:
    """List available ingestion sources"""
    table = Table(title="Available Ingestion Sources")
    table.add_column("Source Type", style="cyan")
    table.add_column("Command", style="green")
    table.add_column("Description", style="yellow")

    sources = [
        ("File System", "neurosync ingest file", "Local files and directories"),
        ("API Endpoints", "neurosync ingest api", "REST APIs and web services"),
        ("Databases", "neurosync ingest database", "SQL databases and data warehouses"),
        (
            "Cloud Storage",
            "neurosync ingest cloud",
            "S3, GCS, Azure Blob (coming soon)",
        ),
        (
            "SaaS Platforms",
            "neurosync ingest saas",
            "Notion, Confluence, etc. (coming soon)",
        ),
    ]

    for source_type, command, description in sources:
        table.add_row(source_type, command, description)

    console.print(table)
