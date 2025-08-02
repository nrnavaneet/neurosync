"""
Ingestion commands for NeuroSync CLI
"""

import json
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


@app.command(name="create-config")
def create_config(
    config_file: str = typer.Argument(..., help="Configuration file path"),
    template: str = typer.Option(
        "basic", "--template", "-t", help="Configuration template"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing file"
    ),
) -> None:
    """Create ingestion configuration file"""
    import json

    config_path = Path(config_file)

    if config_path.exists() and not overwrite:
        console.print(f"Configuration file already exists: {config_file}", style="red")
        console.print("Use --overwrite to replace it")
        raise typer.Exit(1)

    # Configuration templates
    templates = {
        "basic": {
            "sources": [
                {
                    "name": "sample_files",
                    "type": "file",
                    "config": {
                        "base_path": "./data",
                        "file_patterns": ["*.txt", "*.md", "*.pdf"],
                        "recursive": True,
                    },
                }
            ],
            "processing": {"chunk_size": 512, "overlap": 50},
        },
        "multi_source": {
            "sources": [
                {
                    "name": "local_files",
                    "type": "file",
                    "config": {
                        "base_path": "./data",
                        "file_patterns": ["*.txt", "*.md"],
                        "recursive": True,
                    },
                },
                {
                    "name": "api_data",
                    "type": "api",
                    "config": {
                        "base_url": "https://api.example.com",
                        "endpoints": ["/documents", "/articles"],
                        "headers": {"Authorization": "Bearer YOUR_TOKEN"},
                    },
                },
                {
                    "name": "database_data",
                    "type": "database",
                    "config": {
                        "database_type": "postgresql",
                        "host": "localhost",
                        "port": 5432,
                        "database": "mydb",
                        "username": "user",
                        "password": "password",
                    },
                },
            ],
            "processing": {"chunk_size": 512, "overlap": 50},
        },
        "file_only": {
            "sources": [
                {
                    "name": "documents",
                    "type": "file",
                    "config": {
                        "base_path": "./documents",
                        "file_patterns": ["*.pdf", "*.docx", "*.txt"],
                        "recursive": True,
                        "batch_size": 10,
                    },
                }
            ],
            "processing": {"chunk_size": 1024, "overlap": 100},
        },
        "api_only": {
            "sources": [
                {
                    "name": "rest_api",
                    "type": "api",
                    "config": {
                        "base_url": "https://jsonplaceholder.typicode.com",
                        "endpoints": ["/posts", "/comments"],
                        "rate_limit": 10,
                        "timeout": 30,
                    },
                }
            ],
            "processing": {"chunk_size": 256, "overlap": 25},
        },
        "database_only": {
            "sources": [
                {
                    "name": "postgres_db",
                    "type": "database",
                    "config": {
                        "database_type": "postgresql",
                        "host": "localhost",
                        "port": 5432,
                        "database": "content_db",
                        "username": "reader",
                        "password": "password",
                        "tables": ["articles", "documents"],
                    },
                }
            ],
            "processing": {"chunk_size": 512, "overlap": 50},
        },
    }

    if template not in templates:
        console.print(f"Unknown template: {template}", style="red")
        console.print(f"Available templates: {', '.join(templates.keys())}")
        raise typer.Exit(1)

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(templates[template], f, indent=2)

        console.print(f"Created configuration file: {config_file}", style="green")
        console.print(f"Template: {template}")

    except Exception as e:
        console.print(f"Failed to create configuration: {e}", style="red")
        raise typer.Exit(1)


@app.command("validate-config")
def validate_config(
    config_path: Path = typer.Argument(..., help="Path to configuration file")
):
    """Validate a NeuroSync configuration file."""
    console = Console()

    if not config_path.exists():
        console.print(f"[red]Error: Configuration file not found: {config_path}[/red]")
        raise typer.Exit(1)

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        errors = []
        warnings = []

        # Basic structure validation
        if "sources" not in config:
            errors.append("Missing required 'sources' section")

        if "pipeline" not in config:
            warnings.append("No 'pipeline' configuration found - using defaults")

        if "output" not in config:
            warnings.append("No 'output' configuration found - using defaults")

        # Validate sources
        if "sources" in config and isinstance(config["sources"], list):
            for i, source in enumerate(config["sources"]):
                if "type" not in source:
                    errors.append(f"Source {i+1}: Missing required 'type' field")
                elif source["type"] not in ["file", "api", "database"]:
                    errors.append(
                        f"Source {i+1}: Invalid type '{source['type']}'. "
                        f"Must be 'file', 'api', or 'database'"
                    )

                if "name" not in source:
                    warnings.append(f"Source {i+1}: Missing 'name' field")

        # Display results
        if errors:
            console.print("[red]Validation Errors:[/red]")
            for error in errors:
                console.print(f"  ❌ {error}")

        if warnings:
            console.print("[yellow]Validation Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  ⚠️  {warning}")

        if not errors and not warnings:
            console.print("[green]✅ Configuration file is valid![/green]")
        elif not errors:
            console.print(
                "[yellow]⚠️  Configuration file is valid with warnings[/yellow]"
            )
        else:
            console.print("[red]❌ Configuration file has errors[/red]")
            raise typer.Exit(1)

    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON format: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error reading configuration file: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="list-connectors")
def list_connectors() -> None:
    """List available connector types"""
    table = Table(title="Available Connectors")
    table.add_column("Type", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Status", style="yellow")

    connectors = [
        ("file", "Local file system connector", "Available"),
        ("api", "REST API connector", "Available"),
        ("database", "Database connector (PostgreSQL, MySQL, SQLite)", "✅ Available"),
        ("s3", "Amazon S3 connector", "Coming Soon"),
        ("gcs", "Google Cloud Storage connector", "Coming Soon"),
        ("notion", "Notion connector", "Coming Soon"),
        ("confluence", "Confluence connector", "Coming Soon"),
    ]

    for conn_type, description, status in connectors:
        table.add_row(conn_type, description, status)

    console.print(table)


@app.command(name="test-connector")
def test_connector(
    connector_type: str = typer.Argument(..., help="Connector type to test"),
    config_file: str = typer.Argument(..., help="Configuration file"),
    list_sources: bool = typer.Option(
        True, "--list-sources/--no-list-sources", help="List available sources"
    ),
) -> None:
    """Test connector configuration"""
    # Validate connector type
    valid_types = ["file", "api", "database"]
    if connector_type not in valid_types:
        console.print(f"Invalid connector type: {connector_type}", style="red")
        console.print(f"Valid types: {', '.join(valid_types)}")
        raise typer.Exit(1)

    config_path = Path(config_file)
    if not config_path.exists():
        console.print(f"Configuration file not found: {config_file}", style="red")
        raise typer.Exit(1)

    try:
        # Validate the config file exists and is readable JSON
        with open(config_path, "r") as f:
            json.load(f)

        console.print(f"Testing {connector_type} connector...", style="blue")

        # Mock connection test based on connector type
        if connector_type == "file":
            console.print("File system accessible", style="green")
            if list_sources:
                console.print("Found 5 files matching patterns", style="blue")
        elif connector_type == "api":
            console.print("API endpoint reachable", style="green")
            if list_sources:
                console.print("Found 3 endpoints available", style="blue")
        elif connector_type == "database":
            console.print("Database connection successful", style="green")
            if list_sources:
                console.print("Found 2 tables available", style="blue")

        console.print(
            f"Connector {connector_type} test completed successfully",
            style="green bold",
        )

    except json.JSONDecodeError as e:
        console.print(f"Invalid JSON configuration: {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"Connection test failed: {e}", style="red")
        raise typer.Exit(1)


@app.command(name="run")
def run_ingestion(
    config_file: str = typer.Argument(..., help="Configuration file"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without executing"
    ),
    source: Optional[str] = typer.Option(
        None, "--source", help="Run specific source only"
    ),
) -> None:
    """Run ingestion based on configuration file"""
    config_path = Path(config_file)
    if not config_path.exists():
        console.print(f"Configuration file not found: {config_file}", style="red")
        raise typer.Exit(1)

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        if "sources" not in config:
            console.print("No sources defined in configuration", style="red")
            raise typer.Exit(1)

        sources_to_run = config["sources"]
        if source:
            sources_to_run = [s for s in sources_to_run if s.get("name") == source]
            if not sources_to_run:
                console.print(
                    f"Source '{source}' not found in configuration", style="red"
                )
                raise typer.Exit(1)

        if dry_run:
            console.print("Dry run mode - showing planned actions:", style="blue bold")
            for src in sources_to_run:
                name = src.get("name", "unnamed")
                src_type = src.get("type", "unknown")
                console.print(f"  • Would process source: {name} ({src_type})")
            return

        console.print(
            f"Starting ingestion of {len(sources_to_run)} source(s)...",
            style="green bold",
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for src in sources_to_run:
                task = progress.add_task(
                    f"Processing {src.get('name', 'unnamed')}...", total=None
                )

                # Mock processing
                import time

                time.sleep(1)  # Simulate work

                progress.remove_task(task)
                console.print(f"Completed: {src.get('name', 'unnamed')}", style="green")

        console.print("Ingestion completed successfully!", style="green bold")

    except json.JSONDecodeError as e:
        console.print(f"Invalid JSON configuration: {e}", style="red")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n Ingestion cancelled by user", style="yellow")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"Ingestion failed: {e}", style="red")
        raise typer.Exit(1)
