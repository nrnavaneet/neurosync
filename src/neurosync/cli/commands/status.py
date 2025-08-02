"""
Status commands for NeuroSync CLI
"""

import time
from typing import Any, Dict, List, Union

import typer
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from neurosync.core.config.settings import settings
from neurosync.core.logging.logger import get_logger

app = typer.Typer(help="System status commands")
console = Console()
logger = get_logger(__name__)


def check_service_health(service_name: str, host: str, port: int) -> Dict[str, Any]:
    """Check if a service is healthy"""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            return {
                "service": service_name,
                "status": "healthy",
                "host": host,
                "port": port,
                "response_time": "< 5s",
            }
        else:
            return {
                "service": service_name,
                "status": "unhealthy",
                "host": host,
                "port": port,
                "error": "Connection refused",
            }
    except Exception as e:
        return {
            "service": service_name,
            "status": "error",
            "host": host,
            "port": port,
            "error": str(e),
        }


def check_api_health() -> Dict[str, Any]:
    """Check API health with HTTP request"""
    try:
        import httpx

        start_time = time.time()
        response = httpx.get(
            f"http://{settings.API_HOST}:{settings.API_PORT}/health", timeout=5
        )
        response_time = round((time.time() - start_time) * 1000, 2)

        if response.status_code == 200:
            return {
                "service": "NeuroSync API",
                "status": "healthy",
                "host": settings.API_HOST,
                "port": settings.API_PORT,
                "response_time": f"{response_time}ms",
                "version": response.json().get("version", "unknown"),
            }
        else:
            return {
                "service": "NeuroSync API",
                "status": "unhealthy",
                "host": settings.API_HOST,
                "port": settings.API_PORT,
                "error": f"HTTP {response.status_code}",
            }
    except Exception as e:
        return {
            "service": "NeuroSync API",
            "status": "error",
            "host": settings.API_HOST,
            "port": settings.API_PORT,
            "error": str(e),
        }


@app.command()
def system() -> None:
    """Check overall system health"""
    logger.info("Checking system health")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Checking system components...", total=None)

        # Check all services
        services_to_check = [
            ("PostgreSQL", settings.POSTGRES_HOST, settings.POSTGRES_PORT),
            ("Redis", settings.REDIS_HOST, settings.REDIS_PORT),
            ("Airflow", "localhost", 8080),
        ]

        health_results = []

        # Check database and Redis
        for service_name, host, port in services_to_check:
            result = check_service_health(service_name, host, port)
            health_results.append(result)

        # Check API
        api_result = check_api_health()
        health_results.append(api_result)

        progress.update(task, completed=100, total=100)

    # Display results
    table = Table(title="🏥 System Health Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Host:Port", style="blue")
    table.add_column("Response Time", style="green")
    table.add_column("Details", style="yellow")

    overall_healthy = True

    for result in health_results:
        status = result["status"]
        status_color = (
            "green"
            if status == "healthy"
            else "red"
            if status == "unhealthy"
            else "orange"
        )

        if status != "healthy":
            overall_healthy = False

        table.add_row(
            result["service"],
            f"[{status_color}]{status.upper()}[/{status_color}]",
            f"{result['host']}:{result['port']}",
            result.get("response_time", "N/A"),
            result.get("error", result.get("version", "OK")),
        )

    console.print(table)

    # Overall status
    if overall_healthy:
        console.print("\n✅ All systems are healthy!", style="bold green")
    else:
        console.print(
            "\n⚠️  Some systems are unhealthy. Check the details above.",
            style="bold yellow",
        )


@app.command()
def health() -> None:
    """Check overall system health (alias for system command)"""
    system()


@app.command()
def services() -> None:
    """Show detailed service information"""
    logger.info("Displaying service information")

    # Service information
    services_info: List[Dict[str, Union[str, int]]] = [
        {
            "name": "NeuroSync Core API",
            "port": settings.API_PORT,
            "description": "Main API server for NeuroSync",
            "health_endpoint": f"http://localhost:{settings.API_PORT}/health",
        },
        {
            "name": "PostgreSQL Database",
            "port": settings.POSTGRES_PORT,
            "description": "Metadata and configuration storage",
            "health_endpoint": "TCP connection check",
        },
        {
            "name": "Redis Cache",
            "port": settings.REDIS_PORT,
            "description": "Caching and session storage",
            "health_endpoint": "PING command",
        },
        {
            "name": "Apache Airflow",
            "port": 8080,
            "description": "Pipeline orchestration and scheduling",
            "health_endpoint": "http://localhost:8080/health",
        },
        {
            "name": "Vector Store",
            "port": "N/A",
            "description": "FAISS vector database for embeddings",
            "health_endpoint": "File system check",
        },
    ]

    table = Table(title="🔧 NeuroSync Services")
    table.add_column("Service", style="cyan")
    table.add_column("Port", style="green")
    table.add_column("Description", style="blue")
    table.add_column("Health Check", style="yellow")

    for service in services_info:
        table.add_row(
            str(service["name"]),
            str(service["port"]),
            str(service["description"]),
            str(service["health_endpoint"]),
        )

    console.print(table)


@app.command()
def pipelines() -> None:
    """Show pipeline status and statistics"""
    logger.info("Checking pipeline status")

    # Mock pipeline data (will be replaced with real data in later phases)
    pipelines: List[Dict[str, Union[str, int]]] = [
        {
            "name": "document-ingestion",
            "status": "running",
            "last_run": "2 minutes ago",
            "success_rate": "98.5%",
            "total_runs": 247,
            "avg_duration": "3m 45s",
            "next_run": "in 58 minutes",
        },
        {
            "name": "api-data-sync",
            "status": "idle",
            "last_run": "1 hour ago",
            "success_rate": "100%",
            "total_runs": 156,
            "avg_duration": "1m 23s",
            "next_run": "in 2 hours",
        },
        {
            "name": "batch-embedding",
            "status": "failed",
            "last_run": "5 hours ago",
            "success_rate": "92.1%",
            "total_runs": 89,
            "avg_duration": "12m 34s",
            "next_run": "manual trigger required",
        },
    ]

    table = Table(title="📊 Pipeline Status")
    table.add_column("Pipeline", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Last Run", style="green")
    table.add_column("Success Rate", style="blue")
    table.add_column("Total Runs", style="yellow")
    table.add_column("Avg Duration", style="magenta")
    table.add_column("Next Run", style="white")

    for pipeline in pipelines:
        status = str(pipeline["status"])
        status_color = (
            "green" if status == "running" else "blue" if status == "idle" else "red"
        )

        table.add_row(
            str(pipeline["name"]),
            f"[{status_color}]{status.upper()}[/{status_color}]",
            str(pipeline["last_run"]),
            str(pipeline["success_rate"]),
            str(pipeline["total_runs"]),
            str(pipeline["avg_duration"]),
            str(pipeline["next_run"]),
        )

    console.print(table)


@app.command()
def storage() -> None:
    """Show storage usage and statistics"""
    logger.info("Checking storage status")

    import os

    def get_directory_size(path: str) -> int:
        """Get directory size in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, FileNotFoundError):
            pass
        return total_size

    def format_bytes(bytes_value: Union[int, float]) -> str:
        """Format bytes to human readable format"""
        bytes_value = float(bytes_value)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

    # Check storage locations
    storage_locations = [
        ("Data Directory", settings.DATA_DIR),
        ("Vector Store", settings.FAISS_INDEX_PATH),
        ("Upload Directory", settings.UPLOAD_DIR),
        ("Logs Directory", "./logs"),
    ]

    table = Table(title="💾 Storage Usage")
    table.add_column("Location", style="cyan")
    table.add_column("Path", style="blue")
    table.add_column("Size", style="green")
    table.add_column("Status", style="yellow")

    total_size = 0

    for name, path in storage_locations:
        if os.path.exists(path):
            size = get_directory_size(path)
            total_size += size
            size_str = format_bytes(size)
            status = "✅ Available"
        else:
            size_str = "N/A"
            status = "❌ Missing"

        table.add_row(name, path, size_str, status)

    # Add total row
    table.add_row(
        "[bold]Total[/bold]",
        "[bold]All locations[/bold]",
        f"[bold]{format_bytes(total_size)}[/bold]",
        "[bold]Summary[/bold]",
    )

    console.print(table)


@app.command()
def config() -> None:
    """Show current configuration"""
    logger.info("Displaying configuration")

    # Configuration sections
    configs = [
        (
            "Environment",
            [
                ("App Name", settings.APP_NAME),
                ("Version", settings.APP_VERSION),
                ("Environment", settings.ENVIRONMENT),
                ("Debug Mode", str(settings.DEBUG)),
            ],
        ),
        (
            "API Configuration",
            [
                ("Host", settings.API_HOST),
                ("Port", str(settings.API_PORT)),
                ("Workers", str(settings.API_WORKERS)),
            ],
        ),
        (
            "Database",
            [
                ("Host", settings.POSTGRES_HOST),
                ("Port", str(settings.POSTGRES_PORT)),
                ("Database", settings.POSTGRES_DB),
                ("User", settings.POSTGRES_USER),
            ],
        ),
        (
            "Vector Store",
            [
                ("Model", settings.EMBEDDING_MODEL),
                ("Dimension", str(settings.VECTOR_DIMENSION)),
                ("Index Path", settings.FAISS_INDEX_PATH),
                ("Max Chunk Size", str(settings.MAX_CHUNK_SIZE)),
            ],
        ),
        (
            "Logging",
            [
                ("Level", settings.LOG_LEVEL),
                ("Format", settings.LOG_FORMAT),
                ("File Path", settings.LOG_FILE_PATH or "Not set"),
            ],
        ),
    ]

    panels = []
    for section_name, section_config in configs:
        config_text = "\n".join(
            [f"{key}: [green]{value}[/green]" for key, value in section_config]
        )
        panel = Panel(config_text, title=section_name, border_style="blue")
        panels.append(panel)

    console.print(Columns(panels, equal=True, expand=True))


@app.command()
def monitor(
    interval: int = typer.Option(
        5, "--interval", "-i", help="Refresh interval in seconds"
    ),
    duration: int = typer.Option(
        60, "--duration", "-d", help="Monitor duration in seconds"
    ),
) -> None:
    """Real-time system monitoring"""
    logger.info(
        f"Starting real-time monitoring (interval: {interval}s, duration: {duration}s)"
    )

    def generate_status_table() -> Table:
        """Generate current status table"""
        table = Table(
            title=f"🔍 Real-time System Monitor (Updated: {time.strftime('%H:%M:%S')})"
        )
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Response Time", style="green")
        table.add_column("Last Check", style="yellow")

        # Check services
        services = [
            ("API", settings.API_HOST, settings.API_PORT),
            ("PostgreSQL", settings.POSTGRES_HOST, settings.POSTGRES_PORT),
            ("Redis", settings.REDIS_HOST, settings.REDIS_PORT),
        ]

        for service_name, host, port in services:
            start_time = time.time()
            result = check_service_health(service_name, host, port)
            response_time = round((time.time() - start_time) * 1000, 2)

            status = result["status"]
            status_color = "green" if status == "healthy" else "red"

            table.add_row(
                service_name,
                f"[{status_color}]{status.upper()}[/{status_color}]",
                f"{response_time}ms",
                time.strftime("%H:%M:%S"),
            )

        return table

    start_time = time.time()
    with Live(generate_status_table(), refresh_per_second=1 / interval) as live:
        while time.time() - start_time < duration:
            time.sleep(interval)
            live.update(generate_status_table())

    console.print("✅ Monitoring session completed")


@app.command()
def logs(
    service: str = typer.Option(
        "all",
        "--service",
        "-s",
        help="Service name (all, api, postgres, redis, airflow)",
    ),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
) -> None:
    """Show service logs"""
    logger.info(f"Displaying logs for service: {service}")

    if service == "all":
        console.print("📋 Showing logs for all services")
        console.print(
            "Use --service to filter by specific service: api, postgres, redis, airflow"
        )

        # Mock log entries
        log_entries = [
            ("2024-01-15 10:30:25", "API", "INFO", "Server started on port 8000"),
            (
                "2024-01-15 10:30:26",
                "PostgreSQL",
                "INFO",
                "Database connection established",
            ),
            ("2024-01-15 10:30:27", "Redis", "INFO", "Cache server ready"),
            ("2024-01-15 10:30:28", "Airflow", "INFO", "Scheduler started"),
            ("2024-01-15 10:30:30", "API", "INFO", "Health check endpoint registered"),
        ]
    else:
        console.print(f"📋 Showing logs for service: {service}")
        # Mock service-specific logs
        log_entries = [
            (
                "2024-01-15 10:30:25",
                service.upper(),
                "INFO",
                f"{service} service initialized",
            ),
            (
                "2024-01-15 10:30:26",
                service.upper(),
                "DEBUG",
                f"{service} configuration loaded",
            ),
            (
                "2024-01-15 10:30:27",
                service.upper(),
                "INFO",
                f"{service} ready to accept connections",
            ),
        ]

    table = Table(title=f"📄 Service Logs (Last {lines} lines)")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Service", style="blue")
    table.add_column("Level", style="bold")
    table.add_column("Message", style="white")

    for timestamp, srv, level, message in log_entries[-lines:]:
        level_color = (
            "green" if level == "INFO" else "yellow" if level == "DEBUG" else "red"
        )
        table.add_row(
            timestamp, srv, f"[{level_color}]{level}[/{level_color}]", message
        )

    console.print(table)

    if follow:
        console.print("\n👀 Following logs... (Press Ctrl+C to stop)")
        console.print("Note: Live log following will be implemented in later phases")


if __name__ == "__main__":
    app()
