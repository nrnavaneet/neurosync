"""
Pipeline command for NeuroSync CLI
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
