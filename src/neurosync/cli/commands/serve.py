"""
Serve command for starting the NeuroSync API server and interactive chat.

This module provides CLI commands for deploying and interacting with NeuroSync
services. It includes functionality for starting the FastAPI server, configuring
LLM providers, running interactive chat sessions, and performing system health checks.

Key Commands:
    serve: Start the production FastAPI server
    chat: Launch interactive chat interface with RAG capabilities
    config: Configure LLM providers and API keys
    test: Run connection tests and health checks

Features:
    - Production-ready FastAPI server deployment
    - Interactive chat with real-time streaming responses
    - Intelligent LLM provider configuration and fallbacks
    - Comprehensive system health monitoring
    - Environment file management for API keys
    - Vector store integration and testing
    - Rich CLI interface with progress indicators

Dependencies:
    - FastAPI server for REST API endpoints
    - LLM providers (OpenAI, Anthropic, Cohere, etc.)
    - Vector stores (FAISS, Qdrant) for retrieval
    - Embedding models for query processing
    - Rich CLI for enhanced user experience

Example Usage:
    # Start production server
    $ neurosync serve --host 0.0.0.0 --port 8000

    # Interactive configuration
    $ neurosync serve config

    # Start chat session
    $ neurosync serve chat

    # Test system health
    $ neurosync serve test

For deployment configuration and scaling options, see:
    - docs/deployment.md
    - docs/api-reference.md
    - examples/production-setup.py
"""
import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.spinner import Spinner

from neurosync.core.config.settings import Settings
from neurosync.core.logging.logger import get_logger
from neurosync.processing.embedding.manager import EmbeddingManager
from neurosync.serving.llm.manager import LLMManager
from neurosync.serving.rag.retriever import Retriever
from neurosync.storage.vector_store.manager import VectorStoreManager

app = typer.Typer(name="serve", help="Serve NeuroSync API and interactive chat")
console = Console()
logger = get_logger(__name__)


def update_env_file(env_path: Path, key: str, value: str) -> None:
    """
    Update or add key-value pair in environment file.

    Safely updates environment variables in a .env file, either modifying
    existing keys or appending new ones. Preserves existing file structure
    and comments while ensuring clean key-value formatting.

    Args:
        env_path (Path): Path to the .env file
        key (str): Environment variable name (without quotes)
        value (str): Environment variable value (without quotes)

    Behavior:
        - Creates file if it doesn't exist
        - Updates existing key in place if found
        - Appends new key-value pair if not found
        - Preserves file formatting and comments
        - Uses format: KEY=value (no quotes or spaces around =)

    Example:
        >>> from pathlib import Path
        >>> env_path = Path(".env")
        >>> update_env_file(env_path, "OPENAI_API_KEY", "sk-...")
        >>> update_env_file(env_path, "DEBUG", "true")

    Note:
        This function is atomic - it reads the entire file, modifies it,
        and writes it back. For large files or high-concurrency scenarios,
        consider using a more sophisticated approach with file locking.
    """
    lines = []
    key_found = False

    if env_path.exists():
        with open(env_path, "r") as f:
            lines = f.readlines()

    # Update existing key or mark that we need to add it
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            key_found = True
            break

    # Add new key if not found
    if not key_found:
        lines.append(f"{key}={value}\n")

    # Write back to file
    with open(env_path, "w") as f:
        f.writelines(lines)


def setup_llm_providers() -> dict:
    """Interactive setup for LLM providers."""
    console.print("\n[bold blue] LLM Provider Setup[/bold blue]")
    console.print(
        "Configure your preferred LLM providers. "
        "You can skip any provider by pressing Enter."
    )

    providers = {}
    env_path = Path.cwd() / ".env"

    # Create .env if it doesn't exist
    if not env_path.exists():
        env_path.touch()

    provider_configs = [
        {
            "name": "OpenAI",
            "key": "OPENAI_API_KEY",
            "description": "GPT-3.5, GPT-4, and other OpenAI models",
            "url": "https://platform.openai.com/api-keys",
        },
        {
            "name": "Anthropic",
            "key": "ANTHROPIC_API_KEY",
            "description": "Claude models (Claude-3, Claude-2)",
            "url": "https://console.anthropic.com/",
        },
        {
            "name": "Cohere",
            "key": "COHERE_API_KEY",
            "description": "Command models and embeddings",
            "url": "https://dashboard.cohere.ai/api-keys",
        },
        {
            "name": "Google AI",
            "key": "GOOGLE_API_KEY",
            "description": "Gemini models",
            "url": "https://makersuite.google.com/app/apikey",
        },
        {
            "name": "OpenRouter",
            "key": "OPENROUTER_API_KEY",
            "description": "Access to multiple models via one API",
            "url": "https://openrouter.ai/keys",
        },
    ]

    for provider in provider_configs:
        console.print(f"\n[yellow] {provider['name']}[/yellow]")
        console.print(f"   {provider['description']}")
        console.print(f"   Get API key: {provider['url']}")

        # Check if key already exists in environment
        existing_key = os.getenv(provider["key"])
        if existing_key:
            if Confirm.ask("   API key already set. Update it?", default=False):
                api_key = Prompt.ask("   Enter API key", password=True)
                if api_key.strip():
                    providers[provider["name"]] = api_key
                    update_env_file(env_path, provider["key"], api_key)
                    console.print("    Updated!")
            else:
                providers[provider["name"]] = existing_key
                console.print("    Using existing key")
        else:
            api_key = Prompt.ask(
                "   Enter API key (or press Enter to skip)", password=True, default=""
            )
            if api_key.strip():
                providers[provider["name"]] = api_key
                update_env_file(env_path, provider["key"], api_key)
                console.print("    Saved!")
            else:
                console.print("   ⏭  Skipped")

    # Set default model and fallback preferences
    if providers:
        console.print("\n[green] Model Preferences[/green]")

        # Primary model preference
        primary_model = Prompt.ask(
            "Preferred primary model",
            choices=[
                "gpt-4",
                "gpt-3.5-turbo",
                "claude-3-sonnet",
                "claude-3-haiku",
                "command",
                "gemini-pro",
            ],
            default="gpt-3.5-turbo",
        )
        update_env_file(env_path, "DEFAULT_LLM_MODEL", primary_model)

        # Fallback enabled
        enable_fallback = Confirm.ask(
            "Enable automatic fallback between providers?", default=True
        )
        update_env_file(
            env_path, "LLM_ENABLE_FALLBACK", "true" if enable_fallback else "false"
        )

        console.print("\n[green] LLM providers configured successfully![/green]")
        return providers
    else:
        console.print(
            "\n[yellow]  No LLM providers configured. "
            "Some features may not work.[/yellow]"
        )
        return {}


async def test_llm_connection(settings: Settings) -> bool:
    """Test LLM connection."""
    try:
        logger.info("Creating LLM manager...")
        llm_manager = LLMManager(settings)
        logger.info("LLM manager created, testing generation...")
        response = await llm_manager.generate("Hello", max_tokens=5)
        logger.info(f"LLM test response: {response}")
        return bool(response)
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


async def test_vector_store_connection(settings: Settings) -> bool:
    """Test vector store connection."""
    try:
        # Convert Settings to dict for VectorStoreManager
        config = {
            "type": "faiss",
            "path": getattr(settings, "FAISS_INDEX_PATH", "/app/data/vector_store"),
            "dimension": getattr(settings, "VECTOR_DIMENSION", 384),
        }
        vector_store_manager = VectorStoreManager(config)
        # Just check if we can get store info
        _ = vector_store_manager.get_info()
        return True
    except Exception as e:
        logger.error(f"Vector store connection test failed: {e}")
        return False


async def interactive_chat(settings: Settings) -> None:
    """Start interactive chat session."""
    console.print("\n[bold blue] Interactive Chat Mode[/bold blue]")
    console.print("Type 'quit', 'exit', or 'q' to stop chatting")
    console.print("Type '/help' for commands\n")

    # Initialize components
    try:
        with console.status("[bold green]Initializing chat components..."):
            llm_manager = LLMManager(settings)

            # Create embedding manager config
            embedding_config = {
                "type": "huggingface",
                "model_name": "all-MiniLM-L6-v2",
                "enable_monitoring": True,
            }
            embedding_manager = EmbeddingManager(embedding_config)

            # Convert Settings to dict for VectorStoreManager
            config = {
                "type": "faiss",
                "path": getattr(settings, "FAISS_INDEX_PATH", "/app/data/vector_store"),
                "dimension": getattr(settings, "VECTOR_DIMENSION", 384),
            }
            vector_store_manager = VectorStoreManager(config)
            retriever = Retriever(
                embedding_manager=embedding_manager,
                vector_store_manager=vector_store_manager,
            )

        console.print("[green]Chat ready![/green]\n")

        while True:
            try:
                # Get user input
                user_input = Prompt.ask("[bold blue]You[/bold blue]")

                if user_input.lower() in ["quit", "exit", "q"]:
                    console.print("[yellow] Goodbye![/yellow]")
                    break

                if user_input == "/help":
                    console.print(
                        Panel(
                            """[bold]Available Commands:[/bold]
• /help - Show this help
• /models - List available models
• /stats - Show system stats
• quit/exit/q - Exit chat""",
                            title="Help",
                            border_style="blue",
                        )
                    )
                    continue

                if user_input == "/models":
                    models = llm_manager.get_available_models()
                    console.print(
                        f"[green]Available models:[/green] {', '.join(models)}"
                    )
                    continue

                if user_input == "/stats":
                    console.print(
                        f"[green]Current model:[/green] {llm_manager.current_model}"
                    )
                    continue

                # Process chat request
                with Live(Spinner("dots", text="Thinking..."), refresh_per_second=10):
                    # Retrieve relevant documents
                    retrieved_docs = retriever.retrieve(query=user_input, top_k=3)

                    # Build context
                    context_parts = []
                    for doc in retrieved_docs:
                        # Get content from metadata (stored in 'text' field)
                        content = doc.metadata.get("text", "No content available")
                        context_parts.append(f"Content: {content}")

                        # Add source info if available
                        source = doc.metadata.get("source_id", "Unknown source")
                        context_parts.append(f"Source: {source}")
                        context_parts.append("---")

                    context = "\n".join(context_parts)

                    # Build prompt
                    prompt = (
                        "Based on the following context, answer the user's question. "
                        "If the answer cannot be found in the context, "
                        "say so clearly.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {user_input}\n\n"
                        "Answer:"
                    )

                    # Generate response
                    response = await llm_manager.generate(prompt=prompt, max_tokens=500)

                if response:
                    # Display response
                    console.print(f"\n[bold green]Assistant[/bold green]: {response}")

                    # Show sources if available
                    if retrieved_docs:
                        console.print(
                            f"\n[dim]Sources: {len(retrieved_docs)} "
                            "documents used[/dim]"
                        )
                else:
                    console.print("[red]Failed to generate response[/red]")

                console.print()  # Add spacing

            except KeyboardInterrupt:
                console.print("\n[yellow] Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")

    except Exception as e:
        console.print(f"[red]Failed to initialize chat: {str(e)}[/red]")


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    setup: bool = typer.Option(True, help="Run interactive setup"),
):
    """Start the NeuroSync API server."""
    console.print(
        Panel(
            "[bold blue]NeuroSync API Server[/bold blue]",
            subtitle="Starting FastAPI server for document processing and chat",
        )
    )

    # Run setup if requested
    if setup:
        providers = setup_llm_providers()
        if not providers:
            if not Confirm.ask("Continue without LLM providers?", default=False):
                console.print("[yellow]Setup cancelled.[/yellow]")
                raise typer.Exit(1)

    # Test connections
    console.print("\n[bold blue] Testing connections...[/bold blue]")
    settings = Settings()

    async def test_connections():
        llm_ok = await test_llm_connection(settings)
        vector_ok = await test_vector_store_connection(settings)

        console.print(f"   LLM: {' Ready' if llm_ok else ' Failed'}")
        console.print(f"   Vector Store: {' Ready' if vector_ok else ' Failed'}")

        if not vector_ok:
            console.print(
                "[yellow]  Vector store not ready. "
                "Run 'neurosync pipeline run' first.[/yellow]"
            )

    asyncio.run(test_connections())

    # Start server
    console.print(f"\n[green] Starting server at http://{host}:{port}[/green]")
    console.print(f" API docs: http://{host}:{port}/docs")

    try:
        import uvicorn

        uvicorn.run(
            "neurosync.serving.api.server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except ImportError:
        console.print("[red] uvicorn not installed. Run: pip install uvicorn[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow] Server stopped[/yellow]")


@app.command()
def chat(
    setup: bool = typer.Option(True, help="Run interactive setup"),
):
    """Start interactive chat session."""
    console.print(
        Panel(
            "[bold blue] NeuroSync Interactive Chat[/bold blue]",
            subtitle="Chat with your documents using AI",
        )
    )

    # Run setup if requested
    if setup:
        providers = setup_llm_providers()
        if not providers:
            console.print("[red] No LLM providers configured. Cannot start chat.[/red]")
            raise typer.Exit(1)

    # Load settings and start chat
    settings = Settings()
    asyncio.run(interactive_chat(settings))


@app.command()
def config():
    """Configure LLM providers and settings."""
    console.print(
        Panel(
            "[bold blue]  NeuroSync Configuration[/bold blue]",
            subtitle="Set up LLM providers and preferences",
        )
    )

    setup_llm_providers()
    console.print("\n[green] Configuration complete![/green]")


@app.command()
def test():
    """Test LLM and vector store connections."""
    console.print(
        Panel(
            "[bold blue] Connection Tests[/bold blue]",
            subtitle="Verify that all services are working",
        )
    )

    settings = Settings()

    async def run_tests():
        console.print("\n[blue]Testing LLM connection...[/blue]")
        llm_ok = await test_llm_connection(settings)
        console.print(f"   {' LLM Ready' if llm_ok else ' LLM Failed'}")

        console.print("\n[blue]Testing vector store connection...[/blue]")
        vector_ok = await test_vector_store_connection(settings)
        console.print(
            f"   {' Vector Store Ready' if vector_ok else ' Vector Store Failed'}"
        )

        if llm_ok and vector_ok:
            console.print("\n[green] All systems ready![/green]")
        else:
            console.print("\n[yellow]  Some systems need attention.[/yellow]")
            if not vector_ok:
                console.print("   Run: neurosync pipeline run")

    asyncio.run(run_tests())


if __name__ == "__main__":
    app()
