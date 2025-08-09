"""
NeuroSync Full Pipeline - End-to-End Data Processing Pipeline.

This module implements the complete NeuroSync pipeline that orchestrates
the entire data processing workflow from ingestion to chat-ready deployment.
It provides a unified interface for running all pipeline phases with
intelligent configuration selection and progress monitoring.

Pipeline Phases:
    1. Ingestion: Extract data from various sources (files, APIs, databases)
    2. Processing: Clean, normalize, and chunk text content
    3. Embedding: Generate vector representations using embedding models
    4. Storage: Store vectors in optimized vector databases
    5. Serving: Deploy LLM-powered chat interface with retrieval

Key Features:
    - Intelligent auto-configuration based on data type and API keys
    - Interactive template selection with rich CLI interface
    - Real-time progress monitoring with detailed phase timing
    - Comprehensive error handling and recovery mechanisms
    - Production-ready deployment with monitoring capabilities
    - Extensible architecture supporting custom configurations

Classes:
    FullPipeline: Main pipeline orchestrator with end-to-end processing

The pipeline supports multiple execution modes:
    - Auto mode: Intelligent configuration selection
    - Interactive mode: User-guided template selection
    - Custom mode: User-provided configuration files
    - Batch mode: Silent execution for automation

Example:
    >>> from neurosync.pipelines.pipeline import FullPipeline
    >>> pipeline = FullPipeline()
    >>> # Auto mode with intelligent defaults
    >>> await pipeline.run_full_pipeline("/path/to/data", auto=True)
    >>> # Interactive mode with user selection
    >>> await pipeline.run_full_pipeline("/path/to/data", interactive=True)

For detailed configuration options and advanced usage, see:
    - docs/pipeline-configuration.md
    - docs/template-system.md
    - examples/pipeline-usage.py
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from neurosync.core.logging.logger import get_logger
from neurosync.ingestion.manager import IngestionManager
from neurosync.pipelines.embedding_pipeline import EmbeddingPipeline
from neurosync.processing.manager import ProcessingManager
from neurosync.serving.llm.manager import LLMManager

logger = get_logger(__name__)
console = Console()


class FullPipeline:
    """
    Comprehensive end-to-end pipeline orchestrator.

    The FullPipeline class coordinates all phases of the NeuroSync data
    processing workflow, from initial data ingestion through to deployment
    of a chat-ready LLM interface. It provides intelligent configuration
    management, progress monitoring, and error handling throughout the
    entire pipeline execution.

    Key Capabilities:
        - Automatic data type detection and configuration selection
        - Interactive template selection with rich CLI interface
        - Real-time progress tracking with detailed timing metrics
        - Intelligent fallback and error recovery mechanisms
        - Multi-phase validation and quality assurance
        - Production deployment with monitoring integration

    Pipeline Phases:
        1. Configuration: Select optimal templates based on input and API keys
        2. Ingestion: Extract and normalize data from various sources
        3. Processing: Clean, chunk, and prepare text for embedding
        4. Embedding: Generate vector representations using selected models
        5. Storage: Index vectors in optimized database systems
        6. Serving: Deploy LLM interface with retrieval capabilities

    Attributes:
        console (Console): Rich console for formatted output
        logger (Logger): Structured logging instance
        pipeline_start_time (Optional[float]): Pipeline execution start time
        phase_timings (Dict[str, float]): Timing data for each phase
        templates (Dict[str, Any]): Available configuration templates

    The pipeline supports flexible execution modes:
        - Auto: Intelligent defaults with minimal user interaction
        - Interactive: Guided configuration with user selections
        - Custom: User-provided configuration files
        - Batch: Silent execution suitable for automation

    Example:
        >>> pipeline = FullPipeline()
        >>> # Quick auto-configuration
        >>> await pipeline.run_full_pipeline("/data", auto=True)
        >>>
        >>> # Interactive configuration
        >>> await pipeline.run_full_pipeline("/data", interactive=True)
        >>>
        >>> # Custom configuration
        >>> config = {...}  # Custom configuration dict
        >>> await pipeline.run_full_pipeline("/data", config=config)
    """

    def __init__(self):
        self.console = console
        self.logger = logger
        self.pipeline_start_time = None
        self.phase_timings = {}

        # Configuration templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Any]:
        """Load all available configuration templates."""
        return {
            "ingestion": {
                "file_basic": {
                    "name": "Basic File Ingestion",
                    "description": "Ingest files from local directory",
                    "config": {
                        "sources": [
                            {
                                "name": "local_files",
                                "type": "file",
                                "config": {
                                    "base_path": "{input_path}",
                                    "file_patterns": [
                                        "*.txt",
                                        "*.md",
                                        "*.pdf",
                                        "*.docx",
                                        "*.json",
                                    ],
                                    "recursive": True,
                                    "batch_size": 10,
                                },
                            }
                        ]
                    },
                },
                "file_advanced": {
                    "name": "Advanced File Processing",
                    "description": "Enhanced file processing with filters",
                    "config": {
                        "sources": [
                            {
                                "name": "enhanced_files",
                                "type": "file",
                                "config": {
                                    "base_path": "{input_path}",
                                    "file_patterns": [
                                        "*.txt",
                                        "*.md",
                                        "*.pdf",
                                        "*.docx",
                                        "*.json",
                                        "*.csv",
                                        "*.html",
                                    ],
                                    "recursive": True,
                                    "batch_size": 20,
                                    "max_file_size": 10485760,  # 10MB
                                    "exclude_patterns": ["*.tmp", "*.log"],
                                },
                            }
                        ]
                    },
                },
                "api_basic": {
                    "name": "REST API Ingestion",
                    "description": "Ingest data from REST APIs",
                    "config": {
                        "sources": [
                            {
                                "name": "api_source",
                                "type": "api",
                                "config": {
                                    "base_url": "{input_path}",
                                    "endpoints": ["/api/data", "/api/documents"],
                                    "rate_limit": 10,
                                    "timeout": 30,
                                    "headers": {"Content-Type": "application/json"},
                                },
                            }
                        ]
                    },
                },
                "database_postgres": {
                    "name": "PostgreSQL Database",
                    "description": "Ingest from PostgreSQL database",
                    "config": {
                        "sources": [
                            {
                                "name": "postgres_source",
                                "type": "database",
                                "config": {
                                    "database_type": "postgresql",
                                    "connection_string": "{input_path}",
                                    "tables": ["documents", "articles", "content"],
                                    "query": (
                                        "SELECT * FROM {table} WHERE "
                                        "created_at > NOW() - INTERVAL '30 days'"
                                    ),
                                },
                            }
                        ]
                    },
                },
                "database_mysql": {
                    "name": "MySQL Database",
                    "description": "Ingest from MySQL database",
                    "config": {
                        "sources": [
                            {
                                "name": "mysql_source",
                                "type": "database",
                                "config": {
                                    "database_type": "mysql",
                                    "connection_string": "{input_path}",
                                    "tables": ["documents", "articles", "content"],
                                    "batch_size": 1000,
                                },
                            }
                        ]
                    },
                },
                "multi_source": {
                    "name": "Multi-Source Ingestion",
                    "description": "Combine multiple data sources",
                    "config": {
                        "sources": [
                            {
                                "name": "local_files",
                                "type": "file",
                                "config": {
                                    "base_path": "{input_path}",
                                    "file_patterns": ["*.txt", "*.md", "*.pdf"],
                                    "recursive": True,
                                },
                            },
                            {
                                "name": "api_data",
                                "type": "api",
                                "config": {
                                    "base_url": "https://api.example.com",
                                    "endpoints": ["/documents"],
                                    "rate_limit": 5,
                                },
                            },
                        ]
                    },
                },
            },
            "processing": {
                "recursive": {
                    "name": "Recursive Chunking",
                    "description": (
                        "Simple recursive text splitting with configurable separators"
                    ),
                    "config": {
                        "preprocessing": [
                            {"name": "whitespace_normalizer", "enabled": True}
                        ],
                        "chunking": {
                            "strategy": "recursive",
                            "chunk_size": 1000,
                            "chunk_overlap": 200,
                            "separators": ["\n\n", "\n", ". ", " "],
                        },
                        "filtering": {"min_quality_score": 0.1},
                    },
                },
                "semantic": {
                    "name": "Semantic Chunking",
                    "description": (
                        "Semantic-aware chunking using NLP models for better "
                        "context preservation"
                    ),
                    "config": {
                        "preprocessing": [
                            {"name": "html_cleaner", "enabled": True},
                            {"name": "whitespace_normalizer", "enabled": True},
                        ],
                        "chunking": {
                            "strategy": "semantic",
                            "chunk_size": 1500,
                            "chunk_overlap": 300,
                            "model": "en_core_web_sm",
                        },
                        "filtering": {"min_quality_score": 0.3},
                    },
                },
                "sliding_window": {
                    "name": "Sliding Window Chunking",
                    "description": (
                        "Fixed-size overlapping windows for consistent chunk sizes"
                    ),
                    "config": {
                        "preprocessing": [
                            {"name": "whitespace_normalizer", "enabled": True}
                        ],
                        "chunking": {
                            "strategy": "sliding_window",
                            "chunk_size": 1000,
                            "chunk_overlap": 200,
                        },
                        "filtering": {"min_quality_score": 0.1},
                    },
                },
                "token_aware_sliding": {
                    "name": "Token-Aware Sliding Window",
                    "description": (
                        "Sliding window chunking with token counting for "
                        "precise control"
                    ),
                    "config": {
                        "preprocessing": [
                            {"name": "whitespace_normalizer", "enabled": True}
                        ],
                        "chunking": {
                            "strategy": "token_aware_sliding",
                            "chunk_size": 1000,
                            "chunk_overlap": 200,
                        },
                        "filtering": {"min_quality_score": 0.1},
                    },
                },
                "hierarchical": {
                    "name": "Hierarchical Chunking",
                    "description": (
                        "Respect document hierarchy and structure for better context"
                    ),
                    "config": {
                        "preprocessing": [
                            {"name": "html_cleaner", "enabled": True},
                            {"name": "whitespace_normalizer", "enabled": True},
                        ],
                        "chunking": {
                            "strategy": "hierarchical",
                            "chunk_size": 1200,
                            "chunk_overlap": 150,
                            "preserve_structure": True,
                            "min_section_size": 100,
                        },
                        "filtering": {"min_quality_score": 0.2},
                    },
                },
                "document_structure": {
                    "name": "Document Structure-Aware",
                    "description": (
                        "Advanced document analysis with OCR and table extraction"
                    ),
                    "config": {
                        "preprocessing": [
                            {"name": "html_cleaner", "enabled": True},
                            {"name": "whitespace_normalizer", "enabled": True},
                        ],
                        "chunking": {
                            "strategy": "document_structure",
                            "chunk_size": 1200,
                            "chunk_overlap": 150,
                            "ocr_enabled": True,
                            "table_extraction": True,
                            "preserve_table_structure": True,
                            "min_confidence": 0.7,
                        },
                        "filtering": {"min_quality_score": 0.2},
                    },
                },
                "code_aware": {
                    "name": "Code-Aware Processing",
                    "description": (
                        "Specialized processing for source code files with "
                        "function/class boundaries"
                    ),
                    "config": {
                        "preprocessing": [
                            {"name": "whitespace_normalizer", "enabled": True}
                        ],
                        "chunking": {
                            "strategy": "recursive",
                            "chunk_size": 2000,
                            "chunk_overlap": 100,
                            "separators": [
                                "\n\nclass ",
                                "\n\ndef ",
                                "\n\nfunction ",
                                "\n\n",
                                "\n",
                                " ",
                            ],
                        },
                        "filtering": {"min_quality_score": 0.1},
                    },
                },
                "advanced": {
                    "name": "Advanced Multi-Strategy",
                    "description": (
                        "Adaptive processing that chooses the best chunking "
                        "strategy per document"
                    ),
                    "config": {
                        "preprocessing": [
                            {"name": "html_cleaner", "enabled": True},
                            {"name": "whitespace_normalizer", "enabled": True},
                        ],
                        "chunking": {
                            "strategy": "semantic",
                            "chunk_size": 1500,
                            "chunk_overlap": 300,
                            "model": "en_core_web_sm",
                        },
                        "filtering": {"min_quality_score": 0.3},
                    },
                },
            },
            "embedding": {
                "huggingface_fast": {
                    "name": "HuggingFace - Fast",
                    "description": "Fast, lightweight embeddings",
                    "config": {
                        "type": "huggingface",
                        "model_name": "all-MiniLM-L6-v2",
                        "enable_monitoring": True,
                        "batch_size": 64,
                    },
                },
                "huggingface_quality": {
                    "name": "HuggingFace - High Quality",
                    "description": "High-quality embeddings for better accuracy",
                    "config": {
                        "type": "huggingface",
                        "model_name": "all-mpnet-base-v2",
                        "enable_monitoring": True,
                        "batch_size": 32,
                    },
                },
                "openai": {
                    "name": "OpenAI Embeddings",
                    "description": "OpenAI's text-embedding models",
                    "config": {
                        "type": "openai",
                        "model_name": "text-embedding-3-small",
                        "api_key": "{openai_api_key}",
                        "max_batch_size": 2048,
                        "enable_monitoring": True,
                    },
                },
                "openai_large": {
                    "name": "OpenAI Large Embeddings",
                    "description": "OpenAI's large embedding model for maximum quality",
                    "config": {
                        "type": "openai",
                        "model_name": "text-embedding-3-large",
                        "api_key": "{openai_api_key}",
                        "max_batch_size": 2048,
                        "enable_monitoring": True,
                    },
                },
            },
            "vector_store": {
                "faiss_flat": {
                    "name": "FAISS Flat Index",
                    "description": (
                        "Simple, fast exact search - best for small to medium datasets"
                    ),
                    "config": {
                        "type": "faiss",
                        "path": "./vector_store",
                        "index_type": "flat",
                        "enable_versioning": True,
                    },
                },
                "faiss_hnsw": {
                    "name": "FAISS HNSW Index",
                    "description": (
                        "Hierarchical NSW for large datasets - excellent "
                        "speed/accuracy balance"
                    ),
                    "config": {
                        "type": "faiss",
                        "path": "./vector_store",
                        "index_type": "hnsw",
                        "index_params": {
                            "M": 16,
                            "ef_construction": 200,
                            "ef_search": 100,
                        },
                        "enable_versioning": True,
                    },
                },
                "faiss_ivf_flat": {
                    "name": "FAISS IVF Flat Index",
                    "description": (
                        "Inverted file index - good for very large datasets "
                        "with reasonable accuracy"
                    ),
                    "config": {
                        "type": "faiss",
                        "path": "./vector_store",
                        "index_type": "ivf",
                        "index_params": {"nlist": 100, "nprobe": 10},
                        "enable_versioning": True,
                    },
                },
                "faiss_ivf_pq": {
                    "name": "FAISS IVF-PQ Index",
                    "description": (
                        "Product quantization - memory efficient for massive datasets"
                    ),
                    "config": {
                        "type": "faiss",
                        "path": "./vector_store",
                        "index_type": "ivf_pq",
                        "index_params": {
                            "nlist": 100,
                            "nprobe": 10,
                            "m": 8,
                            "nbits": 8,
                        },
                        "enable_versioning": True,
                    },
                },
                "qdrant_local": {
                    "name": "Qdrant Local",
                    "description": (
                        "Local Qdrant instance - production-ready vector database"
                    ),
                    "config": {
                        "type": "qdrant",
                        "host": "localhost",
                        "port": 6333,
                        "collection_name": "neurosync_vectors",
                        "distance_metric": "cosine",
                        "enable_versioning": False,
                    },
                },
                "qdrant_cloud": {
                    "name": "Qdrant Cloud",
                    "description": "Qdrant cloud service - managed vector database",
                    "config": {
                        "type": "qdrant",
                        "url": "https://your-cluster.qdrant.io",
                        "api_key": "{qdrant_api_key}",
                        "collection_name": "neurosync_vectors",
                        "distance_metric": "cosine",
                        "enable_versioning": False,
                    },
                },
            },
            "llm": {
                "openai": {
                    "name": "OpenAI GPT Models",
                    "description": "OpenAI's GPT-4o and GPT-4o-mini models",
                    "config": {
                        "providers": ["openai"],
                        "default_provider": "openai",
                        "fallback_enabled": False,
                        "openai": {
                            "api_key": "{openai_api_key}",
                            "model": "gpt-4o-mini",
                            "temperature": 0.7,
                            "max_tokens": 2048,
                        },
                    },
                },
                "anthropic": {
                    "name": "Anthropic Claude",
                    "description": "Anthropic's Claude-3 models",
                    "config": {
                        "providers": ["anthropic"],
                        "default_provider": "anthropic",
                        "fallback_enabled": False,
                        "anthropic": {
                            "api_key": "{anthropic_api_key}",
                            "model": "claude-3-haiku-20240307",
                            "temperature": 0.7,
                            "max_tokens": 2048,
                        },
                    },
                },
                "cohere": {
                    "name": "Cohere Command Models",
                    "description": "Cohere's Command-R models",
                    "config": {
                        "providers": ["cohere"],
                        "default_provider": "cohere",
                        "fallback_enabled": False,
                        "cohere": {
                            "api_key": "{cohere_api_key}",
                            "model": "command-r",
                            "temperature": 0.7,
                            "max_tokens": 2048,
                        },
                    },
                },
                "google": {
                    "name": "Google Gemini",
                    "description": "Google's Gemini models",
                    "config": {
                        "providers": ["google"],
                        "default_provider": "google",
                        "fallback_enabled": False,
                        "google": {
                            "api_key": "{google_api_key}",
                            "model": "gemini-pro",
                            "temperature": 0.7,
                            "max_tokens": 2048,
                        },
                    },
                },
                "openrouter": {
                    "name": "OpenRouter (Multi-Model Access)",
                    "description": "Access to multiple models through OpenRouter",
                    "config": {
                        "providers": ["openrouter"],
                        "default_provider": "openrouter",
                        "fallback_enabled": False,
                        "openrouter": {
                            "api_key": "{openrouter_api_key}",
                            "model": "anthropic/claude-3-haiku",
                            "temperature": 0.7,
                            "max_tokens": 2048,
                        },
                    },
                },
                "multi_provider": {
                    "name": "Multi-Provider with Fallback",
                    "description": "Multiple LLM providers with intelligent fallback",
                    "config": {
                        "providers": ["openai", "anthropic", "cohere"],
                        "default_provider": "openai",
                        "fallback_enabled": True,
                        "openai": {
                            "api_key": "{openai_api_key}",
                            "model": "gpt-4o-mini",
                            "temperature": 0.7,
                            "max_tokens": 2048,
                        },
                        "anthropic": {
                            "api_key": "{anthropic_api_key}",
                            "model": "claude-3-haiku-20240307",
                            "temperature": 0.7,
                            "max_tokens": 2048,
                        },
                        "cohere": {
                            "api_key": "{cohere_api_key}",
                            "model": "command-r",
                            "temperature": 0.7,
                            "max_tokens": 2048,
                        },
                    },
                },
                "enterprise": {
                    "name": "Enterprise Multi-Provider",
                    "description": (
                        "Full enterprise setup with all providers and fallback"
                    ),
                    "config": {
                        "providers": [
                            "openai",
                            "anthropic",
                            "cohere",
                            "google",
                            "openrouter",
                        ],
                        "default_provider": "openai",
                        "fallback_enabled": True,
                        "openai": {
                            "api_key": "{openai_api_key}",
                            "model": "gpt-4o",
                            "temperature": 0.7,
                            "max_tokens": 4096,
                        },
                        "anthropic": {
                            "api_key": "{anthropic_api_key}",
                            "model": "claude-3-sonnet-20240229",
                            "temperature": 0.7,
                            "max_tokens": 4096,
                        },
                        "cohere": {
                            "api_key": "{cohere_api_key}",
                            "model": "command-r-plus",
                            "temperature": 0.7,
                            "max_tokens": 4096,
                        },
                        "google": {
                            "api_key": "{google_api_key}",
                            "model": "gemini-pro",
                            "temperature": 0.7,
                            "max_tokens": 4096,
                        },
                        "openrouter": {
                            "api_key": "{openrouter_api_key}",
                            "model": "anthropic/claude-3-opus",
                            "temperature": 0.7,
                            "max_tokens": 4096,
                        },
                    },
                },
            },
        }

    def show_welcome(self):
        """Display welcome message and pipeline overview."""
        welcome_panel = Panel(
            Text.from_markup(
                "[bold cyan]NeuroSync Full Pipeline[/bold cyan]\n\n"
                "[bold]End-to-End Data Processing Pipeline[/bold]\n\n"
                "This pipeline will guide you through:\n"
                "• [green]Phase 1:[/green] Data Ingestion (files, APIs, databases)\n"
                "• [green]Phase 2:[/green] Intelligent Processing & Chunking "
                "(multiple strategies)\n"
                "• [green]Phase 3:[/green] Embedding Generation (multiple models)\n"
                "• [green]Phase 4:[/green] Vector Store Creation "
                "(FAISS, Qdrant options)\n"
                "• [green]Phase 5:[/green] LLM Integration & Chat\n\n"
                "[yellow]Choose from multiple chunking strategies and "
                "vector store options![/yellow]"
            ),
            title="Welcome to NeuroSync",
            border_style="bright_blue",
            padding=(1, 2),
        )
        console.print(welcome_panel)
        console.print()

    def detect_input_type(self, input_path: str) -> Tuple[str, str]:
        """Detect the input type and suggest appropriate template."""
        path = Path(input_path)

        # Check if it's a file or directory
        if path.exists():
            if path.is_file():
                return "file", "file_basic"
            elif path.is_dir():
                return "file", "file_advanced"

        # Check if it's a URL
        if input_path.startswith(("http://", "https://")):
            return "api", "api_basic"

        # Check if it's a database connection string
        if any(
            db in input_path.lower()
            for db in ["postgresql://", "mysql://", "postgres://", "sqlite://"]
        ):
            if "postgresql://" in input_path or "postgres://" in input_path:
                return "database", "database_postgres"
            elif "mysql://" in input_path:
                return "database", "database_mysql"

        # Default to file
        return "file", "file_basic"

    def show_template_selection(
        self, phase: str, detected_type: Optional[str] = None
    ) -> str:
        """Show template selection interface for a specific phase."""
        templates = self.templates[phase]

        console.print(f"\n[bold cyan]Phase: {phase.title()} Configuration[/bold cyan]")

        table = Table(title=f"Available {phase.title()} Templates")
        table.add_column("ID", style="cyan", width=6)
        table.add_column("Template", style="green")
        table.add_column("Description", style="white")

        template_keys = list(templates.keys())
        for i, key in enumerate(template_keys, 1):
            template = templates[key]
            marker = "⭐" if key == detected_type else ""
            table.add_row(f"{i}{marker}", template["name"], template["description"])

        console.print(table)

        if detected_type:
            console.print("\n[yellow]Recommended template based on your input[/yellow]")

        while True:
            try:
                choice = Prompt.ask(
                    f"\nSelect {phase} template",
                    default=(
                        "1"
                        if not detected_type
                        else str(template_keys.index(detected_type) + 1)
                    ),
                )

                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(template_keys):
                        return template_keys[idx]

                console.print("[red]Invalid selection. Please try again.[/red]")
            except KeyboardInterrupt:
                raise typer.Exit(1)

    def collect_api_keys(self, template_configs: Dict[str, Any]) -> Dict[str, str]:
        """Collect necessary API keys from environment or user."""
        import os

        # Try to load .env file if available
        try:
            from dotenv import load_dotenv

            # Look for .env file in current directory and parent directories
            for env_file in [".env", "../.env", "../../.env"]:
                if os.path.exists(env_file):
                    load_dotenv(env_file)
                    console.print(f"[dim]Loaded environment from {env_file}[/dim]")
                    break
        except ImportError:
            # python-dotenv not available, continue with system env vars only
            pass

        api_keys: Dict[str, str] = {}

        # Check if API keys are needed based on template configurations
        embedding_config = template_configs.get("embedding", {}).get("config", {})
        llm_config = template_configs.get("llm", {}).get("config", {})

        # Check which providers are needed
        providers_needed = set()
        if embedding_config.get("type") == "openai":
            providers_needed.add("openai")

        llm_providers = llm_config.get("providers", [])
        providers_needed.update(llm_providers)

        # If no providers need API keys, return empty
        if not providers_needed:
            console.print(
                "[green]No API keys required for selected configuration[/green]"
            )
            return api_keys

        # Environment variable mappings
        env_var_map = {
            "openai": ["OPENAI_API_KEY", "OPENAI_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_KEY", "CLAUDE_API_KEY"],
            "cohere": ["COHERE_API_KEY", "COHERE_KEY"],
            "google": ["GOOGLE_API_KEY", "GOOGLE_KEY", "GEMINI_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY", "OPENROUTER_KEY"],
        }

        # Placeholder patterns to ignore
        placeholder_patterns = [
            "your-",
            "INSERT_",
            "ADD_",
            "REPLACE_",
            "CHANGE_",
            "PUT_",
            "ENTER_",
            "sk-fake",
            "fake-",
            "test-",
            "demo-",
        ]

        def is_valid_api_key(value: str, provider: str) -> bool:
            """Check if API key is valid (not a placeholder)."""
            if not value or not value.strip():
                return False

            value_lower = value.lower()

            # Check for placeholder patterns
            for pattern in placeholder_patterns:
                if pattern in value_lower:
                    return False

            # Relaxed validation - allow any reasonable API key
            return len(value.strip()) > 5  # Minimum reasonable length

        # First, scan for available API keys in environment
        available_keys = {}
        for provider in providers_needed:
            possible_vars = env_var_map.get(provider, [])
            for var_name in possible_vars:
                env_value = os.getenv(var_name)
                if env_value and is_valid_api_key(env_value, provider):
                    available_keys[provider] = {
                        "key": env_value.strip(),
                        "source": var_name,
                    }
                    break

        # Show available keys and ask user what they want to do
        if available_keys:
            console.print(
                f"\n[green]Found {len(available_keys)} API key(s) in "
                f"environment:[/green]"
            )
            for provider, info in available_keys.items():
                masked_key = (
                    info["key"][:20] + "..." if len(info["key"]) > 20 else info["key"]
                )
                console.print(
                    f" {provider.title()}: {masked_key} (from {info['source']})"
                )

            use_found_keys = Confirm.ask("\n� Use the found API key(s)?", default=True)
            if use_found_keys:
                for provider, info in available_keys.items():
                    api_keys[f"{provider}_api_key"] = info["key"]
                    console.print(
                        f"[green]Using {provider} API key from environment[/green]"
                    )

        # Check which providers still need keys
        missing_providers = [
            p for p in providers_needed if f"{p}_api_key" not in api_keys
        ]

        if missing_providers:
            console.print("\n[yellow]The following providers need API keys:[/yellow]")
            for provider in missing_providers:
                console.print(f"  {provider.title()}")

            # Ask if user wants to continue without all keys
            continue_without_keys = Confirm.ask(
                "\nContinue without setting up all API keys? (You can add them later)",
                default=True,
            )

            if not continue_without_keys:
                # User wants to provide keys now
                for provider in missing_providers:
                    console.print(
                        f"\n[yellow]{provider.title()} API Key Setup[/yellow]"
                    )

                    # Allow skipping individual providers
                    setup_provider = Confirm.ask(
                        f"Set up {provider.title()} API key now?", default=False
                    )

                    if setup_provider:
                        while True:
                            api_key = Prompt.ask(
                                f"Enter your {provider.title()} API key "
                                f"(or 'skip' to skip)",
                                password=True,
                            )

                            if api_key.lower() == "skip":
                                console.print(
                                    f"[yellow]Skipped {provider.title()} "
                                    f"API key[/yellow]"
                                )
                                break

                            if api_key.strip() and len(api_key.strip()) > 3:
                                api_keys[f"{provider}_api_key"] = api_key.strip()
                                console.print(
                                    f"[green]{provider.title()} API key saved[/green]"
                                )

                                # Ask if they want to save to .env file
                                save_to_env = Confirm.ask(
                                    f"Save {provider.title()} API key to .env file?",
                                    default=True,
                                )
                                if save_to_env:
                                    self._save_api_key_to_env(provider, api_key.strip())
                                break

                            console.print(
                                "[red]API key must be at least 4 characters. "
                                "Enter a valid key or 'skip'.[/red]"
                            )

        # Show final summary
        if api_keys:
            console.print(
                f"\n[green]Ready to proceed with {len(api_keys)} API key(s):[/green]"
            )
            for key in api_keys.keys():
                provider = key.replace("_api_key", "")
                console.print(f"  {provider.title()}")
        else:
            console.print(
                "\n[yellow] Proceeding without API keys. "
                "Some features may be limited.[/yellow]"
            )
            console.print(
                "[dim]You can add API keys later by updating your .env file[/dim]"
            )

        return api_keys

    def _save_api_key_to_env(self, provider: str, api_key: str):
        """Save API key to .env file."""
        import os

        try:
            env_file = ".env"
            env_var_names = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "cohere": "COHERE_API_KEY",
                "google": "GOOGLE_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }

            env_var_name = env_var_names.get(provider, f"{provider.upper()}_API_KEY")

            # Read existing .env file
            env_lines = []
            if os.path.exists(env_file):
                with open(env_file, "r") as f:
                    env_lines = f.readlines()

            # Update or add the API key
            updated = False
            for i, line in enumerate(env_lines):
                if line.startswith(f"{env_var_name}="):
                    env_lines[i] = f"{env_var_name}={api_key}\n"
                    updated = True
                    break

            if not updated:
                env_lines.append(f"{env_var_name}={api_key}\n")

            # Write back to .env file
            with open(env_file, "w") as f:
                f.writelines(env_lines)

            console.print(
                f"[green]Saved {provider.title()} API key to .env file[/green]"
            )

        except Exception as e:
            console.print(f"[yellow]Could not save to .env file: {e}[/yellow]")

    def apply_template_substitutions(
        self, config: Dict[str, Any], substitutions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Apply template substitutions to configuration."""
        config_str = json.dumps(config)

        for key, value in substitutions.items():
            config_str = config_str.replace(f"{{{key}}}", value)

        return json.loads(config_str)

    def run_phase_1_ingestion(
        self, input_path: str, template_key: str, substitutions: Dict[str, str]
    ) -> Tuple[str, Dict[str, Any]]:
        """Run Phase 1: Data Ingestion."""
        console.print("\n[bold green]Phase 1: Data Ingestion[/bold green]")

        start_time = time.time()

        # Get and configure template
        template_config = self.templates["ingestion"][template_key]["config"]
        ingestion_config = self.apply_template_substitutions(
            template_config, substitutions
        )

        # Create output file
        output_file = f"data/phase1_ingested_{int(time.time())}.json"
        Path("data").mkdir(exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[blue]Ingesting data...", total=None)

            try:
                # Initialize and run ingestion
                manager = IngestionManager(ingestion_config)
                asyncio.run(manager.ingest_all_sources())

                # Export results
                manager.export_results(output_file, format="json")

                progress.update(task, completed=100, total=100)

            except Exception as e:
                progress.stop()
                console.print(f"[red]Ingestion failed: {e}[/red]")
                raise typer.Exit(1)

        end_time = time.time()
        self.phase_timings["ingestion"] = end_time - start_time

        # Show results
        with open(output_file, "r") as f:
            ingestion_results = json.load(f)

        successful = sum(1 for r in ingestion_results if r.get("success", False))
        total = len(ingestion_results)

        results_table = Table(title="Ingestion Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("Total Items", str(total))
        results_table.add_row("Successful", str(successful))
        results_table.add_row("Failed", str(total - successful))
        results_table.add_row(
            "Success Rate", f"{(successful/total*100):.1f}%" if total > 0 else "0%"
        )
        results_table.add_row(
            "Processing Time", f"{self.phase_timings['ingestion']:.2f}s"
        )
        results_table.add_row("Output File", output_file)

        console.print(results_table)

        return output_file, ingestion_config

    def run_phase_2_processing(
        self, input_file: str, template_key: str, substitutions: Dict[str, str]
    ) -> Tuple[str, Dict[str, Any]]:
        """Run Phase 2: Processing and Chunking."""
        console.print("\n[bold green]Phase 2: Processing & Chunking[/bold green]")

        start_time = time.time()

        # Get and configure template
        template_config = self.templates["processing"][template_key]["config"]
        processing_config = self.apply_template_substitutions(
            template_config, substitutions
        )

        # Create output file
        output_file = f"data/phase2_processed_{int(time.time())}.json"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[blue]Processing and chunking...", total=None)

            try:
                # Load ingested data
                with open(input_file, "r") as f:
                    ingested_data = json.load(f)

                # Initialize processing manager
                manager = ProcessingManager(processing_config)

                all_chunks = []
                for item in ingested_data:
                    if not item.get("success", False):
                        continue

                    # Create IngestionResult from loaded data
                    from datetime import datetime

                    from neurosync.ingestion.base import (
                        ContentType,
                        IngestionResult,
                        SourceMetadata,
                        SourceType,
                    )

                    # Reconstruct metadata object with proper enum conversion
                    metadata_dict = item["metadata"]

                    # Convert string representations back to enums
                    source_type_str = metadata_dict["source_type"].replace(
                        "SourceType.", ""
                    )
                    content_type_str = metadata_dict["content_type"].replace(
                        "ContentType.", ""
                    )

                    metadata = SourceMetadata(
                        source_id=metadata_dict["source_id"],
                        source_type=SourceType[source_type_str],
                        content_type=ContentType[content_type_str],
                        file_path=metadata_dict.get("file_path"),
                        url=metadata_dict.get("url"),
                        size_bytes=metadata_dict["size_bytes"],
                        created_at=(
                            datetime.fromisoformat(metadata_dict["created_at"])
                            if isinstance(metadata_dict["created_at"], str)
                            else metadata_dict["created_at"]
                        ),
                        modified_at=(
                            datetime.fromisoformat(metadata_dict["modified_at"])
                            if isinstance(metadata_dict["modified_at"], str)
                            else metadata_dict["modified_at"]
                        ),
                        checksum=metadata_dict.get("checksum"),
                        encoding=metadata_dict.get("encoding"),
                        language=metadata_dict.get("language"),
                        title=metadata_dict.get("title"),
                        author=metadata_dict.get("author"),
                        description=metadata_dict.get("description"),
                        tags=metadata_dict.get("tags", []),
                        custom_metadata=metadata_dict.get("custom_metadata", {}),
                    )

                    # Create IngestionResult object
                    ingestion_result = IngestionResult(
                        success=item["success"],
                        source_id=item["source_id"],
                        content=item["content"],
                        metadata=metadata,
                        processing_time_seconds=item["processing_time_seconds"],
                        raw_size_bytes=item["raw_size_bytes"],
                        processed_size_bytes=item["processed_size_bytes"],
                    )

                    # Process the ingestion result
                    chunks = manager.process(ingestion_result)
                    all_chunks.extend(chunks)

                # Save chunks
                chunks_data = [chunk.to_dict() for chunk in all_chunks]
                with open(output_file, "w") as f:
                    json.dump(chunks_data, f, indent=2, default=str)

                progress.update(task, completed=100, total=100)

            except Exception as e:
                progress.stop()
                import traceback

                console.print(f"[red]Processing failed: {e}[/red]")
                console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
                raise typer.Exit(1)

        end_time = time.time()
        self.phase_timings["processing"] = end_time - start_time

        # Show results
        results_table = Table(title="Processing Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("Total Chunks", str(len(all_chunks)))
        avg_size = (
            sum(len(c.content) for c in all_chunks) // len(all_chunks)
            if all_chunks
            else 0
        )
        results_table.add_row(
            "Avg Chunk Size",
            f"{avg_size} chars",
        )
        results_table.add_row(
            "Processing Time", f"{self.phase_timings['processing']:.2f}s"
        )
        results_table.add_row("Output File", output_file)

        console.print(results_table)

        return output_file, processing_config

    def run_phase_3_embedding(
        self,
        chunks_file: str,
        embedding_template_key: str,
        vector_store_template_key: str,
        substitutions: Dict[str, str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run Phase 3: Generate Embeddings and Phase 4: Vector Store."""
        console.print(
            "\n[bold green]Phase 3 & 4: Embeddings & Vector Store[/bold green]"
        )

        start_time = time.time()

        # Get templates and configure
        embedding_template = self.templates["embedding"][embedding_template_key]
        embedding_config = self.apply_template_substitutions(
            embedding_template["config"], substitutions
        )

        # Get user-selected vector store template
        vector_store_template = self.templates["vector_store"][
            vector_store_template_key
        ]
        vector_store_config = self.apply_template_substitutions(
            vector_store_template["config"], substitutions
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Load chunks
            load_task = progress.add_task("[blue]Loading chunks...", total=None)

            try:
                from neurosync.ingestion.base import (
                    ContentType,
                    SourceMetadata,
                    SourceType,
                )
                from neurosync.processing.base import Chunk

                with open(chunks_file, "r") as f:
                    chunk_data = json.load(f)

                chunks = []
                for data in chunk_data:
                    # Reconstruct chunk objects
                    source_metadata_data = data["source_metadata"]
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

                progress.update(load_task, completed=100, total=100)

                # Create and run embedding pipeline
                embed_task = progress.add_task(
                    "[blue]Creating embeddings & vector store...", total=len(chunks)
                )

                pipeline = EmbeddingPipeline(
                    embedding_config=embedding_config,
                    vector_store_config=vector_store_config,
                    enable_hybrid_search=False,
                )

                # Run pipeline with progress updates
                pipeline.run(chunks, batch_size=32, create_backup=False)

                progress.update(embed_task, completed=len(chunks))

            except Exception as e:
                progress.stop()
                console.print(f"[red]Embedding/Vector Store creation failed: {e}[/red]")
                raise typer.Exit(1)

        end_time = time.time()
        self.phase_timings["embedding"] = end_time - start_time

        # Show results
        metrics = pipeline.get_metrics()
        vector_store_info = metrics.get("vector_store", {})
        embedding_info = metrics.get("embedding", {})

        results_table = Table(title=" Embedding & Vector Store Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("Vectors Created", str(vector_store_info.get("count", 0)))
        results_table.add_row(
            "Embedding Dimension", str(vector_store_info.get("dimension", 0))
        )
        results_table.add_row(
            "Vector Store Type", vector_store_info.get("type", "unknown")
        )
        results_table.add_row(
            "Total Processing Time", f"{self.phase_timings['embedding']:.2f}s"
        )

        if embedding_info:
            results_table.add_row(
                "Avg Time per Text",
                f"{embedding_info.get('average_time_per_text', 0):.3f}s",
            )
            results_table.add_row(
                "Embedding Errors", str(embedding_info.get("error_count", 0))
            )

        console.print(results_table)

        return embedding_config, vector_store_config

    def run_phase_5_llm_setup(
        self,
        template_key: str,
        substitutions: Dict[str, str],
        embedding_config: Dict[str, Any],
        vector_store_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Phase 5: LLM Setup and Chat Interface."""
        console.print("\n[bold green]Phase 5: LLM Setup & Chat[/bold green]")

        # Get and configure LLM template
        llm_template = self.templates["llm"][template_key]
        llm_config = self.apply_template_substitutions(
            llm_template["config"], substitutions
        )

        try:
            # Save configurations for serving
            config_dir = Path("config")
            config_dir.mkdir(exist_ok=True)

            with open("config/embedding_config.json", "w") as f:
                json.dump(embedding_config, f, indent=2)

            with open("config/vector_store_config.json", "w") as f:
                json.dump(vector_store_config, f, indent=2)

            with open("config/llm_config.json", "w") as f:
                json.dump(llm_config, f, indent=2)

            # Test LLM connection (but don't fail if it doesn't work)
            with console.status("[bold blue]Testing LLM connection..."):
                try:
                    llm_manager = LLMManager(llm_config)
                    test_response = llm_manager.generate_response(
                        "Say 'Hello' if you can read this message.", max_tokens=50
                    )
                    console.print("[green]LLM connection successful![/green]")
                    console.print(f"[dim]Test response: {test_response[:100]}...[/dim]")
                except Exception as llm_error:
                    console.print(
                        f"[yellow]LLM connection failed: "
                        f"{str(llm_error)[:100]}...[/yellow]"
                    )
                    console.print(
                        "[dim]Chat will work with search results only "
                        "(no AI generation)[/dim]"
                    )

        except Exception as e:
            console.print(f"[red]Configuration save failed: {e}[/red]")
            raise typer.Exit(1)

        return llm_config

    def start_interactive_chat(
        self,
        embedding_config: Dict[str, Any],
        vector_store_config: Dict[str, Any],
        llm_config: Dict[str, Any],
    ):
        """Start interactive chat session."""
        console.print("\n[bold cyan]Starting Interactive Chat Session[/bold cyan]")

        try:
            # Try to use the full retrieval system first, with fallback to simple search
            use_full_retrieval = True
            retriever = None
            llm_manager = None

            try:
                # Initialize components
                from neurosync.processing.embedding.manager import EmbeddingManager
                from neurosync.serving.rag.retriever import Retriever
                from neurosync.storage.vector_store.manager import VectorStoreManager

                embedding_manager = EmbeddingManager(embedding_config)
                vector_store_manager = VectorStoreManager(vector_store_config)
                retriever = Retriever(embedding_manager, vector_store_manager)
                llm_manager = LLMManager(llm_config)

                console.print("[green]Full retrieval system initialized[/green]")

            except Exception as init_error:
                console.print(
                    f"[yellow]Full retrieval system failed "
                    f"({str(init_error)[:100]}...)[/yellow]"
                )
                console.print("[yellow]Falling back to simple text search...[/yellow]")
                use_full_retrieval = False

                # Load processed chunks for simple search
                data_files = list(Path("data").glob("phase2_processed_*.json"))
                if not data_files:
                    console.print(
                        "[red]No processed data found for fallback search.[/red]"
                    )
                    raise typer.Exit(1)

                # Load the most recent processed file
                latest_file = max(data_files, key=os.path.getctime)
                with open(latest_file) as f:
                    chunks_data = json.load(f)

                console.print(
                    f"[green]Loaded {len(chunks_data)} document chunks "
                    f"for simple search[/green]"
                )

                def simple_search(query, chunks, top_k=5):
                    """Simple keyword-based search through chunks."""
                    query_words = query.lower().split()
                    scored_chunks = []

                    for chunk in chunks:
                        content = chunk.get("content", "").lower()
                        score = sum(1 for word in query_words if word in content)
                        if score > 0:
                            scored_chunks.append((score, chunk))

                    # Sort by score and return top results
                    scored_chunks.sort(key=lambda x: x[0], reverse=True)
                    return [(chunk, score) for score, chunk in scored_chunks[:top_k]]

            # Chat loop
            console.print(
                "\n[yellow]Chat is ready! Type 'quit', 'exit', 'bye', "
                "or 'goodbye' to exit.[/yellow]"
            )
            console.print(
                "[dim]Your questions will be answered using the processed data.[/dim]\n"
            )

            while True:
                try:
                    query = Prompt.ask("\n[bold cyan]You")

                    # Enhanced termination conditions
                    termination_words = [
                        "quit",
                        "exit",
                        "q",
                        "bye",
                        "goodbye",
                        "stop",
                        "end",
                    ]
                    if query.lower().strip() in termination_words:
                        break

                    if not query.strip():
                        continue

                    with console.status("[bold blue]Thinking..."):
                        if use_full_retrieval and retriever:
                            # Use full retrieval system
                            try:
                                search_results = retriever.retrieve(query, top_k=5)

                                # Build context from search results
                                context_parts = []
                                for result in search_results:
                                    content = result.metadata.get(
                                        "content", result.metadata.get("text", "")
                                    )
                                    if content:
                                        context_parts.append(content)

                                context = "\n\n".join(context_parts)

                                # Try LLM response if available
                                if llm_manager:
                                    try:
                                        response = llm_manager.generate_response(
                                            f"Context: {context}\n\nQuestion: {query}",
                                            max_tokens=1000,
                                        )
                                    except Exception:
                                        # Fallback to search results
                                        if search_results and context.strip():
                                            response = (
                                                f"**Found relevant information "
                                                f"about '{query}':**\n\n"
                                            )
                                            for i, result in enumerate(
                                                search_results[:3], 1
                                            ):
                                                content = result.metadata.get(
                                                    "content",
                                                    result.metadata.get("text", ""),
                                                )
                                                if content:
                                                    response += (
                                                        f"**Result {i} "
                                                        f"(Relevance: "
                                                        f"{result.score:.2f}):**\n"
                                                    )
                                                    ellipsis = (
                                                        "..."
                                                        if len(content) > 400
                                                        else ""
                                                    )
                                                    response += (
                                                        f"{content[:400]}"
                                                        f"{ellipsis}\n\n"
                                                    )
                                            response += (
                                                "*Note: LLM unavailable - "
                                                "showing search results only*"
                                            )
                                        else:
                                            response = (
                                                f"No relevant information found "
                                                f"for '{query}' and LLM service "
                                                f"unavailable."
                                            )
                                else:
                                    # No LLM, use search results only
                                    if search_results and context.strip():
                                        response = (
                                            f"**Found relevant information "
                                            f"about '{query}':**\n\n"
                                        )
                                        for i, result in enumerate(
                                            search_results[:3], 1
                                        ):
                                            content = result.metadata.get(
                                                "content",
                                                result.metadata.get("text", ""),
                                            )
                                            if content:
                                                response += (
                                                    f"**Result {i} "
                                                    f"(Relevance: "
                                                    f"{result.score:.2f}):**\n"
                                                )
                                                ellipsis = (
                                                    "..." if len(content) > 400 else ""
                                                )
                                                response += (
                                                    f"{content[:400]}{ellipsis}\n\n"
                                                )
                                        response += (
                                            "*Note: Search results only "
                                            "(no LLM configured)*"
                                        )
                                    else:
                                        response = (
                                            f"No relevant information found "
                                            f"for '{query}'."
                                        )

                            except Exception as retrieval_error:
                                console.print(
                                    f"[red]Retrieval error: {retrieval_error}[/red]"
                                )
                                response = "Search system error. Please try again."

                        else:
                            # Use simple search fallback
                            relevant_chunks = simple_search(query, chunks_data, top_k=3)

                            if relevant_chunks:
                                response = (
                                    f"**Found {len(relevant_chunks)} relevant "
                                    f"section(s) about '{query}':**\n\n"
                                )

                                for i, (chunk, score) in enumerate(relevant_chunks, 1):
                                    content = chunk.get("content", "")
                                    source_metadata = chunk.get("source_metadata", {})
                                    file_path = source_metadata.get(
                                        "file_path", "Unknown source"
                                    )

                                    if content:
                                        display_content = (
                                            content[:400] + "..."
                                            if len(content) > 400
                                            else content
                                        )
                                        file_name = (
                                            Path(file_path).name
                                            if file_path != "Unknown source"
                                            else file_path
                                        )
                                        response += (
                                            f"**Section {i}** (from {file_name}):\n"
                                        )
                                        response += f"{display_content}\n\n"

                                response += (
                                    "*This information was found by "
                                    "searching your processed documents*"
                                )

                            else:
                                response = (
                                    f"I couldn't find any relevant information about "
                                    f"'{query}' in your documents. Try rephrasing your "
                                    f"question or using different keywords."
                                )

                    # Display response
                    console.print(f"\n[bold green]Assistant:[/bold green] {response}")

                    # Show sources info if using full retrieval
                    if (
                        use_full_retrieval
                        and "search_results" in locals()
                        and search_results
                    ):
                        console.print(
                            f"\n[dim] Sources: "
                            f"{len(search_results)} documents found[/dim]"
                        )

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

            console.print("\n[yellow] Chat session ended. Goodbye![/yellow]")

        except Exception as e:
            console.print(f"[red]Failed to start chat: {e}[/red]")
            raise typer.Exit(1)

    def show_pipeline_summary(self):
        """Show final pipeline summary."""
        total_time = sum(self.phase_timings.values())

        summary_table = Table(title="Pipeline Execution Summary")
        summary_table.add_column("Phase", style="cyan")
        summary_table.add_column("Duration", style="green")
        summary_table.add_column("Status", style="green")

        for phase, duration in self.phase_timings.items():
            summary_table.add_row(phase.title(), f"{duration:.2f}s", "Complete")

        summary_table.add_row(
            "[bold]Total Pipeline Time[/bold]",
            f"[bold]{total_time:.2f}s[/bold]",
            "[bold green] Success[/bold green]",
        )

        console.print("\n")
        console.print(summary_table)

        # Show saved configurations
        config_panel = Panel(
            "[green]Configuration files saved:[/green]\n\n"
            "• [cyan]config/embedding_config.json[/cyan] - Embedding settings\n"
            "• [cyan]config/vector_store_config.json[/cyan] - Vector store settings\n"
            "• [cyan]config/llm_config.json[/cyan] - LLM configuration\n\n"
            "[yellow]You can reuse these configurations with individual "
            "commands![/yellow]",
            title="Saved Configurations",
            border_style="green",
        )
        console.print(config_panel)

    def _detect_best_llm_template(self) -> str:
        """Detect the best LLM template based on available API keys."""

        # Ensure dotenv is loaded first
        try:
            from dotenv import load_dotenv

            # Look for .env file in current directory and parent directories
            for env_file in [".env", "../.env", "../../.env"]:
                if os.path.exists(env_file):
                    load_dotenv(env_file)
                    break
        except ImportError:
            # python-dotenv not available, continue with system env vars only
            pass

        # Environment variable mappings
        env_var_map = {
            "openai": ["OPENAI_API_KEY", "OPENAI_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_KEY", "CLAUDE_API_KEY"],
            "cohere": ["COHERE_API_KEY", "COHERE_KEY"],
            "google": ["GOOGLE_API_KEY", "GOOGLE_KEY", "GEMINI_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY", "OPENROUTER_KEY"],
        }

        # Placeholder patterns to ignore
        placeholder_patterns = [
            "your-",
            "INSERT_",
            "ADD_",
            "REPLACE_",
            "CHANGE_",
            "PUT_",
            "ENTER_",
            "sk-fake",
            "fake-",
            "test-",
            "demo-",
            "sk-or-v1-123",
            "1234567890abcdef",
        ]

        def is_valid_api_key(value: str) -> bool:
            if not value or not value.strip() or len(value.strip()) <= 5:
                return False
            value_lower = value.lower()
            return not any(pattern in value_lower for pattern in placeholder_patterns)

        # Check for available API keys in priority order
        priority_order = ["openrouter", "openai", "anthropic", "cohere", "google"]

        for provider in priority_order:
            var_names = env_var_map.get(provider, [])
            for var_name in var_names:
                env_value = os.getenv(var_name)
                if env_value and is_valid_api_key(env_value):
                    console.print(
                        f"[green] Found {provider.title()} API key, "
                        f"using {provider} template[/green]"
                    )
                    return provider

        # No valid API keys found, return a template that doesn't need API keys
        console.print(
            "[yellow]  No valid API keys found, will use mock LLM responses[/yellow]"
        )
        return "openai"  # We'll handle the missing key gracefully in the LLM manager

    def run_full_pipeline(self, input_path: str, auto_mode: bool = False):
        """Run the complete end-to-end pipeline."""
        self.pipeline_start_time = time.time()

        try:
            # Phase 0: Detection and Template Selection
            detected_source_type, detected_ingestion_template = self.detect_input_type(
                input_path
            )

            if auto_mode:
                # Use auto-detected templates with smart LLM selection
                ingestion_template = detected_ingestion_template
                processing_template = "advanced"
                embedding_template = "huggingface_fast"
                vector_store_template = "faiss_hnsw"  # Good balance for auto mode
                llm_template = self._detect_best_llm_template()
            else:
                # Interactive template selection
                ingestion_template = self.show_template_selection(
                    "ingestion", detected_ingestion_template
                )
                processing_template = self.show_template_selection("processing")
                embedding_template = self.show_template_selection("embedding")
                vector_store_template = self.show_template_selection("vector_store")
                llm_template = self.show_template_selection("llm")

            # Show selected templates
            selection_table = Table(title=" Selected Templates")
            selection_table.add_column("Phase", style="cyan")
            selection_table.add_column("Template", style="green")

            selection_table.add_row(
                "Ingestion", self.templates["ingestion"][ingestion_template]["name"]
            )
            selection_table.add_row(
                "Processing", self.templates["processing"][processing_template]["name"]
            )
            selection_table.add_row(
                "Embedding", self.templates["embedding"][embedding_template]["name"]
            )
            selection_table.add_row(
                "Vector Store",
                self.templates["vector_store"][vector_store_template]["name"],
            )
            selection_table.add_row("LLM", self.templates["llm"][llm_template]["name"])

            console.print(selection_table)

            # Collect API keys
            template_configs = {
                "embedding": self.templates["embedding"][embedding_template],
                "llm": self.templates["llm"][llm_template],
            }
            api_keys = self.collect_api_keys(template_configs)

            # Prepare substitutions
            # Handle single file vs directory for ingestion
            path_obj = Path(input_path)
            if path_obj.is_file():
                # For single files, use parent directory for ingestion but track
                # the specific file
                ingestion_path = str(path_obj.parent)
                console.print(
                    f"[dim]Single file detected. "
                    f"Using directory: {ingestion_path}[/dim]"
                )
                console.print(f"[dim]Target file: {path_obj.name}[/dim]")
            else:
                ingestion_path = input_path

            substitutions = {"input_path": ingestion_path, **api_keys}

            if not auto_mode:
                proceed = Confirm.ask("\nReady to start the pipeline?")
                if not proceed:
                    console.print("[yellow]Pipeline cancelled.[/yellow]")
                    raise typer.Exit(0)

            # Phase 1: Ingestion
            ingested_file, ingestion_config = self.run_phase_1_ingestion(
                input_path, ingestion_template, substitutions
            )

            # Phase 2: Processing
            processed_file, processing_config = self.run_phase_2_processing(
                ingested_file, processing_template, substitutions
            )

            # Phase 3 & 4: Embedding & Vector Store
            embedding_config, vector_store_config = self.run_phase_3_embedding(
                processed_file, embedding_template, vector_store_template, substitutions
            )

            # Phase 5: LLM Setup
            llm_config = self.run_phase_5_llm_setup(
                llm_template, substitutions, embedding_config, vector_store_config
            )

            # Show summary
            self.show_pipeline_summary()

            # Start chat - available in both auto and interactive modes
            if auto_mode:
                # In auto mode, ask if user wants to start chat
                console.print(
                    "\n[cyan]Auto mode completed! Chat interface is ready.[/cyan]"
                )
                start_chat = Confirm.ask(
                    "Start interactive chat session?", default=True
                )
            else:
                # In interactive mode, ask if user wants to start chat
                start_chat = Confirm.ask(
                    "\nStart interactive chat session?", default=True
                )

            if start_chat:
                self.start_interactive_chat(
                    embedding_config, vector_store_config, llm_config
                )

            console.print("\n[bold green]Pipeline completed successfully![/bold green]")

        except Exception as e:
            console.print(f"\n[red]Pipeline failed: {e}[/red]")
            raise typer.Exit(1)


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

    # Change to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    import os

    os.chdir(output_path)

    # Initialize and run pipeline
    pipeline = FullPipeline()

    # Show welcome message
    if not auto:
        pipeline.show_welcome()

    # Run the full pipeline
    pipeline.run_full_pipeline(input_path, auto_mode=auto)
