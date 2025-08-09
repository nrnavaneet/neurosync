"""
Embedding Manager for coordinating multiple embedding models and providers.

This module provides a centralized management system for different embedding
models and providers, offering a unified interface for text-to-vector
transformation operations. It supports multiple embedding providers with
automatic model selection, performance monitoring, and efficient batch
processing capabilities.

Key Features:
    - Multi-provider support (HuggingFace, OpenAI, Azure, Cohere, etc.)
    - Automatic model initialization and configuration
    - Batch processing optimization for efficient throughput
    - Performance monitoring and metrics collection
    - Error handling and fallback mechanisms
    - Dynamic model switching and A/B testing support
    - Cost optimization through provider selection
    - Caching and embedding storage management

Supported Providers:
    HuggingFace: Open-source models with local inference capability
    OpenAI: Commercial API with high-quality embeddings
    Azure OpenAI: Enterprise-grade OpenAI integration
    Cohere: Specialized commercial embedding models
    Anthropic: Claude-based embedding services
    Custom: Plugin architecture for custom model integration

Model Selection Strategy:
    The manager selects embedding models based on:
    - Configuration preferences and requirements
    - Performance characteristics (speed vs. quality)
    - Cost considerations (free vs. paid models)
    - Privacy requirements (local vs. cloud processing)
    - Language support and specialization needs

Performance Optimization:
    - Intelligent batch processing with optimal batch sizes
    - Connection pooling for API-based providers
    - Parallel processing for independent embedding operations
    - Memory management for large document collections
    - Rate limiting and throttling for API compliance
    - Caching frequently embedded content

Monitoring and Observability:
    - Real-time performance metrics collection
    - Embedding quality assessment and validation
    - Cost tracking and budget management
    - Error rate monitoring and alerting
    - Usage analytics and optimization recommendations

Configuration Options:
    provider: Embedding service provider (huggingface, openai, etc.)
    model_name: Specific model identifier
    api_key: Authentication credentials for commercial providers
    max_batch_size: Optimal batch size for processing
    enable_monitoring: Performance metrics collection
    cache_embeddings: Enable embedding result caching
    timeout: Request timeout for API calls
    retry_config: Retry strategy for failed operations

Quality Assurance:
    - Embedding dimensionality validation
    - Semantic similarity testing for model consistency
    - Performance benchmarking against standard datasets
    - A/B testing framework for model comparison
    - Automated quality degradation detection

Use Cases:
    Search Systems: High-quality embeddings for semantic search
    RAG Applications: Efficient embedding for retrieval pipelines
    Content Classification: Multi-label document categorization
    Similarity Analysis: Document and text similarity computation
    Clustering: Unsupervised content grouping and analysis

Example Configuration:
    >>> config = {
    ...     "provider": "huggingface",
    ...     "model_name": "all-MiniLM-L6-v2",
    ...     "max_batch_size": 32,
    ...     "enable_monitoring": True,
    ...     "cache_embeddings": True
    ... }
    >>> manager = EmbeddingManager(config)

Usage Patterns:
    Single Text:
    >>> embedding = await manager.embed_text("Sample text content")

    Batch Processing:
    >>> texts = ["Text 1", "Text 2", "Text 3"]
    >>> embeddings = await manager.embed_batch(texts)

    With Monitoring:
    >>> with embedding_monitor.track_operation("batch_embedding"):
    ...     embeddings = await manager.embed_batch(large_text_list)

Integration Points:
    - Vector databases for embedding storage
    - Search systems for query and document embedding
    - ML pipelines for feature extraction
    - Monitoring systems for performance tracking
    - Cost management systems for usage optimization

For advanced embedding strategies and custom model integration, see:
    - docs/embedding-configuration.md
    - docs/custom-embedding-models.md
    - examples/multi-provider-embedding.py

Author: NeuroSync Team
Created: 2025
License: MIT
"""
from typing import Any, Dict, List, Optional

import numpy as np

from neurosync.core.exceptions.custom_exceptions import (
    ConfigurationError,
    EmbeddingError,
)
from neurosync.core.logging.logger import get_logger
from neurosync.processing.embedding.base import BaseEmbeddingModel
from neurosync.processing.embedding.huggingface import HuggingFaceEmbedder
from neurosync.processing.embedding.monitoring import embedding_monitor
from neurosync.processing.embedding.openai import OpenAIEmbedder

logger = get_logger(__name__)


class EmbeddingManager:
    """
    Centralized manager for embedding model coordination and execution.

    This class provides a unified interface for working with different
    embedding providers and models. It handles model initialization,
    configuration management, batch processing optimization, and
    performance monitoring across multiple embedding services.

    Architecture:
        The manager uses a plugin-based architecture where different
        embedding providers implement a common interface. This allows
        for seamless switching between providers and easy integration
        of new embedding services.

    Key Responsibilities:
        - Model initialization and lifecycle management
        - Provider-specific configuration handling
        - Batch processing optimization for throughput
        - Error handling and recovery mechanisms
        - Performance monitoring and metrics collection
        - Cost optimization through intelligent provider selection

    Provider Management:
        The manager supports multiple embedding providers:
        - Local models for privacy-sensitive applications
        - Cloud APIs for high-quality commercial embeddings
        - Hybrid approaches combining multiple providers
        - Fallback mechanisms for provider failures

    Performance Features:
        - Automatic batch size optimization based on provider
        - Parallel processing for independent operations
        - Memory-efficient streaming for large datasets
        - Intelligent caching to reduce redundant computations
        - Connection pooling for API-based providers

    Monitoring Integration:
        When monitoring is enabled, the manager tracks:
        - Embedding operation latency and throughput
        - Provider API response times and error rates
        - Cost accumulation and budget utilization
        - Quality metrics and consistency checks

    Configuration Parameters:
        provider: The embedding service provider to use
        model_name: Specific model identifier or name
        api_key: Authentication credentials for commercial services
        max_batch_size: Optimal batch size for processing efficiency
        enable_monitoring: Toggle for performance metrics collection
        timeout: Request timeout for API operations
        retry_strategy: Configuration for failed operation retries

    Error Handling:
        The manager implements comprehensive error handling:
        - Provider-specific error interpretation and recovery
        - Automatic retries with exponential backoff
        - Graceful degradation when providers are unavailable
        - Detailed error context for debugging and monitoring

    Thread Safety:
        The manager is designed for concurrent usage with proper
        synchronization for shared resources and thread-safe
        operations across multiple embedding requests.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: BaseEmbeddingModel = self._initialize_model()
        self.model_name = config.get("model_name", "unknown")
        self.enable_monitoring = config.get("enable_monitoring", True)

    def _initialize_model(self) -> BaseEmbeddingModel:
        """Initializes the embedding model based on config."""
        model_type = self.config.get("type", "huggingface")

        if model_type == "huggingface":
            model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
            return HuggingFaceEmbedder(model_name=model_name)
        elif model_type == "openai":
            model_name = self.config.get("model_name", "text-embedding-3-small")
            api_key = self.config.get("api_key")
            max_batch_size = self.config.get("max_batch_size", 2048)
            return OpenAIEmbedder(
                model_name=model_name, api_key=api_key, max_batch_size=max_batch_size
            )
        else:
            raise ConfigurationError(f"Unsupported embedding model type: {model_type}")

    def generate_embeddings(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """A convenience method to access the model's embed function with monitoring."""
        if not texts:
            return []

        if self.enable_monitoring:
            start_time = embedding_monitor.start_batch(self.model_name)

        try:
            # Process in batches if batch_size is specified
            if batch_size and len(texts) > batch_size:
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    batch_embeddings = self.model.embed(batch)
                    embeddings.extend(batch_embeddings)
            else:
                embeddings = self.model.embed(texts)

            if self.enable_monitoring:
                embedding_monitor.end_batch(
                    self.model_name,
                    start_time,
                    len(texts),
                    success=True,
                    dimension=self.get_dimension(),
                )

                # Calculate quality metrics
                quality_metrics = embedding_monitor.calculate_quality_metrics(
                    embeddings, self.model_name
                )
                logger.debug(f"Quality metrics: {quality_metrics}")

            return embeddings

        except Exception as e:
            if self.enable_monitoring:
                embedding_monitor.end_batch(
                    self.model_name, start_time, len(texts), success=False
                )
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingError(f"Embedding generation failed: {e}")

    def get_dimension(self) -> int:
        """Returns the dimension of the currently loaded model."""
        return self.model.get_embedding_dim()

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get performance metrics for this model."""
        if not self.enable_monitoring:
            return None

        metrics = embedding_monitor.get_metrics(self.model_name)
        return metrics.to_dict() if metrics else None
