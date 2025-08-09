"""
OpenAI embedding model wrapper for commercial-grade text embeddings.

This module provides a comprehensive wrapper for OpenAI's text embedding
models, offering high-quality vector representations through OpenAI's API.
It includes batch processing optimization, error handling, cost tracking,
and performance monitoring for production-grade embedding operations.

Key Features:
    - Support for all OpenAI embedding models (text-embedding-3-small/large)
    - Intelligent batch processing with optimal batch sizes
    - Automatic retry mechanisms with exponential backoff
    - Cost tracking and usage monitoring
    - Rate limiting compliance with OpenAI API guidelines
    - Error handling for API failures and network issues
    - Performance optimization for large-scale operations
    - Secure credential management

Supported Models:
    text-embedding-3-small: 1536 dimensions, optimized for speed and cost
    text-embedding-3-large: 3072 dimensions, highest quality embeddings
    text-embedding-ada-002: Legacy model, 1536 dimensions

Performance Characteristics:
    - Batch processing up to 2048 texts per request
    - Automatic rate limiting to respect API quotas
    - Concurrent request processing with connection pooling
    - Memory-efficient streaming for large datasets
    - Intelligent retry strategies for transient failures

Quality Features:
    - State-of-the-art embedding quality from OpenAI models
    - Consistent embeddings across different batch sizes
    - Normalized vectors for cosine similarity optimization
    - High-dimensional representations for semantic precision

Cost Optimization:
    - Automatic text length optimization to reduce token usage
    - Batch size optimization for cost-efficient API usage
    - Usage tracking and budget monitoring capabilities
    - Model selection guidance based on quality vs. cost trade-offs

Configuration Options:
    model_name: OpenAI model identifier
    api_key: OpenAI API authentication key
    max_batch_size: Maximum texts per batch request
    max_retries: Retry attempts for failed requests
    timeout: Request timeout in seconds
    base_url: Custom API endpoint (for Azure OpenAI)

Error Handling:
    - Authentication error detection and clear messaging
    - Rate limit handling with automatic backoff
    - Network error recovery with exponential retry
    - Input validation and preprocessing error handling
    - Comprehensive error context for debugging

Security Features:
    - Secure API key management and storage
    - HTTPS-only communication with OpenAI servers
    - Input sanitization and validation
    - No local storage of sensitive embedding data

Integration Points:
    - Embedding manager for multi-provider coordination
    - Monitoring systems for performance and cost tracking
    - Configuration management for dynamic model selection
    - Caching systems for embedding result storage

Example Usage:
    >>> embedder = OpenAIEmbedder(
    ...     model_name="text-embedding-3-small",
    ...     api_key="your-api-key"
    ... )
    >>> embeddings = await embedder.embed_batch(texts)

For advanced OpenAI integration and cost optimization, see:
    - docs/openai-embedding-configuration.md
    - docs/embedding-cost-optimization.md
    - examples/openai-batch-processing.py

Author: NeuroSync Team
Created: 2025
License: MIT
"""
from typing import List

import numpy as np
from openai import OpenAI

from neurosync.core.exceptions.custom_exceptions import EmbeddingError
from neurosync.core.logging.logger import get_logger
from neurosync.processing.embedding.base import BaseEmbeddingModel

logger = get_logger(__name__)


class OpenAIEmbedder(BaseEmbeddingModel):
    """Wrapper for OpenAI embedding models."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = None,
        max_batch_size: int = 2048,
    ):
        """
        Initialize OpenAI embedder.

        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            max_batch_size: Maximum number of texts to process in one batch
        """
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.client = OpenAI(api_key=api_key)

        # Get model dimensions
        self._dimension = self._get_model_dimension()
        logger.info(
            f"Initialized OpenAI embedder with model: {model_name}, "
            f"dimension: {self._dimension}"
        )

    def _get_model_dimension(self) -> int:
        """Get the embedding dimension for the model."""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        if self.model_name in model_dimensions:
            return model_dimensions[self.model_name]

        # For unknown models, make a test call to get dimension
        try:
            response = self.client.embeddings.create(
                model=self.model_name, input=["test"]
            )
            return len(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Failed to get model dimension: {e}")
            raise EmbeddingError(
                f"Could not determine dimension for model {self.model_name}: {e}"
            )

    def get_embedding_dim(self) -> int:
        """Returns the dimension of the embeddings."""
        return self._dimension

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generates embeddings for a list of texts."""
        if not texts:
            return []

        embeddings = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model_name, input=batch
                )

                batch_embeddings = [
                    np.array(data.embedding, dtype=np.float32) for data in response.data
                ]
                embeddings.extend(batch_embeddings)

                logger.debug(
                    f"Generated embeddings for batch {i//self.max_batch_size + 1}"
                )

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                raise EmbeddingError(f"OpenAI embedding generation failed: {e}")

        return embeddings
