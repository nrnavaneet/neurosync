"""
Manager to handle different embedding models.
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
    """Selects and uses the configured embedding model."""

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
