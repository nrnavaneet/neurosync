"""
Metrics and monitoring for embedding operations.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from neurosync.core.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding operations."""

    model_name: str
    total_texts: int = 0
    total_vectors: int = 0
    total_time_seconds: float = 0.0
    average_time_per_text: float = 0.0
    batch_times: List[float] = field(default_factory=list)
    error_count: int = 0
    dimension: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "model_name": self.model_name,
            "total_texts": self.total_texts,
            "total_vectors": self.total_vectors,
            "total_time_seconds": self.total_time_seconds,
            "average_time_per_text": self.average_time_per_text,
            "error_count": self.error_count,
            "dimension": self.dimension,
            "created_at": self.created_at.isoformat(),
            "avg_batch_time": np.mean(self.batch_times) if self.batch_times else 0.0,
            "min_batch_time": min(self.batch_times) if self.batch_times else 0.0,
            "max_batch_time": max(self.batch_times) if self.batch_times else 0.0,
        }


class EmbeddingMonitor:
    """Monitor embedding performance and quality."""

    def __init__(self):
        self.metrics: Dict[str, EmbeddingMetrics] = {}

    def start_batch(self, model_name: str) -> float:
        """Start timing a batch operation."""
        if model_name not in self.metrics:
            self.metrics[model_name] = EmbeddingMetrics(model_name=model_name)
        return time.time()

    def end_batch(
        self,
        model_name: str,
        start_time: float,
        batch_size: int,
        success: bool = True,
        dimension: Optional[int] = None,
    ) -> None:
        """End timing a batch operation and update metrics."""
        elapsed_time = time.time() - start_time

        if model_name not in self.metrics:
            self.metrics[model_name] = EmbeddingMetrics(model_name=model_name)

        metrics = self.metrics[model_name]
        metrics.batch_times.append(elapsed_time)
        metrics.total_time_seconds += elapsed_time

        if success:
            metrics.total_texts += batch_size
            metrics.total_vectors += batch_size
        else:
            metrics.error_count += 1

        if dimension:
            metrics.dimension = dimension

        # Update average time per text
        if metrics.total_texts > 0:
            metrics.average_time_per_text = (
                metrics.total_time_seconds / metrics.total_texts
            )

        logger.debug(
            f"Batch completed for {model_name}: {batch_size} texts in "
            f"{elapsed_time:.2f}s"
        )

    def get_metrics(self, model_name: str) -> Optional[EmbeddingMetrics]:
        """Get metrics for a specific model."""
        return self.metrics.get(model_name)

    def get_all_metrics(self) -> Dict[str, EmbeddingMetrics]:
        """Get all metrics."""
        return self.metrics

    def reset_metrics(self, model_name: Optional[str] = None) -> None:
        """Reset metrics for a specific model or all models."""
        if model_name:
            if model_name in self.metrics:
                del self.metrics[model_name]
        else:
            self.metrics.clear()

    def clear_all_metrics(self) -> None:
        """Clear all metrics - alias for reset_metrics()."""
        self.reset_metrics()

    def log_summary(self, model_name: str) -> None:
        """Log a summary of metrics for a model."""
        metrics = self.metrics.get(model_name)
        if not metrics:
            logger.warning(f"No metrics found for model: {model_name}")
            return

        logger.info(
            f"Embedding metrics for {model_name}: "
            f"Texts: {metrics.total_texts}, "
            f"Time: {metrics.total_time_seconds:.2f}s, "
            f"Avg per text: {metrics.average_time_per_text:.3f}s, "
            f"Errors: {metrics.error_count}"
        )

    def calculate_quality_metrics(
        self, embeddings: List[np.ndarray], model_name: str
    ) -> Dict[str, float]:
        """Calculate quality metrics for embeddings."""
        if len(embeddings) == 0:
            return {}

        embeddings_array = np.array(embeddings)

        # Calculate basic statistics
        norms = np.linalg.norm(embeddings_array, axis=1)

        quality_metrics = {
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
            "dimension": embeddings_array.shape[1],
            "count": len(embeddings),
        }

        # Calculate diversity (average pairwise cosine distance)
        if len(embeddings) > 1:
            # Normalize embeddings
            normalized = embeddings_array / norms[:, np.newaxis]

            # Calculate pairwise similarities
            similarities = np.dot(normalized, normalized.T)

            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(similarities, k=1)
            valid_similarities = upper_triangle[upper_triangle != 0]

            if len(valid_similarities) > 0:
                quality_metrics["mean_similarity"] = float(np.mean(valid_similarities))
                quality_metrics["diversity_score"] = float(
                    1.0 - np.mean(valid_similarities)
                )

        logger.debug(f"Quality metrics for {model_name}: {quality_metrics}")
        return quality_metrics


# Global monitor instance
embedding_monitor = EmbeddingMonitor()
