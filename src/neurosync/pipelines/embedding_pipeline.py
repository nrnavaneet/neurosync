"""
Embedding Pipeline for NeuroSync
"""

from typing import Any, Dict, List, Optional

from neurosync.core.environment import setup_threading_environment
from neurosync.core.logging.logger import get_logger
from neurosync.processing.base import Chunk
from neurosync.processing.embedding.manager import EmbeddingManager
from neurosync.storage.vector_store.base import Vector
from neurosync.storage.vector_store.hybrid_search import HybridSearchEngine
from neurosync.storage.vector_store.manager import VectorStoreManager

# Set up threading environment to prevent PyTorch/FAISS conflicts
setup_threading_environment()

logger = get_logger(__name__)


class EmbeddingPipeline:
    """Orchestrates the process of embedding chunks and storing them."""

    def __init__(
        self,
        embedding_config: Dict[str, Any],
        vector_store_config: Dict[str, Any],
        enable_hybrid_search: bool = False,
        hybrid_search_config: Optional[Dict[str, Any]] = None,
    ):
        self.embedding_manager = EmbeddingManager(embedding_config)

        # Ensure the vector store is initialized with the correct dimension
        vector_store_config["dimension"] = self.embedding_manager.get_dimension()
        self.vector_store_manager = VectorStoreManager(vector_store_config)

        # Initialize hybrid search if enabled
        self.hybrid_search_engine = None
        if enable_hybrid_search:
            hybrid_config = hybrid_search_config or {}
            self.hybrid_search_engine = HybridSearchEngine(
                vector_store_manager=self.vector_store_manager,
                embedding_manager=self.embedding_manager,
                **hybrid_config,
            )

    def run(
        self, chunks: List[Chunk], batch_size: int = 32, create_backup: bool = False
    ):
        """
        Runs the full pipeline: embeds chunks and upserts them into the vector store.
        """
        logger.info(f"Starting embedding pipeline for {len(chunks)} chunks...")

        # Create backup if requested
        if create_backup:
            try:
                backup_id = self.vector_store_manager.create_backup(
                    f"Pre-pipeline backup - {len(chunks)} chunks"
                )
                logger.info(f"Created backup: {backup_id}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

        total_chunks = len(chunks)
        processed_chunks = 0

        try:
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_chunks + batch_size - 1) // batch_size

                logger.info(
                    f"Processing batch {batch_num}/{total_batches} "
                    f"({len(batch_chunks)} chunks)..."
                )

                # 1. Prepare texts and vectors
                texts_to_embed = [chunk.content for chunk in batch_chunks]
                embeddings = self.embedding_manager.generate_embeddings(
                    texts_to_embed, batch_size=min(batch_size, len(texts_to_embed))
                )

                vectors_to_upsert = []
                for j, chunk in enumerate(batch_chunks):
                    vector = Vector(
                        id=chunk.chunk_id,
                        embedding=embeddings[j],
                        metadata={
                            "source_id": chunk.source_metadata.source_id,
                            "text": chunk.content[:200],  # Store a snippet for context
                            "quality_score": chunk.quality_score,
                            "sequence_num": chunk.sequence_num,
                            "content_type": chunk.source_metadata.content_type.value,
                        },
                    )
                    vectors_to_upsert.append(vector)

                # 2. Upsert vectors into the store
                self.vector_store_manager.upsert(vectors_to_upsert)
                processed_chunks += len(batch_chunks)

                logger.debug(
                    f"Batch {batch_num} completed. "
                    f"Progress: {processed_chunks}/{total_chunks}"
                )

            # 3. Index for hybrid search if enabled
            if self.hybrid_search_engine:
                logger.info("Indexing documents for hybrid search...")
                self.hybrid_search_engine.index_documents(chunks)

            # 4. Optimize the vector store
            logger.info("Optimizing vector store...")
            self.vector_store_manager.optimize()

            logger.info("Embedding pipeline completed successfully.")

            # Log final metrics
            self._log_completion_metrics()

        except Exception as e:
            logger.error(
                f"Pipeline failed after processing {processed_chunks} chunks: {e}"
            )
            raise

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store with the given query.

        Args:
            query: The search query
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search (auto-detect if None)
            filters: Metadata filters
        """
        if use_hybrid is None:
            use_hybrid = self.hybrid_search_engine is not None

        if use_hybrid and self.hybrid_search_engine:
            logger.debug("Using hybrid search")
            hybrid_results = self.hybrid_search_engine.search(
                query, top_k, filters=filters
            )

            # Convert to dict format
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.metadata,
                    "dense_score": result.dense_score,
                    "sparse_score": result.sparse_score,
                    "hybrid_score": result.hybrid_score,
                }
                for result in hybrid_results
            ]
        else:
            logger.debug("Using dense vector search")
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
            dense_results = self.vector_store_manager.search(
                query_embedding, top_k, filters
            )

            # Convert to dict format
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in dense_results
            ]

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for the pipeline."""
        metrics = {}

        # Vector store metrics
        store_info = self.vector_store_manager.get_info()
        metrics["vector_store"] = store_info

        # Embedding metrics
        embedding_metrics = self.embedding_manager.get_metrics()
        if embedding_metrics:
            metrics["embedding"] = embedding_metrics

        # Hybrid search metrics
        if self.hybrid_search_engine:
            metrics["hybrid_search"] = self.hybrid_search_engine.get_stats()

        return metrics

    def _log_completion_metrics(self) -> None:
        """Log completion metrics."""
        metrics = self.get_metrics()

        store_info = metrics.get("vector_store", {})
        logger.info(
            f"Pipeline metrics - "
            f"Vectors: {store_info.get('count', 0)}, "
            f"Dimension: {store_info.get('dimension', 0)}, "
            f"Store type: {store_info.get('type', 'unknown')}"
        )

        if "embedding" in metrics:
            embedding_info = metrics["embedding"]
            logger.info(
                "Embedding metrics - "
                f"Total texts: {embedding_info.get('total_texts', 0)}, "
                f"Avg time per text: "
                f"{embedding_info.get('average_time_per_text', 0):.3f}s, "
                f"Errors: {embedding_info.get('error_count', 0)}"
            )

        if "hybrid_search" in metrics:
            hybrid_info = metrics["hybrid_search"]
            logger.info(
                f"Hybrid search - "
                f"Sparse enabled: {hybrid_info.get('sparse_enabled', False)}, "
                f"Documents: {hybrid_info.get('sparse_documents', 0)}"
            )
