"""
The retrieval engine for finding and re-ranking relevant context.
"""
from typing import Any, Dict, List

import numpy as np

from neurosync.core.logging.logger import get_logger
from neurosync.processing.embedding.manager import EmbeddingManager
from neurosync.storage.vector_store.base import SearchResult
from neurosync.storage.vector_store.manager import VectorStoreManager

logger = get_logger(__name__)


class Retriever:
    """Handles query embedding and context retrieval with advanced ranking."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store_manager: VectorStoreManager,
        rerank_enabled: bool = True,
    ):
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager
        self.rerank_enabled = rerank_enabled

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        diversity_threshold: float = 0.7,
        min_score: float = 0.3,
    ) -> List[SearchResult]:
        """
        Embeds a query, searches the vector store, and returns relevant results.

        Args:
            query: The search query
            top_k: Number of results to return
            diversity_threshold: Minimum diversity score between results
            min_score: Minimum relevance score threshold
        """
        logger.info(f"Retrieving context for query: '{query}'")

        # 1. Embed the query
        query_embeddings = self.embedding_manager.generate_embeddings([query])
        if not query_embeddings:
            logger.warning("Failed to generate query embedding")
            return []

        query_embedding = query_embeddings[0]

        # 2. Search the vector store with larger initial set for reranking
        initial_k = top_k * 2 if self.rerank_enabled else top_k
        search_results = self.vector_store_manager.search(query_embedding, initial_k)

        if not search_results:
            logger.info("No results found in vector store")
            return []

        # 3. Filter by minimum score
        filtered_results = [
            result for result in search_results if result.score >= min_score
        ]

        if not filtered_results:
            logger.info(f"No results above minimum score threshold: {min_score}")
            return []

        # 4. Apply reranking and diversity filtering
        if self.rerank_enabled:
            final_results = self._rerank_and_diversify(
                filtered_results, query_embedding, top_k, diversity_threshold
            )
        else:
            final_results = filtered_results[:top_k]

        logger.info(f"Retrieved {len(final_results)} relevant chunks.")
        return final_results

    def _rerank_and_diversify(
        self,
        results: List[SearchResult],
        query_embedding: np.ndarray,
        top_k: int,
        diversity_threshold: float,
    ) -> List[SearchResult]:
        """Apply diversity-based reranking to results."""
        if len(results) <= top_k:
            return results

        selected_results = []
        remaining_results = results.copy()

        # Always select the top result
        if remaining_results:
            selected_results.append(remaining_results.pop(0))

        # Select diverse results
        while len(selected_results) < top_k and remaining_results:
            best_candidate = None
            best_score = -1.0

            for candidate in remaining_results:
                # Calculate diversity score against already selected results
                diversity_score = self._calculate_diversity_score(
                    candidate, selected_results
                )

                # Combined score: relevance + diversity
                combined_score = candidate.score * 0.7 + diversity_score * 0.3

                if (
                    combined_score > best_score
                    and diversity_score >= diversity_threshold
                ):
                    best_score = combined_score
                    best_candidate = candidate

            if best_candidate:
                selected_results.append(best_candidate)
                remaining_results.remove(best_candidate)
            else:
                # If no diverse candidate found, add the next best by relevance
                if remaining_results:
                    selected_results.append(remaining_results.pop(0))

        return selected_results

    def _calculate_diversity_score(
        self, candidate: SearchResult, selected_results: List[SearchResult]
    ) -> float:
        """Calculate diversity score for a candidate against selected results."""
        if not selected_results:
            return 1.0

        similarities = []
        for selected in selected_results:
            # Simple text overlap similarity (can be enhanced with embeddings)
            candidate_text = getattr(candidate.metadata, "text", "").lower()
            selected_text = getattr(selected.metadata, "text", "").lower()

            # Jaccard similarity
            candidate_words = set(candidate_text.split())
            selected_words = set(selected_text.split())

            if not candidate_words and not selected_words:
                similarity = 1.0
            elif not candidate_words or not selected_words:
                similarity = 0.0
            else:
                intersection = len(candidate_words.intersection(selected_words))
                union = len(candidate_words.union(selected_words))
                similarity = intersection / union if union > 0 else 0.0

            similarities.append(similarity)

        # Return inverse of max similarity (higher diversity = lower similarity)
        max_similarity = max(similarities) if similarities else 0.0
        return 1.0 - max_similarity

    def get_context_window_stats(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Get statistics about the context window."""
        if not results:
            return {"total_chars": 0, "total_tokens": 0, "chunk_count": 0}

        total_chars = sum(
            len(getattr(result.metadata, "text", "")) for result in results
        )

        # Rough token estimation (1 token ≈ 4 characters)
        estimated_tokens = total_chars // 4

        return {
            "total_chars": total_chars,
            "estimated_tokens": estimated_tokens,
            "chunk_count": len(results),
            "average_chunk_size": total_chars // len(results) if results else 0,
        }
