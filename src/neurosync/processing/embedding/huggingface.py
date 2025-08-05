"""
Hugging Face sentence-transformers embedding model.
"""
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from neurosync.core.logging.logger import get_logger
from neurosync.processing.embedding.base import BaseEmbeddingModel

logger = get_logger(__name__)


class HuggingFaceEmbedder(BaseEmbeddingModel):
    """Wrapper for Hugging Face sentence-transformer models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading Hugging Face model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully.")

    def get_embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generates embeddings for a list of texts."""
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )

        # Ensure we return a list of individual numpy arrays
        if len(texts) == 1:
            # For single text, ensure we return a 1D array
            if len(embeddings.shape) == 2:
                return [embeddings.flatten()]
            else:
                return [embeddings]
        else:
            # For multiple texts, return list of 1D arrays
            return [embedding for embedding in embeddings]
