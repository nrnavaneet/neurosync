"""
Base classes for embedding model wrappers.
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbeddingModel(ABC):
    """Abstract base class for all embedding models."""

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Returns the dimension of the embeddings."""
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Converts a list of texts into a list of vector embeddings."""
        pass
