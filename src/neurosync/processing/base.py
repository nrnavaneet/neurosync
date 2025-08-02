"""
Base classes and data structures for the processing phase.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from neurosync.ingestion.base import SourceMetadata


@dataclass
class Chunk:
    """
    A dataclass representing a single chunk of processed text,
    ready for embedding and indexing.
    """

    content: str
    source_metadata: SourceMetadata

    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    sequence_num: int = 0

    # For hierarchical and relationship mapping
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)

    # For quality and filtering
    quality_score: float = 0.0

    # Additional metadata from processing
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Chunk object to a dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "sequence_num": self.sequence_num,
            "parent_chunk_id": self.parent_chunk_id,
            "child_chunk_ids": self.child_chunk_ids,
            "quality_score": self.quality_score,
            "source_metadata": self.source_metadata.to_dict(),
            "processing_metadata": self.processing_metadata,
        }


class BasePreprocessor(ABC):
    """Abstract base class for all content preprocessors."""

    @abstractmethod
    def process(self, content: str) -> str:
        """Process the content and return the cleaned version."""
        pass

    def __call__(self, content: str) -> str:
        return self.process(content)


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Split the content into a list of text chunks."""
        pass
