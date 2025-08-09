"""
Base classes and data structures for the text processing phase.

This module defines the core abstractions and data structures used throughout
the processing pipeline. It provides base classes for preprocessing and chunking
operations, as well as the fundamental Chunk data structure that represents
processed text segments ready for embedding and indexing.

Classes:
    Chunk: Data structure representing a processed text segment
    BasePreprocessor: Abstract base class for text preprocessing operations
    BaseChunker: Abstract base class for text chunking strategies

The processing phase transforms raw ingested content into structured chunks
that are optimized for vector embedding and retrieval. This includes:
    - Text cleaning and normalization
    - Language detection and filtering
    - Content chunking with overlap strategies
    - Metadata enrichment and quality scoring
    - Hierarchical relationship mapping

Example:
    >>> from neurosync.processing.base import Chunk
    >>> from neurosync.ingestion.base import SourceMetadata
    >>>
    >>> metadata = SourceMetadata(source_id="doc1", source_type="file")
    >>> chunk = Chunk(content="Sample text content", source_metadata=metadata)
    >>> print(f"Chunk ID: {chunk.chunk_id}")
    >>> print(f"Content: {chunk.content}")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from neurosync.ingestion.base import SourceMetadata


@dataclass
class Chunk:
    """
    A processed text segment ready for embedding and vector storage.

    The Chunk class represents a single unit of processed text content
    along with its metadata, relationships, and quality metrics. Each chunk
    is designed to be semantically coherent and appropriately sized for
    embedding generation and retrieval operations.

    Attributes:
        content (str): The processed text content of the chunk
        source_metadata (SourceMetadata): Metadata from the original source
        chunk_id (str): Unique identifier for the chunk (auto-generated UUID)
        sequence_num (int): Sequential position within the source document
        parent_chunk_id (Optional[str]): ID of parent chunk in hierarchy
        child_chunk_ids (List[str]): IDs of child chunks in hierarchy
        quality_score (float): Quality assessment score (0.0 to 1.0)
        processing_metadata (Dict[str, Any]): Additional processing metadata

    The chunk hierarchy supports advanced retrieval patterns:
        - Parent chunks provide broader context
        - Child chunks provide granular details
        - Sibling chunks are related content segments

    Quality scoring helps filter low-value content:
        - 1.0: High quality, rich semantic content
        - 0.5: Average quality, some useful information
        - 0.0: Poor quality, minimal useful content

    Example:
        >>> from neurosync.ingestion.base import SourceMetadata
        >>> metadata = SourceMetadata(source_id="doc1", source_type="file")
        >>> chunk = Chunk(
        ...     content="This is a sample text chunk.",
        ...     source_metadata=metadata,
        ...     sequence_num=1,
        ...     quality_score=0.8
        ... )
        >>> print(chunk.chunk_id)  # Auto-generated UUID
        >>> chunk_dict = chunk.to_dict()  # For serialization
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
        """
        Convert the Chunk object to a dictionary for serialization.

        Transforms the chunk into a JSON-serializable dictionary format
        suitable for storage, transmission, or further processing. All
        nested objects are also converted to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of the chunk with:
                - chunk_id: Unique chunk identifier
                - content: Text content
                - sequence_num: Position in source document
                - parent_chunk_id: Parent chunk reference (if any)
                - child_chunk_ids: List of child chunk references
                - quality_score: Quality assessment value
                - source_metadata: Original source metadata as dict
                - processing_metadata: Additional processing data

        Example:
            >>> chunk = Chunk(content="Sample", source_metadata=metadata)
            >>> chunk_dict = chunk.to_dict()
            >>> print(chunk_dict["chunk_id"])  # UUID string
            >>> print(chunk_dict["content"])   # "Sample"
        """
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
    """
    Abstract base class for all content preprocessors.

    Preprocessors are responsible for cleaning and normalizing raw text
    content before chunking and embedding. They handle tasks such as:
        - Removing unwanted characters and formatting
        - Normalizing whitespace and encoding
        - Language detection and filtering
        - Content validation and quality assessment

    Subclasses must implement the process() method to define specific
    preprocessing logic. The __call__ method provides a convenient
    interface for using preprocessors as callable objects.

    Example:
        >>> from neurosync.processing.preprocessing.cleaners import TextCleaner
        >>> preprocessor = TextCleaner()
        >>> clean_text = preprocessor.process("Raw text content...")
        >>> # Or use as callable
        >>> clean_text = preprocessor("Raw text content...")
    """

    @abstractmethod
    def process(self, content: str) -> str:
        """
        Process raw content and return the cleaned version.

        This method must be implemented by all preprocessor subclasses
        to define their specific cleaning and normalization logic.

        Args:
            content (str): Raw text content to be processed

        Returns:
            str: Cleaned and normalized text content

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    def __call__(self, content: str) -> str:
        """
        Convenience method to use preprocessor as a callable.

        Allows preprocessors to be used as function-like objects,
        providing a more intuitive interface for text processing.

        Args:
            content (str): Raw text content to be processed

        Returns:
            str: Processed text content
        """
        return self.process(content)


class BaseChunker(ABC):
    """
    Abstract base class for all text chunking strategies.

    Chunkers are responsible for splitting processed text into smaller,
    semantically coherent segments that are optimal for embedding and
    retrieval. Different strategies include:
        - Fixed-size chunking with overlap
        - Semantic boundary detection
        - Document structure awareness
        - Hierarchical chunk relationships

    All chunkers accept chunk size and overlap parameters to control
    the granularity and continuity of the resulting chunks.

    Attributes:
        chunk_size (int): Target size for each chunk in tokens/characters
        chunk_overlap (int): Overlap size between adjacent chunks

    Example:
        >>> from neurosync.processing.chunking.recursive_chunker import RecursiveChunker
        >>> chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
        >>> chunks = chunker.chunk(content="Long text content...", metadata={})
        >>> print(f"Generated {len(chunks)} chunks")
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initialize chunker with size and overlap parameters.

        Args:
            chunk_size (int): Target size for each chunk in tokens/characters
            chunk_overlap (int): Number of tokens/characters to overlap between chunks

        Note:
            The exact interpretation of chunk_size and chunk_overlap may vary
            between different chunking strategies (tokens vs characters vs words).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Split content into a list of text chunks.

        This method must be implemented by all chunker subclasses to define
        their specific chunking strategy and logic.

        Args:
            content (str): Processed text content to be chunked
            metadata (Dict[str, Any]): Additional metadata that may influence chunking

        Returns:
            List[str]: List of text chunks, ordered by appearance in source

        Raises:
            NotImplementedError: If not implemented by subclass

        Note:
            The returned chunks should maintain the logical order from the
            source content and include appropriate overlap for context continuity.
        """
        pass
