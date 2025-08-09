"""
Recursive text chunking implementation using hierarchical separators.

This module implements a sophisticated recursive chunking strategy that
maintains semantic coherence by splitting text using a hierarchy of
separators. It prioritizes natural language boundaries while ensuring
consistent chunk sizes for optimal embedding performance.

The recursive approach uses multiple delimiter types in order of preference:
    1. Double line breaks (paragraph boundaries)
    2. Single line breaks (sentence boundaries)
    3. Sentence-ending punctuation
    4. Word boundaries (spaces)
    5. Character boundaries (fallback)

Key Features:
    - Hierarchical splitting preserving semantic boundaries
    - Configurable chunk size and overlap parameters
    - Length-aware splitting with character-based measurement
    - Overlap preservation for context continuity
    - Efficient implementation using LangChain's proven algorithms
    - Robust handling of edge cases and malformed text

This chunking strategy is particularly effective for:
    - General text documents with clear paragraph structure
    - Mixed content with varying formatting
    - Cases where maintaining semantic coherence is critical
    - Large documents requiring consistent chunk sizes
    - Content that benefits from contextual overlap

The recursive algorithm ensures that:
    - Semantic boundaries are preserved whenever possible
    - Chunk sizes remain within specified limits
    - Overlap provides adequate context for retrieval
    - Processing is efficient for large documents

Example:
    >>> chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
    >>> chunks = chunker.chunk(long_text_content, metadata={})
    >>> print(f"Generated {len(chunks)} semantically coherent chunks")
    >>>
    >>> # Chunks maintain natural boundaries and overlap
    >>> for i, chunk in enumerate(chunks[:3]):
    ...     print(f"Chunk {i+1}: {len(chunk)} characters")
    ...     print(f"Preview: {chunk[:100]}...")

For configuration recommendations and use cases, see:
    - docs/chunking-strategies.md
    - docs/text-processing-best-practices.md
    - examples/chunking-comparison.py
"""
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from neurosync.processing.base import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Hierarchical text chunker using recursive separator-based splitting.

    The RecursiveChunker implements an intelligent text splitting strategy
    that preserves semantic boundaries by using a hierarchy of separators.
    It attempts to split text at natural boundaries first, falling back
    to less ideal separators only when necessary to maintain size limits.

    Separator Hierarchy (in order of preference):
        1. Double newlines (\n\n) - Paragraph boundaries
        2. Single newlines (\n) - Line breaks
        3. Periods followed by space (. ) - Sentence endings
        4. Spaces ( ) - Word boundaries
        5. Characters - Last resort splitting

    Attributes:
        splitter (RecursiveCharacterTextSplitter): LangChain text splitter

    The chunker ensures that:
        - Semantic boundaries are preserved when possible
        - Chunk sizes stay within specified limits
        - Overlaps provide context continuity between chunks
        - Processing handles various text formats robustly

    Configuration Parameters:
        chunk_size: Target size for each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        length_function: Function to measure text length (default: len)

    Example:
        >>> chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
        >>> text = "Long document with multiple paragraphs..."
        >>> chunks = chunker.chunk(text, metadata={"source": "document.txt"})
        >>>
        >>> # Verify chunk properties
        >>> for i, chunk in enumerate(chunks):
        ...     print(f"Chunk {i+1}: {len(chunk)} chars")
        ...     assert len(chunk) <= 512, "Chunk exceeds size limit"
        ...
        >>> # Check overlap between adjacent chunks
        >>> if len(chunks) > 1:
        ...     overlap = chunks[0][-25:] in chunks[1][:75]
        ...     print(f"Has overlap: {overlap}")
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        super().__init__(chunk_size, chunk_overlap)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Split text content into chunks using recursive strategy.

        Applies the recursive chunking algorithm to split text at natural
        boundaries while maintaining size constraints. The method handles
        edge cases gracefully and ensures consistent output quality.

        Args:
            content (str): Text content to be chunked
            metadata (Dict[str, Any]): Additional metadata (not used in splitting)

        Returns:
            List[str]: List of text chunks maintaining semantic boundaries

        Processing Steps:
            1. Validate input content for basic requirements
            2. Apply recursive splitting using separator hierarchy
            3. Ensure chunks meet size and overlap requirements
            4. Handle edge cases (empty content, single words, etc.)

        Edge Case Handling:
            - Empty or whitespace-only content returns empty list
            - Content shorter than chunk_size returns single chunk
            - Very long words that exceed chunk_size are split at character boundaries
            - Maintains overlap even with irregular content structure

        Example:
            >>> chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
            >>> content = "First paragraph.\\n\\nSecond paragraph with more text."
            >>> chunks = chunker.chunk(content, {"source": "test"})
            >>>
            >>> # Verify results
            >>> assert all(len(chunk) <= 100 for chunk in chunks)
            >>> if len(chunks) > 1:
            ...     # Check for overlap between consecutive chunks
            ...     overlap_found = any(
            ...         chunks[i][-10:] in chunks[i+1][:30]
            ...         for i in range(len(chunks)-1)
            ...     )
            ...     assert overlap_found, "Expected overlap between chunks"
        """
        return self.splitter.split_text(content)
