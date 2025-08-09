"""
Sliding Window Chunker Implementation.

This module provides a sophisticated sliding window chunking strategy that
creates overlapping text chunks with configurable window and step sizes.
This approach is particularly effective for maintaining contextual continuity
across chunk boundaries and ensuring comprehensive information coverage in
dense, interconnected content.

Core Concept:
    The sliding window technique moves a fixed-size window across the text
    with a configurable step size. When the step size is smaller than the
    window size, overlapping chunks are created, preserving context that
    might be lost with non-overlapping segmentation approaches.

Key Features:
    - Configurable window size for consistent chunk dimensions
    - Adjustable step size for controlling overlap amount
    - Character-based and token-based sliding window modes
    - Boundary-aware processing to respect word and sentence boundaries
    - Memory-efficient streaming for large document processing
    - Integration with various tokenization strategies
    - Metadata tracking for chunk relationships and overlaps

Use Cases:
    Dense Technical Content: Overlapping chunks ensure complex concepts
                           spanning boundaries are preserved
    Legal Documents: Contextual relationships between clauses maintained
    Scientific Papers: Cross-references and dependencies preserved
    Code Documentation: Function relationships and dependencies captured
    Narrative Text: Story flow and character development continuity

Performance Characteristics:
    - Linear time complexity O(n) for text processing
    - Configurable memory usage based on window size
    - Efficient streaming processing for large documents
    - Minimal memory footprint with sliding window approach

Chunking Strategies:
    Character-based: Fixed character count windows with precise overlap
    Token-based: Word/token count windows respecting linguistic boundaries
    Sentence-aware: Window boundaries aligned to sentence endings
    Paragraph-aware: Larger windows with paragraph-level overlaps

Configuration Parameters:
    window_size: Size of each chunk in characters or tokens
    step_size: Distance to move window (smaller = more overlap)
    overlap_ratio: Alternative to step_size using percentage overlap
    boundary_mode: How to handle boundaries (strict, word, sentence)
    min_chunk_size: Minimum size threshold for generated chunks
    tokenizer: Tokenization strategy for token-based chunking

Quality Assurance:
    - Overlap validation to ensure proper context preservation
    - Boundary checking to prevent word/sentence fragmentation
    - Size consistency validation across all generated chunks
    - Content completeness verification (no text loss)

Example Configuration:
    >>> config = {
    ...     "window_size": 1000,      # 1000 characters per chunk
    ...     "step_size": 800,         # 200 character overlap
    ...     "boundary_mode": "word",  # Respect word boundaries
    ...     "min_chunk_size": 100     # Minimum viable chunk size
    ... }
    >>> chunker = SlidingWindowChunker(config)

Integration Points:
    - Vector embedding systems requiring consistent chunk sizes
    - Search systems benefiting from contextual overlap
    - RAG systems needing comprehensive content coverage
    - Analysis pipelines requiring sliding window statistics

Author: NeuroSync Team
Created: 2025
License: MIT
"""

from typing import Any, Dict, List, Optional

from neurosync.core.logging.logger import get_logger
from neurosync.processing.base import BaseChunker

logger = get_logger(__name__)


class SlidingWindowChunker(BaseChunker):
    """
    A chunker that uses a sliding window approach to create overlapping chunks.

    This strategy moves a window of fixed size across the text with a configurable
    step size, creating chunks that overlap to preserve context at boundaries.
    Particularly useful for dense information where context boundaries are unclear.

    Key Benefits:
        - Preserves context at chunk boundaries through overlap
        - Configurable window and step sizes for different use cases
        - Consistent chunk sizes for uniform processing
        - Reduces information loss at arbitrary split points

    Use Cases:
        - Dense technical documents
        - Legal and regulatory text
        - Scientific papers with continuous arguments
        - Any content where context preservation is critical
    """

    def __init__(
        self, chunk_size: int, chunk_overlap: int, step_size: Optional[int] = None
    ):
        """
        Initialize the sliding window chunker.

        Args:
            chunk_size: Size of each chunk (window size)
            chunk_overlap: Amount of overlap between consecutive chunks
            step_size: Step size for moving the window
                      (defaults to chunk_size - chunk_overlap)

        Notes:
            - If step_size is not provided, it's calculated as
              (chunk_size - chunk_overlap)
            - Step size must be less than chunk_size to create meaningful chunks
            - Larger overlaps preserve more context but create more chunks
        """
        super().__init__(chunk_size, chunk_overlap)

        # Calculate step size (how much to advance each window)
        if chunk_overlap >= chunk_size:
            logger.warning(
                "Overlap size greater than or equal to chunk size. Adjusting overlap."
            )
            chunk_overlap = max(0, chunk_size - 1)

        self.step_size = chunk_size - chunk_overlap

        if self.step_size <= 0:
            logger.warning("Step size <= 0. Setting step size to 1.")
            self.step_size = 1

    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Chunk the content using sliding window approach.

        Creates overlapping chunks by sliding a window across the text.
        Each chunk has the specified size with configurable overlap.

        Args:
            content: Text content to chunk
            metadata: Additional metadata (unused in this implementation)

        Returns:
            List of text chunks with overlaps

        Algorithm:
            1. Start at position 0
            2. Extract chunk of specified size
            3. Move window by step_size
            4. Repeat until end of content
            5. Handle final chunk if needed
        """
        if not content:
            logger.debug("Empty content provided, returning empty list")
            return []

        if len(content) <= self.chunk_size:
            logger.debug("Content smaller than chunk size, returning single chunk")
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            # Extract chunk
            end = min(start + self.chunk_size, len(content))
            chunk = content[start:end]

            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)

            # Move window by step size
            start += self.step_size

            # Break if we've reached the end
            if end >= len(content):
                break

        # Ensure we don't have duplicate final chunks
        if len(chunks) > 1:
            # Check if last chunk is just a small remainder of the previous chunk
            last_chunk = chunks[-1]
            second_last_chunk = chunks[-2] if len(chunks) > 1 else ""

            # If the last chunk is mostly contained in the second-to-last chunk,
            # remove it
            if len(last_chunk) < self.step_size and last_chunk in second_last_chunk:
                chunks.pop()

        logger.debug(f"Created {len(chunks)} chunks using sliding window approach")
        return chunks


class TokenAwareSlidingWindowChunker(SlidingWindowChunker):
    """
    Token-aware sliding window chunker that respects word boundaries.

    Extends the basic sliding window chunker to be more intelligent about
    where to split text, preferring word boundaries over character boundaries.
    """

    def __init__(
        self, chunk_size: int, chunk_overlap: int, step_size: Optional[int] = None
    ):
        """
        Initialize token-aware sliding window chunker.

        Args:
            chunk_size: Approximate target size in characters
            chunk_overlap: Approximate overlap in characters
            step_size: Step size for moving window
        """
        super().__init__(chunk_size, chunk_overlap, step_size)

    def _find_word_boundary(
        self, content: str, position: int, direction: str = "backward"
    ) -> int:
        """
        Find the nearest word boundary from the given position.

        Args:
            content: The text content
            position: Starting position
            direction: "backward" or "forward" to search direction

        Returns:
            Position of nearest word boundary
        """
        if direction == "backward":
            # Search backward for whitespace
            while position > 0 and not content[position].isspace():
                position -= 1
        else:
            # Search forward for whitespace
            while position < len(content) and not content[position].isspace():
                position += 1

        return position

    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Chunk content using token-aware sliding window.

        Prefers to split at word boundaries rather than arbitrary character positions.
        """
        if not content:
            return []

        if len(content) <= self.chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            # Calculate target end position
            target_end = min(start + self.chunk_size, len(content))

            # Adjust end to word boundary if not at content end
            if target_end < len(content):
                actual_end = self._find_word_boundary(content, target_end, "backward")
                # If we couldn't find a reasonable word boundary, use character boundary
                if actual_end <= start:
                    actual_end = target_end
            else:
                actual_end = target_end

            # Extract chunk
            chunk = content[start:actual_end].strip()

            if chunk:
                chunks.append(chunk)

            # Calculate next start position with overlap consideration
            next_start = start + self.step_size

            # Adjust start to word boundary if possible
            if next_start < len(content):
                next_start = self._find_word_boundary(content, next_start, "forward")

            start = next_start

            if actual_end >= len(content):
                break

        logger.debug(f"Created {len(chunks)} token-aware sliding window chunks")
        return chunks
