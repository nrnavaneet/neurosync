"""
NeuroSync Processing Module - Advanced Text Processing and Chunking.

This module provides comprehensive text processing capabilities for transforming
raw ingested content into optimized chunks ready for embedding and retrieval.
It implements multiple processing strategies, quality assessment mechanisms,
and hierarchical content organization.

Core Components:
    - ProcessingManager: Central orchestrator for text processing pipelines
    - Chunk: Structured representation of processed text segments
    - BasePreprocessor: Abstract interface for text cleaning operations
    - BaseChunker: Abstract interface for chunking strategies

Key Features:
    - Multi-stage preprocessing with configurable pipelines
    - Advanced chunking algorithms (recursive, semantic, hierarchical)
    - Quality assessment and content filtering
    - Language detection and multilingual processing
    - Metadata enrichment and relationship mapping
    - Performance optimization for large-scale processing

Processing Strategies:
    - Recursive Chunking: Hierarchical text splitting with multiple delimiters
    - Semantic Chunking: Boundary detection using NLP models
    - Document Structure: Format-aware chunking preserving document hierarchy
    - Sliding Window: Fixed-size overlapping windows for dense coverage
    - Hierarchical: Parent-child relationships for context preservation

Quality Metrics:
    - Content density and information value assessment
    - Language quality and coherence scoring
    - Structural completeness and formatting evaluation
    - Semantic richness and keyword diversity analysis

The processing pipeline transforms raw content through these stages:
    1. Preprocessing: Clean and normalize text content
    2. Language Detection: Identify and optimize for content language
    3. Chunking: Split into semantically coherent segments
    4. Quality Assessment: Score and filter chunks by value
    5. Metadata Enrichment: Add processing context and relationships
    6. Validation: Ensure output quality and consistency

Example:
    >>> from neurosync.processing import ProcessingManager, Chunk
    >>>
    >>> # Initialize processing manager with configuration
    >>> config = {
    ...     "preprocessing": [{"name": "html_cleaner", "enabled": True}],
    ...     "chunking": {"strategy": "recursive", "chunk_size": 512}
    ... }
    >>> manager = ProcessingManager(config)
    >>>
    >>> # Process ingestion results into chunks
    >>> chunks = manager.process_results(ingestion_results)
    >>> print(f"Generated {len(chunks)} processed chunks")
    >>>
    >>> # Access chunk properties
    >>> for chunk in chunks[:3]:
    ...     print(f"Chunk {chunk.chunk_id}: {len(chunk.content)} chars")
    ...     print(f"Quality: {chunk.quality_score}")

For detailed configuration and strategy documentation, see:
    - docs/processing-configuration.md
    - docs/chunking-strategies.md
    - examples/processing-pipelines.py
"""

from .base import Chunk
from .manager import ProcessingManager

__all__ = [
    "ProcessingManager",
    "Chunk",
]
