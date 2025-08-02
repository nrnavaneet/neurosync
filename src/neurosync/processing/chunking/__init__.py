"""
NeuroSync Chunking Module - Smart Chunking Strategies

This module provides various chunking strategies for processing text content
into optimal chunks for LLM consumption and embedding generation.

Available Chunkers:
    - RecursiveChunker: Character-based recursive splitting using LangChain
    - SpacySemanticChunker: Sentence-based semantic chunking using spaCy
    - SlidingWindowChunker: Sliding window approach with configurable overlap
    - TokenAwareSlidingWindowChunker: Token-aware sliding window respecting
      word boundaries
    - HierarchicalChunker: Document structure-aware chunking
    - DocumentStructureAwareChunker: Advanced OCR and table extraction chunker

Key Features:
    - Multiple chunking strategies optimized for different content types
    - Configurable chunk sizes and overlap settings
    - Semantic awareness for better context preservation
    - Integration with spaCy for advanced NLP processing
    - Quality scoring and filtering capabilities

Author: NeuroSync Team
Created: 2025
See Also: neurosync.processing.base for base classes and interfaces
"""

from .document_structure_chunker import DocumentStructureAwareChunker, StructuredElement
from .hierarchical_chunker import DocumentStructureAnalyzer, HierarchicalChunker
from .recursive_chunker import RecursiveChunker
from .semantic_chunker import SpacySemanticChunker
from .sliding_window_chunker import SlidingWindowChunker, TokenAwareSlidingWindowChunker

__all__ = [
    "RecursiveChunker",
    "SpacySemanticChunker",
    "SlidingWindowChunker",
    "TokenAwareSlidingWindowChunker",
    "HierarchicalChunker",
    "DocumentStructureAnalyzer",
    "DocumentStructureAwareChunker",
    "StructuredElement",
]
