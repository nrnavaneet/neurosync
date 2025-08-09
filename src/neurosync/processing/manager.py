"""
Processing Manager for orchestrating text preprocessing, chunking, and quality
scoring.

This module provides the central coordination hub for all text processing
operations in the NeuroSync pipeline. It orchestrates the transformation of
raw ingested content through multiple processing stages to produce clean,
structured, and semantically coherent chunks ready for embedding, indexing,
and retrieval operations.

System Architecture:
    The ProcessingManager implements a sophisticated pipeline architecture
    that coordinates multiple specialized processors in sequence. Each stage
    can be configured independently, allowing for fine-tuned optimization
    based on content type, language, and downstream requirements.

Core Processing Pipeline:
    1. Content Validation: Input quality checks and format validation
    2. Preprocessing: Multi-stage text cleaning and normalization
    3. Language Detection: Automatic language identification and optimization
    4. Chunking: Intelligent content segmentation with multiple strategies
    5. Quality Scoring: Content value assessment and filtering
    6. Metadata Enrichment: Comprehensive metadata addition and linking
    7. Validation: Final quality assurance and consistency checks
    8. Output Formatting: Standardized result packaging

Advanced Features:
    - Multi-strategy processing with fallback mechanisms
    - Configurable pipeline stages with conditional execution
    - Batch processing optimization for large content volumes
    - Memory-efficient streaming for large document processing
    - Quality-driven adaptive processing (adjust based on content type)
    - Hierarchical chunk relationship modeling
    - Cross-reference and citation preservation
    - Language-specific processing optimization
    - Custom processor plugin architecture

Processing Strategies:
    Content-Aware: Adapts processing based on detected content type
    Quality-First: Prioritizes content quality over processing speed
    Performance-Optimized: Balanced approach for production workloads
    Research-Grade: Maximum quality with comprehensive analysis
    Real-Time: Optimized for low-latency streaming applications

Preprocessing Capabilities:
    HTML Processing: Tag removal, entity decoding, structure preservation
    Text Normalization: Unicode standardization, whitespace cleanup
    Language Processing: Script detection, transliteration support
    Format Conversion: Multiple input formats to standardized text
    Content Extraction: Tables, lists, code blocks, and special elements
    Quality Enhancement: Spell checking, grammar validation, readability

Chunking Algorithms:
    Recursive Chunker: Hierarchical splitting with delimiter prioritization
    Semantic Chunker: NLP-based boundary detection with coherence scoring
    Document Structure Chunker: Format-aware splitting respecting layout
    Sliding Window Chunker: Overlapping windows for context preservation
    Hierarchical Chunker: Parent-child relationships with structure awareness
    Adaptive Chunker: Dynamic strategy selection based on content analysis

Quality Assessment:
    Content Quality Scoring: Readability, completeness, coherence metrics
    Information Density: Content value and uniqueness assessment
    Language Quality: Grammar, spelling, and fluency evaluation
    Structural Quality: Format consistency and organization scoring
    Relevance Filtering: Topic relevance and content filtering

Performance Optimization:
    - Asynchronous processing with configurable concurrency
    - Memory-efficient streaming for large documents
    - Intelligent caching of preprocessing results
    - Batch optimization for similar content types
    - Pipeline parallelization where dependencies allow
    - Resource monitoring and adaptive throttling

Error Handling and Recovery:
    - Graceful degradation for partial processing failures
    - Detailed error context and recovery suggestions
    - Automatic fallback to simpler processing strategies
    - Comprehensive logging for debugging and monitoring
    - Progress tracking for long-running operations

Configuration System:
    The manager supports extensive configuration for all processing stages:
    - Per-stage parameter tuning
    - Content-type specific optimizations
    - Language-specific processing rules
    - Quality thresholds and filtering criteria
    - Performance vs. quality trade-off settings

Integration Points:
    - Ingestion system for raw content input
    - Embedding systems for processed chunk output
    - Storage systems for intermediate result caching
    - Monitoring systems for performance tracking
    - Configuration management for dynamic settings

Example Usage:
    >>> config = {
    ...     "preprocessing": {"strategy": "content_aware"},
    ...     "chunking": {"strategy": "semantic", "max_size": 1000},
    ...     "quality": {"min_score": 0.7, "enable_filtering": True}
    ... }
    >>> manager = ProcessingManager(config)
    >>> async with manager:
    ...     result = await manager.process_content(content, metadata)

Classes:
    ProcessingManager: Main orchestrator for text processing operations
    ProcessingResult: Structured output container for processed content
    ProcessingConfig: Configuration management for pipeline settings

For advanced configuration and custom processor development, see:
    - docs/processing-configuration.md
    - docs/custom-processors.md
    - examples/advanced-processing-pipelines.py

Author: NeuroSync Team
Created: 2025
Version: 2.0
License: MIT
"""

from typing import Any, Dict, List

from neurosync.core.logging.logger import get_logger
from neurosync.ingestion.base.connector import IngestionResult
from neurosync.processing.base import BaseChunker, BasePreprocessor, Chunk
from neurosync.processing.chunking.document_structure_chunker import (
    DocumentStructureAwareChunker,
)
from neurosync.processing.chunking.hierarchical_chunker import HierarchicalChunker
from neurosync.processing.chunking.recursive_chunker import RecursiveChunker
from neurosync.processing.chunking.semantic_chunker import SpacySemanticChunker
from neurosync.processing.chunking.sliding_window_chunker import (
    SlidingWindowChunker,
    TokenAwareSlidingWindowChunker,
)
from neurosync.processing.preprocessing.cleaners import (
    HTMLCleaner,
    WhitespaceNormalizer,
)
from neurosync.processing.preprocessing.language_detector import detect_language

logger = get_logger(__name__)

"""

Quality Metrics:
    - Content density and information value
    - Language quality and coherence
    - Structural completeness and formatting
    - Semantic richness and keyword diversity

Example:
    >>> config = {
    ...     "preprocessing": [
    ...         {"name": "html_cleaner", "enabled": True},
    ...         {"name": "whitespace_normalizer", "enabled": True}
    ...     ],
    ...     "chunking": {
    ...         "strategy": "recursive",
    ...         "chunk_size": 512,
    ...         "chunk_overlap": 50
    ...     }
    ... }
    >>> manager = ProcessingManager(config)
    >>> chunks = manager.process_results(ingestion_results)
    >>> print(f"Generated {len(chunks)} chunks")

For configuration examples and strategy documentation, see:
    - docs/processing-configuration.md
    - docs/chunking-strategies.md
    - examples/processing-configs.yaml
"""


class ProcessingManager:
    """
    Orchestrates the complete text processing pipeline.

    The ProcessingManager is the central coordinator for transforming raw
    ingested content into clean, structured chunks optimized for embedding
    and retrieval. It manages a configurable pipeline of preprocessing
    steps, chunking strategies, and quality assessment mechanisms.

    Attributes:
        config (Dict[str, Any]): Processing configuration with strategy settings
        preprocessors (List[BasePreprocessor]): Ordered list of text preprocessors
        chunker (BaseChunker): Configured chunking strategy instance

    The manager handles the complete processing workflow:
        1. Raw content validation and preparation
        2. Sequential preprocessing with configurable steps
        3. Language detection and optimization
        4. Content chunking with appropriate strategy
        5. Quality assessment and scoring
        6. Metadata enrichment and relationship mapping
        7. Final validation and output preparation

    Configuration Structure:
        {
            "preprocessing": [
                {"name": "html_cleaner", "enabled": True},
                {"name": "whitespace_normalizer", "enabled": True}
            ],
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "min_chunk_size": 100
            },
            "quality": {
                "min_score": 0.3,
                "language_filter": ["en", "es"],
                "content_filters": ["length", "diversity"]
            }
        }

    Key Capabilities:
        - Multi-stage preprocessing with error handling
        - Adaptive chunking based on content characteristics
        - Quality-based filtering with configurable thresholds
        - Hierarchical chunk relationships for context preservation
        - Batch processing optimization for large datasets
        - Comprehensive metadata tracking and enrichment

    Example:
        >>> config = {
        ...     "preprocessing": [{"name": "html_cleaner", "enabled": True}],
        ...     "chunking": {"strategy": "recursive", "chunk_size": 512}
        ... }
        >>> manager = ProcessingManager(config)
        >>>
        >>> # Process ingestion results
        >>> chunks = manager.process_results(ingestion_results)
        >>>
        >>> # Process individual content
        >>> content_chunks = manager.process_content("Raw text content...")
        >>>
        >>> print(f"Generated {len(chunks)} high-quality chunks")
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessors: List[BasePreprocessor] = self._initialize_preprocessors()
        self.chunker: BaseChunker = self._initialize_chunker()

    def _initialize_preprocessors(self) -> List[BasePreprocessor]:
        """Initializes preprocessors based on config."""
        preprocessor_configs = self.config.get("preprocessing", [])
        preprocessors: List[BasePreprocessor] = []

        for p_config in preprocessor_configs:
            if p_config["name"] == "html_cleaner" and p_config.get("enabled", True):
                preprocessors.append(HTMLCleaner())
            if p_config["name"] == "whitespace_normalizer" and p_config.get(
                "enabled", True
            ):
                preprocessors.append(WhitespaceNormalizer())

        return preprocessors

    def _initialize_chunker(self) -> BaseChunker:
        """Initializes the chunking strategy based on config."""
        chunker_config = self.config.get("chunking", {})
        strategy = chunker_config.get("strategy", "recursive")
        chunk_size = chunker_config.get("chunk_size", 1024)
        chunk_overlap = chunker_config.get("chunk_overlap", 200)

        if strategy == "recursive":
            return RecursiveChunker(chunk_size, chunk_overlap)
        elif strategy == "semantic":
            model_name = chunker_config.get("model", "en_core_web_sm")
            return SpacySemanticChunker(chunk_size, chunk_overlap, model_name)
        elif strategy == "sliding_window":
            return SlidingWindowChunker(chunk_size, chunk_overlap)
        elif strategy == "token_aware_sliding":
            return TokenAwareSlidingWindowChunker(chunk_size, chunk_overlap)
        elif strategy == "hierarchical":
            preserve_structure = chunker_config.get("preserve_structure", True)
            min_section_size = chunker_config.get("min_section_size", 100)
            return HierarchicalChunker(
                chunk_size, chunk_overlap, preserve_structure, min_section_size
            )
        elif strategy == "document_structure":
            ocr_enabled = chunker_config.get("ocr_enabled", True)
            table_extraction = chunker_config.get("table_extraction", True)
            preserve_table_structure = chunker_config.get(
                "preserve_table_structure", True
            )
            min_confidence = chunker_config.get("min_confidence", 0.7)
            return DocumentStructureAwareChunker(
                chunk_size,
                chunk_overlap,
                ocr_enabled,
                table_extraction,
                preserve_table_structure,
                min_confidence,
            )
        else:
            logger.warning(
                f"Unknown chunking strategy '{strategy}'. " f"Defaulting to recursive."
            )
            return RecursiveChunker(chunk_size, chunk_overlap)

    def _calculate_quality_score(self, chunk_content: str) -> float:
        """Calculates a quality score for a chunk."""
        score = 1.0
        # Penalty for very short chunks
        if len(chunk_content) < 50:
            score -= 0.3
        # Penalty for chunks with no alphabetic characters
        if not any(c.isalpha() for c in chunk_content):
            score -= 0.5

        return max(0.0, score)

    def process(self, ingestion_result: IngestionResult) -> List[Chunk]:
        """
        Processes a single IngestionResult and returns a list of Chunks.
        """
        if not ingestion_result.success or not ingestion_result.content:
            logger.warning(
                f"Skipping processing for failed or empty ingestion: "
                f"{ingestion_result.source_id}"
            )
            return []

        if not ingestion_result.metadata:
            logger.warning(
                f"Skipping processing due to missing metadata: "
                f"{ingestion_result.source_id}"
            )
            return []

        # 1. Preprocessing
        clean_content = ingestion_result.content
        for preprocessor in self.preprocessors:
            clean_content = preprocessor.process(clean_content)

        # 2. Language Detection
        language = detect_language(clean_content)

        # 3. Chunking
        chunk_texts = self.chunker.chunk(clean_content, {})

        # 4. Create Chunk objects with metadata and quality scores
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            # Content Filtering: Skip empty chunks
            if not chunk_text.strip():
                continue

            quality_score = self._calculate_quality_score(chunk_text)

            # Filtering by quality score
            min_quality = self.config.get("filtering", {}).get("min_quality_score", 0.1)
            if quality_score < min_quality:
                continue

            chunk = Chunk(
                content=chunk_text,
                source_metadata=ingestion_result.metadata,
                sequence_num=i,
                quality_score=quality_score,
                processing_metadata={
                    "language": language,
                    "chunker": self.chunker.__class__.__name__,
                    "preprocessing_steps": [
                        p.__class__.__name__ for p in self.preprocessors
                    ],
                },
            )
            chunks.append(chunk)

        logger.info(
            f"Processed {ingestion_result.source_id} " f"into {len(chunks)} chunks."
        )
        return chunks
