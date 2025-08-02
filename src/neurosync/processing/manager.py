"""
Processing Manager to orchestrate text preprocessing, chunking, and quality scoring.
"""

from typing import Any, Dict, List

from neurosync.core.logging.logger import get_logger
from neurosync.ingestion.base import IngestionResult
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


class ProcessingManager:
    """
    Orchestrates the entire processing pipeline from raw ingested content to
    a list of clean, structured Chunk objects.
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
                f"Unknown chunking strategy '{strategy}'. Defaulting to recursive."
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
            f"Processed {ingestion_result.source_id} into {len(chunks)} chunks."
        )
        return chunks
