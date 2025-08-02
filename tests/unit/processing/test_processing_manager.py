"""
Tests for the ProcessingManager with all chunking strategies.

This module contains comprehensive tests for the ProcessingManager class,
verifying its ability to orchestrate the complete processing pipeline
with various chunking strategies and preprocessing steps.
"""

from unittest.mock import patch

import pytest

from neurosync.ingestion.base import IngestionResult
from neurosync.processing.base import Chunk
from neurosync.processing.manager import ProcessingManager


class TestProcessingManager:
    """Test cases for the ProcessingManager class."""

    def test_default_configuration(self):
        """Test ProcessingManager with default configuration."""
        config = {
            "preprocessing": [
                {"name": "html_cleaner", "enabled": True},
                {"name": "whitespace_normalizer", "enabled": True},
            ],
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
        }

        manager = ProcessingManager(config)

        assert manager.config == config
        assert len(manager.preprocessors) == 2
        assert manager.chunker.__class__.__name__ == "RecursiveChunker"

    def test_recursive_chunker_initialization(self):
        """Test initialization with recursive chunker."""
        config = {
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 512,
                "chunk_overlap": 100,
            }
        }

        manager = ProcessingManager(config)

        assert manager.chunker.__class__.__name__ == "RecursiveChunker"
        assert manager.chunker.chunk_size == 512
        assert manager.chunker.chunk_overlap == 100

    def test_semantic_chunker_initialization(self):
        """Test initialization with semantic chunker."""
        config = {
            "chunking": {
                "strategy": "semantic",
                "chunk_size": 800,
                "chunk_overlap": 150,
                "model": "en_core_web_sm",
            }
        }

        manager = ProcessingManager(config)

        assert manager.chunker.__class__.__name__ == "SpacySemanticChunker"
        assert manager.chunker.chunk_size == 800
        assert manager.chunker.chunk_overlap == 150

    def test_sliding_window_chunker_initialization(self):
        """Test initialization with sliding window chunker."""
        config = {
            "chunking": {
                "strategy": "sliding_window",
                "chunk_size": 600,
                "chunk_overlap": 120,
            }
        }

        manager = ProcessingManager(config)

        assert manager.chunker.__class__.__name__ == "SlidingWindowChunker"
        assert manager.chunker.chunk_size == 600
        assert manager.chunker.chunk_overlap == 120

    def test_token_aware_sliding_chunker_initialization(self):
        """Test initialization with token-aware sliding window chunker."""
        config = {
            "chunking": {
                "strategy": "token_aware_sliding",
                "chunk_size": 750,
                "chunk_overlap": 100,
            }
        }

        manager = ProcessingManager(config)

        assert manager.chunker.__class__.__name__ == "TokenAwareSlidingWindowChunker"
        assert manager.chunker.chunk_size == 750
        assert manager.chunker.chunk_overlap == 100

    def test_hierarchical_chunker_initialization(self):
        """Test initialization with hierarchical chunker."""
        config = {
            "chunking": {
                "strategy": "hierarchical",
                "chunk_size": 1200,
                "chunk_overlap": 200,
                "preserve_structure": True,
                "min_section_size": 150,
            }
        }

        manager = ProcessingManager(config)

        assert manager.chunker.__class__.__name__ == "HierarchicalChunker"
        assert manager.chunker.chunk_size == 1200
        assert manager.chunker.chunk_overlap == 200
        assert manager.chunker.preserve_structure is True
        assert manager.chunker.min_section_size == 150

    def test_document_structure_chunker_initialization(self):
        """Test initialization with document structure-aware chunker."""
        config = {
            "chunking": {
                "strategy": "document_structure",
                "chunk_size": 1000,
                "chunk_overlap": 150,
                "ocr_enabled": True,
                "table_extraction": True,
                "preserve_table_structure": True,
                "min_confidence": 0.8,
            }
        }

        manager = ProcessingManager(config)

        assert manager.chunker.__class__.__name__ == "DocumentStructureAwareChunker"
        assert manager.chunker.chunk_size == 1000
        assert manager.chunker.chunk_overlap == 150
        assert manager.chunker.ocr_enabled is True
        assert manager.chunker.table_extraction is True
        assert manager.chunker.preserve_table_structure is True
        assert manager.chunker.min_confidence == 0.8

    def test_unknown_strategy_fallback(self):
        """Test fallback to recursive chunker for unknown strategy."""
        config = {
            "chunking": {
                "strategy": "unknown_strategy",
                "chunk_size": 500,
                "chunk_overlap": 50,
            }
        }

        with patch("neurosync.processing.manager.logger") as mock_logger:
            manager = ProcessingManager(config)

            assert manager.chunker.__class__.__name__ == "RecursiveChunker"
            mock_logger.warning.assert_called_once()

    def test_preprocessing_initialization(self):
        """Test initialization of preprocessing steps."""
        config = {
            "preprocessing": [
                {"name": "html_cleaner", "enabled": True},
                {"name": "whitespace_normalizer", "enabled": False},
                {"name": "html_cleaner", "enabled": True},  # Duplicate
            ]
        }

        manager = ProcessingManager(config)

        # Should have two HTML cleaners (one enabled, one duplicate)
        assert len(manager.preprocessors) == 2
        assert all(p.__class__.__name__ == "HTMLCleaner" for p in manager.preprocessors)

    def test_disabled_preprocessing_steps(self):
        """Test that disabled preprocessing steps are not included."""
        config = {
            "preprocessing": [
                {"name": "html_cleaner", "enabled": False},
                {"name": "whitespace_normalizer", "enabled": False},
            ]
        }

        manager = ProcessingManager(config)

        assert len(manager.preprocessors) == 0

    def test_successful_processing(self):
        """Test successful processing of ingestion result."""
        config = {
            "preprocessing": [{"name": "html_cleaner", "enabled": True}],
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 100,
                "chunk_overlap": 20,
            },
            "filtering": {"min_quality_score": 0.5},
        }

        manager = ProcessingManager(config)

        ingestion_result = IngestionResult(
            success=True,
            content=(
                "This is a test document with enough content to be split "
                "into multiple chunks for testing purposes."
            ),
            metadata={"source": "test_file.txt", "type": "text"},
            source_id="test_source",
        )

        chunks = manager.process(ingestion_result)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.quality_score >= 0.5 for chunk in chunks)

        # Check metadata preservation
        for chunk in chunks:
            assert chunk.source_metadata == ingestion_result.metadata
            assert "language" in chunk.processing_metadata
            assert "chunker" in chunk.processing_metadata
            assert "preprocessing_steps" in chunk.processing_metadata

    def test_failed_ingestion_handling(self):
        """Test handling of failed ingestion results."""
        config = {"chunking": {"strategy": "recursive"}}
        manager = ProcessingManager(config)

        failed_result = IngestionResult(
            success=False, content="", metadata={}, source_id="failed_source"
        )

        with patch("neurosync.processing.manager.logger") as mock_logger:
            chunks = manager.process(failed_result)

            assert chunks == []
            mock_logger.warning.assert_called_once()

    def test_empty_content_handling(self):
        """Test handling of empty content."""
        config = {"chunking": {"strategy": "recursive"}}
        manager = ProcessingManager(config)

        empty_result = IngestionResult(
            success=True,
            content="",
            metadata={"source": "empty.txt"},
            source_id="empty_source",
        )

        chunks = manager.process(empty_result)
        assert chunks == []

    def test_quality_filtering(self):
        """Test quality score filtering."""
        config = {
            "chunking": {"strategy": "recursive", "chunk_size": 50, "chunk_overlap": 0},
            "filtering": {"min_quality_score": 0.8},
        }

        manager = ProcessingManager(config)

        # Create content with varied quality (some very short chunks)
        ingestion_result = IngestionResult(
            success=True,
            content=(
                "Good chunk with sufficient content. Bad. Another good "
                "chunk with enough content to pass quality check."
            ),
            metadata={"source": "test.txt"},
            source_id="quality_test",
        )

        chunks = manager.process(ingestion_result)

        # Should filter out low-quality chunks
        assert all(chunk.quality_score >= 0.8 for chunk in chunks)

    def test_language_detection_integration(self):
        """Test integration with language detection."""
        config = {
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 200,
                "chunk_overlap": 0,
            }
        }

        manager = ProcessingManager(config)

        ingestion_result = IngestionResult(
            success=True,
            content=(
                "This is an English document with sufficient content " "for processing."
            ),
            metadata={"source": "english.txt"},
            source_id="lang_test",
        )

        with patch("neurosync.processing.manager.detect_language") as mock_detect:
            mock_detect.return_value = "en"

            chunks = manager.process(ingestion_result)

            assert len(chunks) > 0
            mock_detect.assert_called_once()

            for chunk in chunks:
                assert chunk.processing_metadata["language"] == "en"

    def test_preprocessing_integration(self):
        """Test integration with preprocessing steps."""
        config = {
            "preprocessing": [
                {"name": "html_cleaner", "enabled": True},
                {"name": "whitespace_normalizer", "enabled": True},
            ],
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 100,
                "chunk_overlap": 0,
            },
        }

        manager = ProcessingManager(config)

        html_content = "<p>This is <b>HTML</b> content  with   extra   spaces.</p>"
        ingestion_result = IngestionResult(
            success=True,
            content=html_content,
            metadata={"source": "html_test.html"},
            source_id="html_test",
        )

        chunks = manager.process(ingestion_result)

        assert len(chunks) > 0
        # Content should be cleaned (no HTML tags, normalized whitespace)
        combined_content = " ".join(chunk.content for chunk in chunks)
        assert "<p>" not in combined_content
        assert "<b>" not in combined_content
        assert "HTML" in combined_content

        # Check preprocessing metadata
        for chunk in chunks:
            preprocessing_steps = chunk.processing_metadata["preprocessing_steps"]
            assert "HTMLCleaner" in preprocessing_steps
            assert "WhitespaceNormalizer" in preprocessing_steps

    def test_chunker_specific_processing(self):
        """Test processing with different chunking strategies."""
        base_content = """# Document Title

This is the introduction paragraph.

## Section 1

Content for section 1 with detailed information.

## Section 2

Content for section 2 with more details."""

        ingestion_result = IngestionResult(
            success=True,
            content=base_content,
            metadata={"source": "structured_doc.md"},
            source_id="structured_test",
        )

        # Test hierarchical chunker
        hierarchical_config = {
            "chunking": {
                "strategy": "hierarchical",
                "chunk_size": 200,
                "chunk_overlap": 50,
            }
        }

        hierarchical_manager = ProcessingManager(hierarchical_config)
        hierarchical_chunks = hierarchical_manager.process(ingestion_result)

        assert len(hierarchical_chunks) > 0
        assert any("Document Title" in chunk.content for chunk in hierarchical_chunks)

        # Test document structure chunker
        doc_structure_config = {
            "chunking": {
                "strategy": "document_structure",
                "chunk_size": 200,
                "chunk_overlap": 50,
            }
        }

        doc_structure_manager = ProcessingManager(doc_structure_config)
        doc_structure_chunks = doc_structure_manager.process(ingestion_result)

        assert len(doc_structure_chunks) > 0

    def test_chunk_sequence_numbering(self):
        """Test that chunks are properly numbered in sequence."""
        config = {
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 50,
                "chunk_overlap": 10,
            }
        }

        manager = ProcessingManager(config)

        ingestion_result = IngestionResult(
            success=True,
            content=(
                "This is a longer document that will be split into "
                "multiple chunks for sequence testing purposes."
            ),
            metadata={"source": "sequence_test.txt"},
            source_id="sequence_test",
        )

        chunks = manager.process(ingestion_result)

        assert len(chunks) > 1

        # Check sequence numbering
        for i, chunk in enumerate(chunks):
            assert chunk.sequence_num == i

    def test_quality_score_calculation(self):
        """Test quality score calculation logic."""
        config = {"chunking": {"strategy": "recursive"}}
        manager = ProcessingManager(config)

        # Test various content types
        high_quality = (
            "This is a substantial chunk with good content and sufficient length."
        )
        medium_quality = "Short but has alphabetic content."
        low_quality = "12345"
        very_low_quality = "1"

        assert manager._calculate_quality_score(high_quality) == 1.0
        assert manager._calculate_quality_score(medium_quality) < 1.0
        assert manager._calculate_quality_score(low_quality) < 0.5
        assert manager._calculate_quality_score(very_low_quality) < 0.5

    def test_complete_pipeline_integration(self):
        """Test complete processing pipeline integration."""
        config = {
            "preprocessing": [
                {"name": "html_cleaner", "enabled": True},
                {"name": "whitespace_normalizer", "enabled": True},
            ],
            "chunking": {
                "strategy": "hierarchical",
                "chunk_size": 300,
                "chunk_overlap": 50,
                "preserve_structure": True,
            },
            "filtering": {
                "min_quality_score": 0.3,
            },
        }

        manager = ProcessingManager(config)

        complex_content = """
        <html>
        <body>
        <h1>Document    Title</h1>
        <p>This   is   the   intro   paragraph   with   <b>HTML</b>   formatting.</p>

        <h2>Section   1</h2>
        <p>Content for section 1 with detailed information and sufficient length.</p>

        <h2>Section   2</h2>
        <p>Content for section 2 with more comprehensive details and examples.</p>
        </body>
        </html>
        """

        ingestion_result = IngestionResult(
            success=True,
            content=complex_content,
            metadata={"source": "complex_doc.html", "format": "html"},
            source_id="complex_test",
        )

        with patch("neurosync.processing.manager.detect_language") as mock_detect:
            mock_detect.return_value = "en"

            chunks = manager.process(ingestion_result)

            assert len(chunks) > 0

            # Verify complete processing
            for chunk in chunks:
                # Content should be cleaned
                assert "<html>" not in chunk.content
                assert "<body>" not in chunk.content
                assert "<h1>" not in chunk.content

                # Should have normalized whitespace
                assert "   " not in chunk.content

                # Should have proper metadata
                assert chunk.source_metadata["source"] == "complex_doc.html"
                assert chunk.processing_metadata["language"] == "en"
                assert chunk.processing_metadata["chunker"] == "HierarchicalChunker"
                assert "HTMLCleaner" in chunk.processing_metadata["preprocessing_steps"]
                assert (
                    "WhitespaceNormalizer"
                    in chunk.processing_metadata["preprocessing_steps"]
                )

                # Should meet quality threshold
                assert chunk.quality_score >= 0.3

                # Should have proper sequence
                assert isinstance(chunk.sequence_num, int)


if __name__ == "__main__":
    pytest.main([__file__])
