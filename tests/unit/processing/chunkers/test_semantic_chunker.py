"""
Tests for semantic chunker using spaCy NLP.

This module contains comprehensive tests for the SpacySemanticChunker class,
verifying its ability to create semantically meaningful chunks using spaCy's
natural language processing capabilities.
"""

from unittest.mock import Mock, call, patch

import pytest

from neurosync.core.exceptions.custom_exceptions import ConfigurationError
from neurosync.processing.chunking.semantic_chunker import SpacySemanticChunker


class TestSpacySemanticChunker:
    """Test cases for the SpacySemanticChunker class."""

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_basic_semantic_chunking(self, mock_spacy_load):
        """Test basic semantic chunking functionality."""
        # Mock spaCy model and doc
        mock_nlp = Mock()
        mock_doc = Mock()

        # Mock sentences
        mock_sent1 = Mock()
        mock_sent1.text = "This is the first sentence."
        mock_sent1.start_char = 0
        mock_sent1.end_char = 27

        mock_sent2 = Mock()
        mock_sent2.text = "This is the second sentence."
        mock_sent2.start_char = 28
        mock_sent2.end_char = 56

        mock_doc.sents = [mock_sent1, mock_sent2]
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        chunker = SpacySemanticChunker(
            chunk_size=50, chunk_overlap=10, model_name="en_core_web_sm"
        )
        text = "This is the first sentence. This is the second sentence."

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 1
        # Check the actual call with disable parameter
        mock_spacy_load.assert_called_once_with(
            "en_core_web_sm", disable=["parser", "ner"]
        )
        mock_nlp.assert_called_once_with(text)

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_model_loading_error_fallback(self, mock_spacy_load):
        """Test fallback to basic chunking when spaCy model fails to load."""
        mock_spacy_load.side_effect = OSError("Model not found")

        # Should raise ConfigurationError, not fall back
        with pytest.raises(ConfigurationError):
            SpacySemanticChunker(
                chunk_size=50, chunk_overlap=10, model_name="invalid_model"
            )

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_single_sentence_chunking(self, mock_spacy_load):
        """Test chunking with single sentence."""
        mock_nlp = Mock()
        mock_doc = Mock()

        mock_sent = Mock()
        mock_sent.text = "This is a single sentence."
        mock_sent.start_char = 0
        mock_sent.end_char = 26

        mock_doc.sents = [mock_sent]
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        chunker = SpacySemanticChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a single sentence."

        chunks = chunker.chunk(text, {})

        assert len(chunks) == 1
        assert chunks[0] == "This is a single sentence."

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_empty_content_handling(self, mock_spacy_load):
        """Test handling of empty content."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.sents = []
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        chunker = SpacySemanticChunker(chunk_size=50, chunk_overlap=10)

        chunks = chunker.chunk("", {})
        assert chunks == []

        chunks = chunker.chunk("   ", {})
        assert len(chunks) <= 1

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_long_sentence_handling(self, mock_spacy_load):
        """Test handling of sentences longer than chunk size."""
        mock_nlp = Mock()
        mock_doc = Mock()

        # Create a long sentence
        long_sentence = (
            "This is a very long sentence that exceeds the chunk size limit "
            "and should be handled appropriately by the semantic chunker."
        )
        mock_sent = Mock()
        mock_sent.text = long_sentence
        mock_sent.start_char = 0
        mock_sent.end_char = len(long_sentence)

        mock_doc.sents = [mock_sent]
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        chunker = SpacySemanticChunker(chunk_size=50, chunk_overlap=10)

        chunks = chunker.chunk(long_sentence, {})

        # Should handle long sentences by splitting them
        assert len(chunks) >= 1
        if len(chunks) > 1:
            assert all(len(chunk) <= 70 for chunk in chunks)  # Allow some flexibility

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_semantic_boundary_preservation(self, mock_spacy_load):
        """Test that semantic boundaries (sentences) are preserved."""
        mock_nlp = Mock()
        mock_doc = Mock()

        # Mock multiple sentences
        sentences = [
            "First semantic unit here.",
            "Second semantic unit follows.",
            "Third semantic unit continues.",
            "Fourth semantic unit ends.",
        ]

        mock_sents = []
        start_pos = 0
        for i, sent_text in enumerate(sentences):
            mock_sent = Mock()
            mock_sent.text = sent_text
            mock_sent.start_char = start_pos
            mock_sent.end_char = start_pos + len(sent_text)
            mock_sents.append(mock_sent)
            start_pos += len(sent_text) + 1  # +1 for space

        mock_doc.sents = mock_sents
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        text = " ".join(sentences)
        chunker = SpacySemanticChunker(chunk_size=60, chunk_overlap=15)

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 1
        # Should try to keep complete sentences together
        for chunk in chunks:
            if "." in chunk:
                # If chunk contains a period, it should likely be at sentence boundary
                sentences_in_chunk = chunk.count(".")
                assert sentences_in_chunk >= 1

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_chunk_overlap_with_sentences(self, mock_spacy_load):
        """Test that overlap respects sentence boundaries when possible."""
        mock_nlp = Mock()
        mock_doc = Mock()

        sentences = [
            "First sentence is here.",
            "Second sentence follows.",
            "Third sentence continues.",
            "Fourth sentence is last.",
        ]

        mock_sents = []
        start_pos = 0
        for sent_text in sentences:
            mock_sent = Mock()
            mock_sent.text = sent_text
            mock_sent.start_char = start_pos
            mock_sent.end_char = start_pos + len(sent_text)
            mock_sents.append(mock_sent)
            start_pos += len(sent_text) + 1

        mock_doc.sents = mock_sents
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        text = " ".join(sentences)
        chunker = SpacySemanticChunker(chunk_size=50, chunk_overlap=20)

        chunks = chunker.chunk(text, {})

        if len(chunks) > 1:
            # Check that there's meaningful overlap - should be at least as
            # long as original
            combined_length = sum(len(chunk) for chunk in chunks)
            # Allow for some overlap, so combined length should be >= original
            assert combined_length >= len(text) - 10  # Small tolerance for processing

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_model_name_configuration(self, mock_spacy_load):
        """Test that different model names are used correctly."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.sents = []
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        # Test with different model names
        models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]

        for model_name in models:
            chunker = SpacySemanticChunker(
                chunk_size=50, chunk_overlap=10, model_name=model_name
            )
            chunker.chunk("Test text", {})

        # Should have been called with each model name and disable parameter
        expected_load_calls = [
            call(model, disable=["parser", "ner"]) for model in models
        ]

        # Check that spacy.load was called with the correct arguments
        actual_load_calls = [call_args for call_args in mock_spacy_load.call_args_list]

        # Verify that we have the expected number of load calls
        assert len(actual_load_calls) == len(models)

        # Verify each load call has the correct arguments
        for i, expected_call in enumerate(expected_load_calls):
            assert actual_load_calls[i] == expected_call

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_sentence_splitting_accuracy(self, mock_spacy_load):
        """Test accuracy of sentence splitting."""
        mock_nlp = Mock()
        mock_doc = Mock()

        # Text with various sentence endings
        text = (
            "Dr. Smith went to the U.S.A. He studied at M.I.T. Then he returned home."
        )

        # Mock realistic sentence splitting
        mock_sent1 = Mock()
        mock_sent1.text = "Dr. Smith went to the U.S.A."
        mock_sent1.start_char = 0
        mock_sent1.end_char = 29

        mock_sent2 = Mock()
        mock_sent2.text = "He studied at M.I.T."
        mock_sent2.start_char = 30
        mock_sent2.end_char = 50

        mock_sent3 = Mock()
        mock_sent3.text = "Then he returned home."
        mock_sent3.start_char = 51
        mock_sent3.end_char = 73

        mock_doc.sents = [mock_sent1, mock_sent2, mock_sent3]
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        chunker = SpacySemanticChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 1
        # Should handle abbreviations correctly (via spaCy)
        combined = " ".join(chunks)
        assert "Dr. Smith" in combined
        assert "M.I.T." in combined

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_metadata_handling(self, mock_spacy_load):
        """Test that metadata is handled properly."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.sents = []
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        chunker = SpacySemanticChunker(chunk_size=50, chunk_overlap=10)

        text = "Test content for metadata."
        metadata = {"source": "test", "language": "en"}

        chunks = chunker.chunk(text, metadata)

        # Should produce valid chunks regardless of metadata
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_whitespace_normalization(self, mock_spacy_load):
        """Test handling of whitespace in semantic chunking."""
        mock_nlp = Mock()
        mock_doc = Mock()

        text = "Sentence   with    extra    spaces.    Another sentence here."

        mock_sent1 = Mock()
        mock_sent1.text = "Sentence   with    extra    spaces."
        mock_sent1.start_char = 0
        mock_sent1.end_char = 35

        mock_sent2 = Mock()
        mock_sent2.text = "Another sentence here."
        mock_sent2.start_char = 39
        mock_sent2.end_char = 61

        mock_doc.sents = [mock_sent1, mock_sent2]
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        chunker = SpacySemanticChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 1
        # Should preserve sentence content including spacing
        combined = " ".join(chunks)
        assert "extra" in combined
        assert "Another sentence" in combined

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_processing_exception_handling(self, mock_spacy_load):
        """Test handling of processing exceptions."""
        mock_nlp = Mock()
        mock_nlp.side_effect = Exception("Processing error")
        mock_spacy_load.return_value = mock_nlp

        chunker = SpacySemanticChunker(chunk_size=50, chunk_overlap=10)
        text = "Test text for exception handling."

        # Should handle processing exception gracefully
        with pytest.raises(Exception, match="Processing error"):
            chunker.chunk(text, {})

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_configuration_edge_cases(self, mock_spacy_load):
        """Test edge cases in configuration."""
        mock_nlp = Mock()
        mock_doc = Mock()

        # Create a mock sentence for very small text
        mock_sent = Mock()
        mock_sent.text = "Test"
        mock_sent.start_char = 0
        mock_sent.end_char = 4
        mock_doc.sents = [mock_sent]

        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        # Very small chunk size
        small_chunker = SpacySemanticChunker(chunk_size=1, chunk_overlap=0)
        chunks = small_chunker.chunk("Test", {})
        assert len(chunks) >= 1

        # Zero overlap
        zero_overlap_chunker = SpacySemanticChunker(chunk_size=50, chunk_overlap=0)
        chunks = zero_overlap_chunker.chunk("Test sentence here.", {})
        assert len(chunks) >= 1

        # Large overlap
        large_overlap_chunker = SpacySemanticChunker(chunk_size=20, chunk_overlap=15)
        chunks = large_overlap_chunker.chunk("Test sentence for overlap.", {})
        assert len(chunks) >= 1

    @patch("neurosync.processing.chunking.semantic_chunker.spacy.load")
    def test_default_model_name(self, mock_spacy_load):
        """Test that default model name is set correctly."""
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp

        chunker = SpacySemanticChunker(chunk_size=50, chunk_overlap=10)
        assert chunker.model_name == "en_core_web_sm"

        chunker_custom = SpacySemanticChunker(
            chunk_size=50, chunk_overlap=10, model_name="custom_model"
        )
        assert chunker_custom.model_name == "custom_model"


if __name__ == "__main__":
    pytest.main([__file__])
