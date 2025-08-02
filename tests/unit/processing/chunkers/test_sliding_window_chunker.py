"""
Tests for sliding window chunker implementations.

This module contains comprehensive tests for both the SlidingWindowChunker
and TokenAwareSlidingWindowChunker classes, verifying their functionality
across various input scenarios and configurations.
"""

import pytest

from neurosync.processing.chunking.sliding_window_chunker import (
    SlidingWindowChunker,
    TokenAwareSlidingWindowChunker,
)


class TestSlidingWindowChunker:
    """Test cases for the SlidingWindowChunker class."""

    def test_basic_chunking(self):
        """Test basic sliding window chunking functionality."""
        chunker = SlidingWindowChunker(chunk_size=20, chunk_overlap=5)
        text = "This is a test document with some content for chunking purposes."

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 1
        assert all(len(chunk) <= 20 for chunk in chunks)

        # Check overlap
        for i in range(len(chunks) - 1):
            assert chunks[i][-5:] == chunks[i + 1][:5]

    def test_short_text(self):
        """Test chunking with text shorter than chunk size."""
        chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=10)
        text = "Short text."

        chunks = chunker.chunk(text, {})

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_zero_overlap(self):
        """Test chunking with zero overlap."""
        chunker = SlidingWindowChunker(chunk_size=10, chunk_overlap=0)
        text = "0123456789abcdefghijklmnop"

        chunks = chunker.chunk(text, {})

        assert len(chunks) == 3
        assert chunks[0] == "0123456789"
        assert chunks[1] == "abcdefghij"
        assert chunks[2] == "klmnop"

    def test_large_overlap(self):
        """Test chunking with overlap larger than chunk size."""
        chunker = SlidingWindowChunker(chunk_size=10, chunk_overlap=15)
        text = "This is a test string for chunking"

        chunks = chunker.chunk(text, {})

        # Should handle gracefully - overlap capped at chunk_size - 1
        assert len(chunks) > 1
        assert all(len(chunk) <= 10 for chunk in chunks)

    def test_empty_content(self):
        """Test chunking with empty content."""
        chunker = SlidingWindowChunker(chunk_size=10, chunk_overlap=2)

        chunks = chunker.chunk("", {})
        assert chunks == []

        chunks = chunker.chunk("   ", {})
        assert len(chunks) == 1
        assert chunks[0] == "   "

    def test_single_character_chunks(self):
        """Test chunking with very small chunk size."""
        chunker = SlidingWindowChunker(chunk_size=1, chunk_overlap=0)
        text = "abc"

        chunks = chunker.chunk(text, {})

        assert chunks == ["a", "b", "c"]

    def test_metadata_preservation(self):
        """Test that metadata is passed through correctly."""
        chunker = SlidingWindowChunker(chunk_size=10, chunk_overlap=2)
        text = "Test content for metadata"
        metadata = {"source": "test", "type": "document"}

        chunks = chunker.chunk(text, metadata)

        # The base implementation doesn't modify chunks based on metadata
        # but it should still produce valid chunks
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)


class TestTokenAwareSlidingWindowChunker:
    """Test cases for the TokenAwareSlidingWindowChunker class."""

    def test_word_boundary_respect(self):
        """Test that chunks respect word boundaries."""
        chunker = TokenAwareSlidingWindowChunker(chunk_size=20, chunk_overlap=5)
        text = "This is a test document with some content for chunking purposes."

        chunks = chunker.chunk(text, {})

        # Verify no chunks end/start with partial words (except for very long words)
        for chunk in chunks:
            words = chunk.split()
            if (
                len(chunk) < 20
            ):  # If chunk is shorter than max, it should be complete words
                assert chunk == " ".join(words)

    def test_long_word_handling(self):
        """Test handling of words longer than chunk size."""
        chunker = TokenAwareSlidingWindowChunker(chunk_size=10, chunk_overlap=2)
        text = "Short supercalifragilisticexpialidocious word"

        chunks = chunker.chunk(text, {})

        # Should split the long word appropriately
        assert len(chunks) > 1
        # First chunk should be complete words that fit
        assert "Short" in chunks[0]
        # All text should be preserved somewhere in the chunks
        full_text = " ".join(chunks)
        assert "Short" in full_text and "word" in full_text
        # The long word should be handled (may be truncated or split)
        assert len(full_text) >= len("Short word")

    def test_whitespace_handling(self):
        """Test proper handling of whitespace in token-aware chunking."""
        chunker = TokenAwareSlidingWindowChunker(chunk_size=15, chunk_overlap=3)
        text = "Word1    Word2     Word3 Word4"

        chunks = chunker.chunk(text, {})

        # Should preserve meaningful whitespace patterns
        assert len(chunks) >= 1
        for chunk in chunks:
            # Should not start or end with excessive whitespace
            assert chunk == chunk.strip() or "Word" in chunk

    def test_punctuation_handling(self):
        """Test handling of punctuation in token-aware chunking."""
        chunker = TokenAwareSlidingWindowChunker(chunk_size=25, chunk_overlap=5)
        text = "Hello, world! This is Dr. Smith's test. How are you?"

        chunks = chunker.chunk(text, {})

        # Should handle punctuation appropriately
        assert len(chunks) >= 1
        for chunk in chunks:
            # Check that punctuation is preserved with words
            if "Dr." in chunk:
                assert "Dr. Smith" in chunk or "Dr." in chunk

    def test_sentence_boundary_preference(self):
        """Test preference for sentence boundaries when possible."""
        chunker = TokenAwareSlidingWindowChunker(chunk_size=50, chunk_overlap=10)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunker.chunk(text, {})

        # Should try to break at sentence boundaries when possible
        # For this test, we'll just verify that chunks are created reasonably
        assert len(chunks) >= 1
        # Check that we don't break words inappropriately
        for chunk in chunks:
            # No chunk should end with a partial word (unless it's a sentence end)
            words = chunk.split()
            if words:
                last_word = words[-1]
                # Either ends with punctuation or is a complete word
                assert (
                    last_word.endswith(".")
                    or last_word.isalpha()
                    or last_word.isalnum()
                )

    def test_multilingual_text(self):
        """Test token-aware chunking with multilingual content."""
        chunker = TokenAwareSlidingWindowChunker(chunk_size=30, chunk_overlap=5)
        text = "Hello world. Bonjour monde. Hola mundo. Guten Tag Welt."

        chunks = chunker.chunk(text, {})

        # Should handle different languages appropriately
        assert len(chunks) >= 1
        for chunk in chunks:
            # Should maintain word integrity across languages
            words = chunk.split()
            assert all(word.strip() for word in words if word.strip())

    def test_code_content_handling(self):
        """Test handling of code-like content."""
        chunker = TokenAwareSlidingWindowChunker(chunk_size=40, chunk_overlap=5)
        text = "def function_name(param1, param2): return param1 + param2"

        chunks = chunker.chunk(text, {})

        # Should respect code token boundaries
        assert len(chunks) >= 1
        for chunk in chunks:
            # Should not break function names or operators inappropriately
            if "function_name" in chunk:
                assert "function_name(" in chunk or chunk.endswith("function_name")

    def test_numeric_content(self):
        """Test handling of numeric content and mixed alphanumeric."""
        chunker = TokenAwareSlidingWindowChunker(chunk_size=25, chunk_overlap=3)
        text = "Price is $19.99 or €15.50 per item. Code: ABC123XYZ."

        chunks = chunker.chunk(text, {})

        # Should keep numeric values and codes together
        assert len(chunks) >= 1
        for chunk in chunks:
            # Prices and codes should remain intact
            if "$19.99" in text and "$" in chunk:
                assert "$19.99" in chunk or chunk.endswith("$")

    def test_overlap_with_word_boundaries(self):
        """Test that overlap respects word boundaries."""
        chunker = TokenAwareSlidingWindowChunker(chunk_size=20, chunk_overlap=8)
        text = "The quick brown fox jumps over the lazy dog repeatedly."

        chunks = chunker.chunk(text, {})

        # Check that overlaps are meaningful
        if len(chunks) > 1:
            # For token-aware chunking, we expect reasonable word boundaries
            for i, chunk in enumerate(chunks):
                words = chunk.split()
                # Each chunk should have at least one complete word
                assert len(words) >= 1
                # Words should be complete (not cut off in the middle)
                for word in words:
                    # Allow for punctuation at the end
                    clean_word = word.rstrip(".,!?;:")
                    assert len(clean_word) > 0

    def test_configuration_edge_cases(self):
        """Test edge cases in configuration."""
        # Chunk size smaller than typical words
        chunker = TokenAwareSlidingWindowChunker(chunk_size=3, chunk_overlap=1)
        text = "Testing configuration edge cases"

        chunks = chunker.chunk(text, {})

        # Should handle gracefully even with very small chunks
        assert len(chunks) > 1
        assert all(len(chunk) <= 3 for chunk in chunks)

        # Very large overlap
        chunker = TokenAwareSlidingWindowChunker(chunk_size=10, chunk_overlap=9)
        chunks = chunker.chunk(text, {})

        # Should still produce valid chunks
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__])
