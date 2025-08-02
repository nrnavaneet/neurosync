"""
Tests for recursive character-based chunker.

This module contains comprehensive tests for the RecursiveChunker class,
verifying its ability to split text using LangChain's recursive character
text splitter while maintaining semantic coherence.
"""

import pytest

from neurosync.processing.chunking.recursive_chunker import RecursiveChunker


class TestRecursiveChunker:
    """Test cases for the RecursiveChunker class."""

    def test_basic_chunking(self):
        """Test basic recursive chunking functionality."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = (
            "This is a test document with multiple sentences. Each sentence "
            "contains some content that should be properly chunked."
        )

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks)

        # Check that content is preserved
        combined_content = " ".join(chunks)
        assert "test document" in combined_content
        assert "properly chunked" in combined_content

    def test_short_text(self):
        """Test chunking with text shorter than chunk size."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        text = "Short text."

        chunks = chunker.chunk(text, {})

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_content(self):
        """Test chunking with empty content."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)

        chunks = chunker.chunk("", {})
        assert chunks == []

        chunks = chunker.chunk("   ", {})
        assert len(chunks) <= 1

    def test_zero_overlap(self):
        """Test chunking with zero overlap."""
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        text = "This is a sentence. This is another sentence. And one more sentence."

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 1
        # With zero overlap, there should be no repeated content
        all_content = "".join(chunks)
        assert len(all_content) <= len(text) + 10  # Allow for some separator variations

    def test_large_overlap(self):
        """Test chunking with large overlap."""
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=25)
        text = (
            "This is a test document with enough content to create multiple "
            "overlapping chunks."
        )

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 1
        # Should have significant overlap between chunks
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            overlap_found = False
            for i in range(len(chunks) - 1):
                current_words = chunks[i].split()
                next_words = chunks[i + 1].split()

                # Look for common words indicating overlap
                if any(word in next_words for word in current_words[-3:]):
                    overlap_found = True
                    break

            assert (
                overlap_found or len(chunks[0]) < 30
            )  # Overlap found or chunks are very short

    def test_paragraph_splitting(self):
        """Test that recursive chunker handles paragraph breaks appropriately."""
        chunker = RecursiveChunker(chunk_size=80, chunk_overlap=20)

        text = """First paragraph with some content.

Second paragraph with different content.

Third paragraph with more information."""

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 1
        # Should maintain some paragraph structure when possible
        combined = " ".join(chunks)
        assert "First paragraph" in combined
        assert "Second paragraph" in combined
        assert "Third paragraph" in combined

    def test_sentence_boundary_respect(self):
        """Test that chunker tries to respect sentence boundaries."""
        chunker = RecursiveChunker(chunk_size=40, chunk_overlap=10)

        text = (
            "First sentence here. Second sentence follows. Third sentence "
            "continues. Fourth sentence ends."
        )

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 1
        # Most chunks should end with sentence terminators when possible
        sentence_endings = 0
        for chunk in chunks:
            if chunk.rstrip().endswith("."):
                sentence_endings += 1

        # At least some chunks should end with periods (sentence boundaries)
        assert sentence_endings > 0

    def test_word_boundary_preservation(self):
        """Test that chunker avoids breaking words when possible."""
        chunker = RecursiveChunker(chunk_size=25, chunk_overlap=5)

        text = (
            "This contains some supercalifragilisticexpialidocious words "
            "that are quite long."
        )

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 1
        # Most chunks should not start or end mid-word (except for very long words)
        clean_boundaries = 0
        for chunk in chunks:
            chunk_trimmed = chunk.strip()
            if chunk_trimmed:
                # Check if chunk starts and ends cleanly (not mid-word)
                starts_clean = (
                    chunk_trimmed[0].isupper() or not chunk_trimmed[0].isalpha()
                )
                ends_clean = not chunk_trimmed[-1].isalpha() or chunk_trimmed.endswith(
                    "."
                )

                if starts_clean or ends_clean:
                    clean_boundaries += 1

        # Should have some clean word boundaries
        assert clean_boundaries > 0

    def test_metadata_handling(self):
        """Test that metadata is handled properly."""
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)

        text = "Test content with metadata handling."
        metadata = {"source": "test", "type": "document"}

        chunks = chunker.chunk(text, metadata)

        # The recursive chunker doesn't modify behavior based on metadata,
        # but should still produce valid chunks
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_special_characters(self):
        """Test handling of special characters and formatting."""
        chunker = RecursiveChunker(chunk_size=40, chunk_overlap=10)

        text = (
            "Text with special chars: @#$%^&*()! And numbers: 12345. "
            "Also unicode: café, naïve, résumé."
        )

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 1
        # Special characters should be preserved
        combined = " ".join(chunks)
        assert "@#$%^&*()" in combined
        assert "12345" in combined
        assert "café" in combined

    def test_code_content(self):
        """Test handling of code-like content."""
        chunker = RecursiveChunker(chunk_size=60, chunk_overlap=15)

        text = """def function_name(param1, param2):
    result = param1 + param2
    return result

class MyClass:
    def __init__(self):
        self.value = 42"""

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 1
        # Code structure should be somewhat preserved
        combined = " ".join(chunks)
        assert "def function_name" in combined
        assert "class MyClass" in combined

    def test_multilingual_content(self):
        """Test handling of multilingual text."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)

        text = (
            "English text here. Texto en español aquí. Texte français ici. "
            "Deutsche Text hier."
        )

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 1
        # All languages should be preserved
        combined = " ".join(chunks)
        assert "English" in combined
        assert "español" in combined
        assert "français" in combined
        assert "Deutsche" in combined

    def test_chunker_configuration(self):
        """Test chunker with different configurations."""
        # Small chunks
        small_chunker = RecursiveChunker(chunk_size=10, chunk_overlap=2)
        text = "This is a test sentence for configuration testing."

        small_chunks = small_chunker.chunk(text, {})
        assert all(len(chunk) <= 10 for chunk in small_chunks)

        # Large chunks
        large_chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        large_chunks = large_chunker.chunk(text, {})

        # Should have fewer, larger chunks
        assert len(large_chunks) <= len(small_chunks)
        if large_chunks:
            assert len(large_chunks[0]) >= len(small_chunks[0])

    def test_very_long_text(self):
        """Test chunking of very long text."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)

        # Create long text
        long_text = " ".join(
            [f"Sentence number {i} with some content." for i in range(50)]
        )

        chunks = chunker.chunk(long_text, {})

        assert len(chunks) > 5  # Should create multiple chunks
        assert all(len(chunk) <= 100 for chunk in chunks)

        # Content should be preserved
        combined = " ".join(chunks)
        assert "Sentence number 1" in combined
        assert "Sentence number 49" in combined

    def test_single_word_chunks(self):
        """Test edge case with very small chunk size."""
        chunker = RecursiveChunker(chunk_size=5, chunk_overlap=1)
        text = "Test"

        chunks = chunker.chunk(text, {})

        assert len(chunks) == 1
        assert chunks[0] == "Test"

    def test_whitespace_handling(self):
        """Test proper handling of various whitespace patterns."""
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)

        text = "Word1    Word2\n\nWord3\tWord4     Word5"

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 1
        # Whitespace patterns should be reasonably preserved
        combined = " ".join(chunks)
        assert "Word1" in combined
        assert "Word5" in combined


if __name__ == "__main__":
    pytest.main([__file__])
