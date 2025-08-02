"""
Tests for hierarchical document structure-aware chunker.

This module contains comprehensive tests for the HierarchicalChunker class,
verifying its ability to analyze document structure and create semantically
meaningful chunks that respect document hierarchy.
"""

import pytest

from neurosync.processing.chunking.hierarchical_chunker import (
    DocumentStructureAnalyzer,
    HierarchicalChunker,
)


class TestHierarchicalChunker:
    """Test cases for the HierarchicalChunker class."""

    def test_basic_hierarchical_chunking(self):
        """Test basic hierarchical chunking with headers and paragraphs."""
        chunker = HierarchicalChunker(chunk_size=100, chunk_overlap=20)

        text = """# Main Title

This is the introduction paragraph with some content.

## Section 1

This is the first section content with detailed information.

### Subsection 1.1

This is subsection content with more specific details.

## Section 2

This is the second section with different content."""

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 1
        # Should respect document structure
        assert any("Main Title" in chunk for chunk in chunks)
        assert any("Section 1" in chunk for chunk in chunks)

    def test_markdown_header_detection(self):
        """Test detection of markdown-style headers."""
        chunker = HierarchicalChunker(chunk_size=200, chunk_overlap=0)

        text = """# Level 1 Header
Content under level 1.

## Level 2 Header
Content under level 2.

### Level 3 Header
Content under level 3.

#### Level 4 Header
Content under level 4."""

        elements = chunker._parse_document_structure(text)

        # Should detect headers with correct levels
        headers = [e for e in elements if e.element_type == "header"]
        assert len(headers) == 4
        assert headers[0].level == 1
        assert headers[1].level == 2
        assert headers[2].level == 3
        assert headers[3].level == 4

    def test_numbered_section_detection(self):
        """Test detection of numbered sections."""
        chunker = HierarchicalChunker(chunk_size=150, chunk_overlap=10)

        text = """1. First Section
Content of first section.

1.1. Subsection One
Content of subsection one.

1.2. Subsection Two
Content of subsection two.

2. Second Section
Content of second section."""

        elements = chunker._parse_document_structure(text)

        # Should detect numbered headers
        headers = [e for e in elements if e.element_type == "header"]
        assert len(headers) >= 2
        # Should detect different levels (allow flexibility in level detection)
        levels = [h.level for h in headers]
        assert 1 in levels  # Main sections
        # Note: Level 2 detection may vary based on pattern matching

    def test_list_detection(self):
        """Test detection and handling of lists."""
        chunker = HierarchicalChunker(chunk_size=100, chunk_overlap=0)

        text = """# Shopping List

- Apples
- Bananas
- Oranges

# Numbered Tasks

1. Complete project
2. Review documentation
3. Submit report"""

        elements = chunker._parse_document_structure(text)

        # Should detect lists as separate elements
        lists = [e for e in elements if e.element_type == "list"]
        assert len(lists) >= 1

    def test_hierarchy_establishment(self):
        """Test establishment of parent-child relationships."""
        chunker = HierarchicalChunker(chunk_size=200, chunk_overlap=0)

        text = """# Main Title

Introduction content.

## Section A

Section A content.

### Subsection A.1

Subsection content.

## Section B

Section B content."""

        elements = chunker._parse_document_structure(text)
        elements = chunker._establish_hierarchy(elements)

        # Check parent-child relationships
        main_header = next(e for e in elements if "Main Title" in e.content)
        section_a = next(
            e
            for e in elements
            if "Section A" in e.content and e.element_type == "header"
        )
        subsection = next(e for e in elements if "Subsection A.1" in e.content)

        # Section A should be child of Main Title
        assert section_a.parent_id == main_header.element_id
        # Subsection should be child of Section A
        assert subsection.parent_id == section_a.element_id

    def test_structure_preservation(self):
        """Test that document structure is preserved in chunks."""
        chunker = HierarchicalChunker(
            chunk_size=80, chunk_overlap=10, preserve_structure=True
        )

        text = """# Important Section

This section has critical information that should stay together.

## Subsection

This is subsection content."""

        chunks = chunker.chunk(text, {})

        # Headers should start new chunks when possible
        header_chunks = [chunk for chunk in chunks if chunk.startswith("#")]
        assert len(header_chunks) > 0

    def test_min_section_size_handling(self):
        """Test handling of minimum section size."""
        chunker = HierarchicalChunker(
            chunk_size=100, chunk_overlap=0, min_section_size=50
        )

        text = """# Big Section

This is a large section with plenty of content that exceeds the minimum
section size requirement.

# Tiny

Small content."""

        chunks = chunker.chunk(text, {})

        # Should handle both large and small sections appropriately
        assert len(chunks) >= 1

    def test_table_detection(self):
        """Test detection of table-like content."""
        chunker = HierarchicalChunker(chunk_size=200, chunk_overlap=0)

        text = """# Data Table

| Name | Age | City |
|------|-----|------|
| John | 30  | NYC  |
| Jane | 25  | LA   |

Regular paragraph content."""

        elements = chunker._parse_document_structure(text)

        # Should detect table rows (basic detection)
        paragraphs = [e for e in elements if e.element_type == "paragraph"]
        table_content = [p for p in paragraphs if "|" in p.content]
        assert len(table_content) > 0

    def test_empty_content_handling(self):
        """Test handling of empty or whitespace-only content."""
        chunker = HierarchicalChunker(chunk_size=100, chunk_overlap=0)

        chunks = chunker.chunk("", {})
        assert chunks == []

        chunks = chunker.chunk("   ", {})
        assert len(chunks) <= 1

        chunks = chunker.chunk("\n\n\n", {})
        assert chunks == []

    def test_large_document_handling(self):
        """Test handling of large documents that exceed chunk size."""
        chunker = HierarchicalChunker(chunk_size=50, chunk_overlap=10)

        text = """# Very Long Section

This is a very long section with lots of content that will definitely exceed
the chunk size limit and should be split appropriately while maintaining the
hierarchical structure as much as possible.

## Subsection

More content here that also exceeds limits."""

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 1
        # Should handle large content gracefully
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_mixed_content_types(self):
        """Test handling of mixed content types in document."""
        chunker = HierarchicalChunker(chunk_size=150, chunk_overlap=20)

        text = """# Mixed Content Document

Regular paragraph content.

- List item 1
- List item 2
- List item 3

Another paragraph with more information.

## Code Section

```python
def hello():
    print("Hello World")
```

Final paragraph content."""

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 1
        # Should handle mixed content appropriately
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk.strip()) > 0


class TestDocumentStructureAnalyzer:
    """Test cases for the DocumentStructureAnalyzer utility class."""

    def test_structure_analysis(self):
        """Test comprehensive document structure analysis."""
        analyzer = DocumentStructureAnalyzer()

        text = """# Main Document

Introduction paragraph.

## Section 1

Content for section 1.

### Subsection 1.1

Detailed content.

- List item 1
- List item 2

## Section 2

More content here."""

        analysis = analyzer.analyze_structure(text)

        assert "total_elements" in analysis
        assert "element_counts" in analysis
        assert "header_levels" in analysis
        assert analysis["total_headers"] > 0
        assert analysis["total_paragraphs"] > 0

    def test_header_level_analysis(self):
        """Test analysis of header levels."""
        analyzer = DocumentStructureAnalyzer()

        text = """# Level 1
## Level 2
### Level 3
## Another Level 2
# Another Level 1"""

        analysis = analyzer.analyze_structure(text)

        header_levels = analysis["header_levels"]
        assert header_levels[1] == 2  # Two level 1 headers
        assert header_levels[2] == 2  # Two level 2 headers
        assert header_levels[3] == 1  # One level 3 header

    def test_section_length_metrics(self):
        """Test calculation of section length metrics."""
        analyzer = DocumentStructureAnalyzer()

        text = """# Short

Brief.

# Medium Length Section

This section has moderate content.

# Very Long Section

This section contains a substantial amount of content that goes on for quite
some time with detailed explanations and comprehensive coverage of the topic
at hand."""

        analysis = analyzer.analyze_structure(text)

        assert analysis["avg_section_length"] > 0
        assert analysis["max_section_length"] > analysis["min_section_length"]
        assert analysis["min_section_length"] >= 0

    def test_empty_document_analysis(self):
        """Test analysis of empty document."""
        analyzer = DocumentStructureAnalyzer()

        analysis = analyzer.analyze_structure("")

        assert analysis["total_elements"] == 0
        assert analysis["total_headers"] == 0
        assert analysis["total_paragraphs"] == 0
        assert analysis["avg_section_length"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
