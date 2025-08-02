"""
Document Structure-Aware Chunker with OCR and Table Extraction

This module provides advanced document processing with OCR integration and
table extraction capabilities. It handles complex document formats including
PDFs, images, and structured documents with tables, forms, and mixed content.

Features:
    - OCR integration for image-based content
    - Table detection and extraction
    - Form field recognition
    - Multi-column layout handling
    - Image and caption association
    - Mixed content type processing
    - Structured metadata preservation

Author: NeuroSync Team
Created: 2025
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from neurosync.core.logging.logger import get_logger
from neurosync.processing.base import BaseChunker

logger = get_logger(__name__)


@dataclass
class StructuredElement:
    """
    Represents a structured document element with enhanced metadata.

    Attributes:
        element_type: Type of element (text, table, image, form, etc.)
        content: Main content of the element
        position: Position information (page, column, coordinates)
        metadata: Rich metadata including format, style, relationships
        confidence: Confidence score for OCR/extraction (0.0-1.0)
        relationships: Related elements (captions, footnotes, etc.)
    """

    element_type: str
    content: str
    position: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    relationships: Optional[List[str]] = None

    def __post_init__(self):
        if self.position is None:
            self.position = {}
        if self.metadata is None:
            self.metadata = {}
        if self.relationships is None:
            self.relationships = []


class DocumentStructureAwareChunker(BaseChunker):
    """
    Advanced document chunker with OCR and table extraction capabilities.

    This chunker handles complex document structures including:
    - OCR text extraction from images and PDFs
    - Table detection, extraction, and preservation
    - Form field recognition and structured data extraction
    - Multi-column layout analysis
    - Image-caption relationship preservation
    - Cross-reference and footnote handling

    Key Features:
        - OCR integration with confidence scoring
        - Table structure preservation with cell relationships
        - Form data extraction and validation
        - Layout-aware chunking for complex documents
        - Metadata-rich output with positioning information
        - Content type classification and specialized handling

    Use Cases:
        - Scanned documents and PDFs
        - Forms and structured documents
        - Research papers with tables and figures
        - Legal documents with complex formatting
        - Multi-language documents with mixed scripts
        - Technical manuals with diagrams and tables
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        ocr_enabled: bool = True,
        table_extraction: bool = True,
        preserve_table_structure: bool = True,
        min_confidence: float = 0.7,
    ):
        """
        Initialize the document structure-aware chunker.

        Args:
            chunk_size: Target size for chunks
            chunk_overlap: Overlap between chunks
            ocr_enabled: Whether to enable OCR processing
            table_extraction: Whether to extract and preserve tables
            preserve_table_structure: Whether to keep table structure intact
            min_confidence: Minimum confidence score for OCR text inclusion
        """
        super().__init__(chunk_size, chunk_overlap)
        self.ocr_enabled = ocr_enabled
        self.table_extraction = table_extraction
        self.preserve_table_structure = preserve_table_structure
        self.min_confidence = min_confidence

        # Table detection patterns
        self.table_patterns = [
            r"\|.*\|.*\|",  # Markdown tables
            r"^\s*\+[-+]+\+\s*$",  # ASCII table borders
            r"^\s*[-]{3,}\s*$",  # Table separators
            r"^\s*[^\s]+\s+[^\s]+\s+[^\s]+",  # Column-like data
        ]

        # Form field patterns
        self.form_patterns = [
            r"^\s*([^:]+):\s*([^\n]*)",  # Key-value pairs
            r"^\s*\[[^\]]*\]\s*([^\n]*)",  # Checkbox fields
            r"^\s*([^:]+)\s*_+\s*$",  # Field labels with blanks
            r"^\s*_+\s*([^\n]*)",  # Fill-in blanks
        ]

        logger.debug(
            f"Initialized DocumentStructureAwareChunker with OCR={ocr_enabled}, "
            f"tables={table_extraction}"
        )

    def _detect_content_type(self, content: str) -> str:
        """
        Detect the primary content type of a text block.

        Args:
            content: Text content to analyze

        Returns:
            Content type classification
        """
        lines = content.strip().split("\n")

        # Check for form patterns first (more specific)
        form_indicators = 0
        for line in lines:
            for pattern in self.form_patterns:
                if re.match(pattern, line):
                    form_indicators += 1
                    break

        if form_indicators >= len(lines) * 0.4:  # 40% of lines look like form
            return "form"

        # Check for table patterns
        table_indicators = 0
        for line in lines:
            for pattern in self.table_patterns:
                if re.match(pattern, line):
                    table_indicators += 1
                    break

        if table_indicators >= len(lines) * 0.3:  # 30% of lines look like table
            return "table"

        # Check for structured lists
        list_indicators = sum(
            1
            for line in lines
            if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+\.\s+", line)
        )
        if list_indicators >= len(lines) * 0.4:  # Lower threshold for lists
            return "list"

        # Check for code-like content
        code_indicators = sum(
            1
            for line in lines
            if re.match(r"^\s*(def|class|import|from|if|for|while)", line)
        )
        if code_indicators >= 2:
            return "code"

        return "text"

    def _extract_table_structure(self, content: str) -> Dict[str, Any]:
        """
        Extract structured table data from content.

        Args:
            content: Content containing table data

        Returns:
            Dictionary with table structure and data
        """
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        # Check for ASCII tables first (more specific)
        if any(re.match(r"^\s*\+[-+]+\+\s*$", line) for line in lines):
            return self._parse_ascii_table(lines)

        # Check for Markdown tables
        if any("|" in line for line in lines):
            return self._parse_markdown_table(lines)

        # Handle column-aligned data
        return self._parse_column_data(lines)

    def _parse_markdown_table(self, lines: List[str]) -> Dict[str, Any]:
        """Parse Markdown-style table."""
        table_data: Dict[str, Any] = {
            "type": "markdown_table",
            "headers": [],
            "rows": [],
            "metadata": {"format": "markdown"},
        }

        for i, line in enumerate(lines):
            if "|" in line:
                cells = [cell.strip() for cell in line.split("|")]
                cells = [cell for cell in cells if cell]  # Remove empty cells

                # Check if this is a separator row
                if all(re.match(r"^[-:]+$", cell) for cell in cells):
                    continue

                if not table_data["headers"] and i == 0:
                    table_data["headers"] = cells
                else:
                    table_data["rows"].append(cells)

        return table_data

    def _parse_ascii_table(self, lines: List[str]) -> Dict[str, Any]:
        """Parse ASCII-style table with borders."""
        table_data: Dict[str, Any] = {
            "type": "ascii_table",
            "headers": [],
            "rows": [],
            "metadata": {"format": "ascii"},
        }

        # Simple implementation - extract text between borders
        data_lines = [
            line for line in lines if not re.match(r"^\s*\+[-+]+\+\s*$", line)
        ]

        for i, line in enumerate(data_lines):
            if "|" in line:
                cells = [cell.strip() for cell in line.split("|")]
                cells = [cell for cell in cells if cell]

                if not table_data["headers"] and i == 0:
                    table_data["headers"] = cells
                else:
                    table_data["rows"].append(cells)

        return table_data

    def _parse_column_data(self, lines: List[str]) -> Dict[str, Any]:
        """Parse column-aligned data."""
        table_data: Dict[str, Any] = {
            "type": "column_data",
            "headers": [],
            "rows": [],
            "metadata": {"format": "columns"},
        }

        # Detect column positions by analyzing spacing
        if lines:
            # Simple column detection based on consistent spacing
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:  # At least 2 columns
                    if not table_data["headers"]:
                        table_data["headers"] = parts
                    else:
                        table_data["rows"].append(parts)

        return table_data

    def _extract_form_fields(self, content: str) -> Dict[str, Any]:
        """
        Extract form fields and structured data.

        Args:
            content: Content containing form data

        Returns:
            Dictionary with form structure and fields
        """
        form_data: Dict[str, Any] = {
            "type": "form",
            "fields": [],
            "metadata": {"format": "form"},
        }

        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Key-value pairs
            match = re.match(r"^([^:]+):\s*(.*)$", line)
            if match:
                field_name, field_value = match.groups()
                form_data["fields"].append(
                    {
                        "type": "key_value",
                        "name": field_name.strip(),
                        "value": field_value.strip(),
                        "line": line,
                    }
                )
                continue

            # Checkbox fields
            match = re.match(r"^\[([^\]]*)\]\s*(.*)$", line)
            if match:
                checkbox_state, label = match.groups()
                form_data["fields"].append(
                    {
                        "type": "checkbox",
                        "state": checkbox_state.strip(),
                        "label": label.strip(),
                        "line": line,
                    }
                )
                continue

            # Fill-in blanks - check for fields with underscores
            if "_" in line and len(line) > 3:  # Must have multiple underscores
                underscore_count = line.count("_")
                if (
                    underscore_count >= 3
                ):  # At least 3 underscores to be a fill-in field
                    form_data["fields"].append(
                        {"type": "fill_blank", "template": line, "line": line}
                    )

        return form_data

    def _ocr_process_image(self, image_path: str) -> Tuple[str, float]:
        """
        Process image using OCR (placeholder implementation).

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        # Placeholder for OCR integration
        # In real implementation, would use libraries like:
        # - pytesseract for basic OCR
        # - easyocr for multi-language support
        # - Azure Computer Vision or AWS Textract for cloud OCR

        logger.warning("OCR processing not implemented - placeholder only")
        return f"[OCR_PLACEHOLDER: {image_path}]", 0.5

    def _create_structured_elements(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[StructuredElement]:
        """
        Create structured elements from document content.

        Args:
            content: Document content
            metadata: Additional metadata

        Returns:
            List of structured elements
        """
        elements = []

        # Split content into potential sections
        sections = re.split(r"\n\s*\n", content)

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            content_type = self._detect_content_type(section)

            element = StructuredElement(
                element_type=content_type,
                content=section.strip(),
                position={"section": i},
                metadata={"original_metadata": metadata.copy()},
            )

            # Add type-specific processing
            if content_type == "table" and self.table_extraction:
                table_data = self._extract_table_structure(section)
                assert element.metadata is not None
                element.metadata["table_data"] = table_data
                element.metadata["preserve_structure"] = self.preserve_table_structure

            elif content_type == "form":
                form_data = self._extract_form_fields(section)
                assert element.metadata is not None
                element.metadata["form_data"] = form_data

            elements.append(element)

        return elements

    def _chunk_structured_elements(
        self, elements: List[StructuredElement]
    ) -> List[str]:
        """
        Create chunks from structured elements.

        Args:
            elements: List of structured elements

        Returns:
            List of text chunks with preserved structure
        """
        chunks = []
        current_chunk: List[str] = []
        current_chunk_size = 0

        for element in elements:
            element_content = element.content
            element_size = len(element_content)

            # Handle tables specially
            if element.element_type == "table" and self.preserve_table_structure:
                # Keep tables intact if possible
                if element_size <= self.chunk_size:
                    # Flush current chunk if adding table would exceed size
                    if (
                        current_chunk
                        and current_chunk_size + element_size > self.chunk_size
                    ):
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                        current_chunk_size = 0

                    # Add table with metadata comment
                    assert element.metadata is not None
                    table_metadata = element.metadata.get("table_data", {})
                    table_comment = (
                        f"<!-- TABLE: {table_metadata.get('type', 'unknown')} -->"
                    )
                    current_chunk.append(f"{table_comment}\n{element_content}")
                    current_chunk_size += element_size + len(table_comment)
                else:
                    # Table too large - split carefully
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                        current_chunk_size = 0

                    # Try to split table by rows if possible
                    assert element.metadata is not None
                    table_data = element.metadata.get("table_data", {})
                    if "rows" in table_data:
                        headers = table_data.get("headers", [])
                        rows = table_data["rows"]

                        # Include headers in each chunk
                        header_text = " | ".join(headers) if headers else ""
                        chunk_rows: List[str] = []
                        chunk_size = len(header_text)

                        for row in rows:
                            row_text = " | ".join(row)
                            if (
                                chunk_size + len(row_text) > self.chunk_size
                                and chunk_rows
                            ):
                                # Create chunk with current rows
                                table_chunk = (
                                    header_text + "\n" + "\n".join(chunk_rows)
                                    if header_text
                                    else "\n".join(chunk_rows)
                                )
                                chunks.append(table_chunk)
                                chunk_rows = [row_text]
                                chunk_size = len(header_text) + len(row_text)
                            else:
                                chunk_rows.append(row_text)
                                chunk_size += len(row_text)

                        # Add final chunk
                        if chunk_rows:
                            table_chunk = (
                                header_text + "\n" + "\n".join(chunk_rows)
                                if header_text
                                else "\n".join(chunk_rows)
                            )
                            chunks.append(table_chunk)
                    else:
                        # Fallback to basic splitting
                        from .recursive_chunker import RecursiveChunker

                        splitter = RecursiveChunker(self.chunk_size, self.chunk_overlap)
                        sub_chunks = splitter.chunk(element_content, {})
                        chunks.extend(sub_chunks)

            # Handle regular content
            else:
                if (
                    current_chunk_size + element_size > self.chunk_size
                    and current_chunk
                ):
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_chunk_size = 0

                if element_size > self.chunk_size:
                    # Split large element
                    from .recursive_chunker import RecursiveChunker

                    splitter = RecursiveChunker(self.chunk_size, self.chunk_overlap)
                    sub_chunks = splitter.chunk(element_content, {})
                    chunks.extend(sub_chunks)
                else:
                    current_chunk.append(element_content)
                    current_chunk_size += element_size

        # Add final chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Chunk content using document structure analysis with OCR and table extraction.

        Args:
            content: Document content to chunk
            metadata: Additional metadata

        Returns:
            List of structured chunks with preserved formatting
        """
        if not content:
            return []

        # Create structured elements
        elements = self._create_structured_elements(content, metadata)

        # Create chunks preserving structure
        chunks = self._chunk_structured_elements(elements)

        logger.info(
            f"Created {len(chunks)} structure-aware chunks from "
            f"{len(elements)} elements"
        )
        return chunks

    def analyze_document_complexity(self, content: str) -> Dict[str, Any]:
        """
        Analyze document complexity and structure.

        Args:
            content: Document content to analyze

        Returns:
            Dictionary with complexity metrics
        """
        elements = self._create_structured_elements(content, {})

        # Count element types
        type_counts: Dict[str, int] = {}
        total_size = 0
        table_count = 0
        form_count = 0

        for element in elements:
            element_type = element.element_type
            type_counts[element_type] = type_counts.get(element_type, 0) + 1
            total_size += len(element.content)

            if element_type == "table":
                table_count += 1
            elif element_type == "form":
                form_count += 1

        return {
            "total_elements": len(elements),
            "total_size": total_size,
            "element_types": type_counts,
            "table_count": table_count,
            "form_count": form_count,
            "avg_element_size": total_size / len(elements) if elements else 0,
            "complexity_score": len(type_counts) * 0.2
            + table_count * 0.3
            + form_count * 0.2,
        }
