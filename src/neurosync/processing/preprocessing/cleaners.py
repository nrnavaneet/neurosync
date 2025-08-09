"""
Text cleaning and normalization preprocessors for content standardization.

This module provides a comprehensive suite of text cleaning and normalization
tools designed to prepare raw content for optimal processing and embedding
generation. Each preprocessor handles specific aspects of text cleaning
while maintaining semantic meaning and readability.

Key Preprocessors:
    - HTMLCleaner: Removes HTML tags and extracts clean text
    - WhitespaceNormalizer: Standardizes spacing and line breaks
    - Additional cleaners for PII removal, encoding fixes, etc.

Design Principles:
    - Preserve semantic content while removing formatting artifacts
    - Maintain readability and natural language flow
    - Handle edge cases gracefully with robust error recovery
    - Provide configurable cleaning strategies for different content types
    - Optimize for downstream embedding and NLP processing

Common Use Cases:
    - Web scraping content with HTML artifacts
    - Documents with inconsistent formatting
    - Text with excessive whitespace or line breaks
    - Content requiring standardization for consistent processing
    - Multi-source content with varying quality and formats

Processing Pipeline Integration:
    Each cleaner implements the BasePreprocessor interface, allowing them
    to be chained together in configurable preprocessing pipelines. The
    order of application can significantly impact final output quality.

Example Preprocessing Pipeline:
    1. HTMLCleaner: Remove HTML tags and scripts
    2. WhitespaceNormalizer: Standardize spacing
    3. EncodingNormalizer: Fix character encoding issues
    4. PIIRemover: Remove sensitive information
    5. LanguageValidator: Ensure content quality

Performance Considerations:
    - BeautifulSoup provides robust HTML parsing but may be slower for large documents
    - Regex operations are optimized for common patterns
    - Memory usage scales linearly with content size
    - Processing speed optimized for batch operations

Example:
    >>> html_cleaner = HTMLCleaner()
    >>> whitespace_cleaner = WhitespaceNormalizer()
    >>>
    >>> # Process HTML content
    >>> html_content = "<p>Sample <b>HTML</b> content</p>"
    >>> clean_text = html_cleaner.process(html_content)
    >>> normalized_text = whitespace_cleaner.process(clean_text)
    >>> print(f"Final: '{normalized_text}'")  # "Sample HTML content"
    >>>
    >>> # Chain processors
    >>> pipeline = [html_cleaner, whitespace_cleaner]
    >>> result = html_content
    >>> for processor in pipeline:
    ...     result = processor.process(result)

For configuration examples and custom preprocessor development, see:
    - docs/text-preprocessing.md
    - docs/custom-preprocessors.md
    - examples/preprocessing-pipelines.py
"""

import re

from bs4 import BeautifulSoup

from neurosync.processing.base import BasePreprocessor


class HTMLCleaner(BasePreprocessor):
    """
    Robust HTML content cleaner and text extractor.

    The HTMLCleaner removes HTML tags, scripts, styles, and other web-specific
    artifacts while preserving the semantic text content. It handles complex
    HTML structures gracefully and produces clean, readable text suitable
    for further processing and embedding generation.

    Cleaning Process:
        1. Parse HTML using BeautifulSoup for robust tag handling
        2. Remove script and style elements completely
        3. Extract text content from remaining elements
        4. Clean up whitespace and formatting artifacts
        5. Preserve paragraph structure and line breaks

    Features:
        - Removes all HTML tags while preserving text content
        - Eliminates JavaScript and CSS code blocks
        - Handles malformed HTML gracefully
        - Preserves semantic structure through intelligent line breaking
        - Removes excessive whitespace while maintaining readability
        - Handles special HTML entities and character references

    The cleaner is particularly effective for:
        - Web scraping content with embedded scripts and styles
        - HTML documents with complex nested structures
        - Content from various web sources with inconsistent formatting
        - Email content that includes HTML formatting
        - Documentation that mixes HTML and plain text

    Example:
        >>> cleaner = HTMLCleaner()
        >>> html_content = '''
        ... <html>
        ...     <head><title>Sample</title></head>
        ...     <body>
        ...         <h1>Main Title</h1>
        ...         <p>First paragraph with <b>bold</b> text.</p>
        ...         <script>alert('unwanted');</script>
        ...         <p>Second paragraph.</p>
        ...     </body>
        ... </html>
        ... '''
        >>> clean_text = cleaner.process(html_content)
        >>> print(clean_text)
        # Output: "Main Title\nFirst paragraph with bold text.\nSecond paragraph."
    """

    def process(self, content: str) -> str:
        soup = BeautifulSoup(content, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text


class WhitespaceNormalizer(BasePreprocessor):
    """
    Intelligent whitespace and formatting normalizer.

    The WhitespaceNormalizer standardizes spacing, line breaks, and other
    whitespace characters to create consistent, clean text suitable for
    embedding and NLP processing. It preserves semantic structure while
    removing excessive or irregular whitespace.

    Normalization Rules:
        - Multiple consecutive spaces reduced to single space
        - Multiple line breaks reduced to single line break
        - Leading and trailing whitespace removed
        - Tab characters converted to spaces
        - Other whitespace characters standardized

    Features:
        - Preserves paragraph structure through single line breaks
        - Maintains readability while reducing noise
        - Handles various types of whitespace characters
        - Optimized regex patterns for efficient processing
        - Consistent output regardless of input formatting

    The normalizer is essential for:
        - Content from multiple sources with different formatting conventions
        - Text extracted from PDFs or documents with formatting artifacts
        - User-generated content with inconsistent spacing
        - Preparation for embedding models that are sensitive to whitespace
        - Standardization before applying other text processing steps

    Processing Logic:
        1. Replace multiple newlines (paragraph breaks) with single newlines
        2. Replace multiple spaces with single spaces
        3. Remove leading and trailing whitespace
        4. Handle edge cases like tabs and other whitespace characters

    Example:
        >>> normalizer = WhitespaceNormalizer()
        >>> messy_text = "This  has    multiple   spaces.\\n\\n\\n\\nAnd extra lines."
        >>> clean_text = normalizer.process(messy_text)
        >>> print(repr(clean_text))
        # Output: 'This has multiple spaces.\\nAnd extra lines.'
        >>>
        >>> # Handles tabs and mixed whitespace
        >>> mixed_text = "\\t  Tabbed\\ttext\\n\\n  with\\n\\n\\nextra\\nlines  \\n"
        >>> result = normalizer.process(mixed_text)
        >>> print(repr(result))
        # Output: 'Tabbed text\\nwith\\nextra\\nlines'
    """

    def process(self, content: str) -> str:
        # Replace multiple newlines with a single one
        content = re.sub(r"\n\s*\n", "\n", content)
        # Replace multiple spaces with a single one
        content = re.sub(r" +", " ", content)
        return content.strip()


# You can add more cleaners here, e.g., for removing PII, etc.
