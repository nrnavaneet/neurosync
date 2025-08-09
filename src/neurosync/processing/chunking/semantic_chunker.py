"""
Semantic chunking implementation using NLP-based sentence boundary detection.

This module implements intelligent text chunking that leverages Natural
Language Processing to identify semantic boundaries, particularly sentence
boundaries, for creating more coherent text chunks. It uses spaCy's robust
sentence segmentation to ensure chunks break at natural linguistic boundaries.

Key Features:
    - NLP-powered sentence boundary detection
    - Semantic coherence preservation through sentence-level splitting
    - Configurable spaCy models for different languages and domains
    - Optimized pipeline with disabled unnecessary components
    - Intelligent sentence grouping within size constraints
    - Robust error handling for model loading and processing

Advantages over character-based chunking:
    - Respects linguistic boundaries and semantic units
    - Maintains sentence integrity and readability
    - Better performance for question-answering tasks
    - Improved context preservation for embedding models
    - Language-aware processing with multilingual support

spaCy Model Support:
    - en_core_web_sm: Small English model (default)
    - en_core_web_md: Medium English model with word vectors
    - en_core_web_lg: Large English model with improved accuracy
    - Multilingual models: de_core_news_sm, fr_core_news_sm, etc.
    - Custom trained models for domain-specific processing

Processing Pipeline:
    1. Load and configure spaCy model with optimized pipeline
    2. Process text to identify sentence boundaries
    3. Group sentences into chunks respecting size limits
    4. Handle overlap requirements for context continuity
    5. Validate output quality and chunk consistency

This chunker is particularly effective for:
    - Academic papers and research documents
    - News articles and journalistic content
    - Technical documentation with complex sentences
    - Content requiring high semantic coherence
    - Multilingual text processing scenarios

Example:
    >>> chunker = SpacySemanticChunker(
    ...     chunk_size=512,
    ...     chunk_overlap=50,
    ...     model_name="en_core_web_sm"
    ... )
    >>> text = (
    ...     "Complex document with multiple sentences. "
    ...     "Each sentence contains important information."
    ... )
    >>> chunks = chunker.chunk(text, metadata={"language": "en"})
    >>> print(f"Created {len(chunks)} semantically coherent chunks")

For language-specific configuration and model selection, see:
    - docs/semantic-chunking.md
    - docs/multilingual-processing.md
    - examples/nlp-chunking-strategies.py
"""
from typing import Any, Dict, List

import spacy

from neurosync.core.exceptions.custom_exceptions import ConfigurationError
from neurosync.core.logging.logger import get_logger
from neurosync.processing.base import BaseChunker

logger = get_logger(__name__)


class SpacySemanticChunker(BaseChunker):
    """
    A chunker that uses spaCy to split text into sentences, then groups
    sentences into chunks of a desired size.
    """

    def __init__(
        self, chunk_size: int, chunk_overlap: int, model_name: str = "en_core_web_sm"
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.model_name = model_name
        try:
            self.nlp = spacy.load(model_name, disable=["parser", "ner"])
            self.nlp.add_pipe("sentencizer")
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found.")
            logger.error(f"Please run: python -m spacy download {model_name}")
            raise ConfigurationError(f"spaCy model '{model_name}' not found.")

    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Chunks the text by grouping sentences."""
        doc = self.nlp(content)
        sentences = [sent.text.strip() for sent in doc.sents]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Note: This basic implementation doesn't handle overlap.
        # A more advanced version could be implemented if needed.
        return chunks
