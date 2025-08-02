"""
Semantic chunker using spaCy for sentence-based splitting.
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
