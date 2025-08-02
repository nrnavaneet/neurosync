"""
Recursive character-based chunker using LangChain's implementation.
"""
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from neurosync.processing.base import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    A chunker that recursively splits text by a list of separators.
    This is effective for maintaining semantic coherence.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        super().__init__(chunk_size, chunk_overlap)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Chunks the text using the recursive strategy."""
        return self.splitter.split_text(content)
