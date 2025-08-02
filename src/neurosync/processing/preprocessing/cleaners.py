"""
Text cleaning and normalization preprocessors.
"""

import re

from bs4 import BeautifulSoup

from neurosync.processing.base import BasePreprocessor


class HTMLCleaner(BasePreprocessor):
    """Strips HTML tags and artifacts from content."""

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
    """Normalizes whitespace, removing extra spaces and newlines."""

    def process(self, content: str) -> str:
        # Replace multiple newlines with a single one
        content = re.sub(r"\n\s*\n", "\n", content)
        # Replace multiple spaces with a single one
        content = re.sub(r" +", " ", content)
        return content.strip()


# You can add more cleaners here, e.g., for removing PII, etc.
