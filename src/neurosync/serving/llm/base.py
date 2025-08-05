"""
Base LLM interface.
"""
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict


class BaseLLM(ABC):
    """Abstract base class for Large Language Model connectors."""

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Generates a stream of text tokens in response to a prompt.
        Yields tokens as they become available.
        """
        # This is a placeholder for the async generator
        if False:
            yield

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Returns information about the loaded model."""
        pass

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates a complete response by collecting all tokens from generate_stream.
        This is a convenience method that most implementations can use.
        """
        tokens = []
        async for token in self.generate_stream(prompt, **kwargs):
            tokens.append(token)
        return "".join(tokens)
