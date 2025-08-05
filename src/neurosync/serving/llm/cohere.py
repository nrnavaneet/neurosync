"""
Cohere LLM implementation.
"""
from typing import Any, AsyncGenerator, Dict

import cohere

from neurosync.core.logging.logger import get_logger
from neurosync.serving.llm.base import BaseLLM

logger = get_logger(__name__)


class CohereLLM(BaseLLM):
    """Cohere LLM connector with streaming support."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "command-r",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
    ):
        self.client = cohere.AsyncClient(api_key=api_key, timeout=timeout)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized Cohere LLM with model: {model_name}")

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response from Cohere."""
        try:
            response = await self.client.chat_stream(
                model=self.model_name,
                message=prompt,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )

            async for chunk in response:
                if chunk.event_type == "text-generation":
                    yield chunk.text

        except Exception as e:
            logger.error(f"Cohere generation failed: {e}")
            yield f"Error: {str(e)}"

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "type": "cohere",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
