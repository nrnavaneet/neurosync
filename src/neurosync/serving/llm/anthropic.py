"""
Anthropic Claude LLM implementation.
"""
from typing import Any, AsyncGenerator, Dict

import anthropic

from neurosync.core.logging.logger import get_logger
from neurosync.serving.llm.base import BaseLLM

logger = get_logger(__name__)


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM connector with streaming support."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
    ):
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized Anthropic LLM with model: {model_name}")

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response from Anthropic."""
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            async for chunk in response:
                if chunk.type == "content_block_delta":
                    if hasattr(chunk.delta, "text"):
                        yield chunk.delta.text

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            yield f"Error: {str(e)}"

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "type": "anthropic",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
