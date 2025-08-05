"""
OpenAI LLM implementation.
"""
from typing import Any, AsyncGenerator, Dict

from openai import AsyncOpenAI

from neurosync.core.logging.logger import get_logger
from neurosync.serving.llm.base import BaseLLM

logger = get_logger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI LLM connector with streaming support."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
    ):
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized OpenAI LLM with model: {model_name}")

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response from OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True,
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            yield f"Error: {str(e)}"

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "type": "openai",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
