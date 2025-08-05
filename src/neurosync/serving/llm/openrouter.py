"""
OpenRouter LLM implementation.
"""
import json
from typing import Any, AsyncGenerator, Dict

import aiohttp

from neurosync.core.logging.logger import get_logger
from neurosync.serving.llm.base import BaseLLM

logger = get_logger(__name__)


class OpenRouterLLM(BaseLLM):
    """OpenRouter LLM connector with streaming support."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = "https://openrouter.ai/api/v1"
        logger.info(f"Initialized OpenRouter LLM with model: {model_name}")

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response from OpenRouter."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": True,
            }

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=data
                ) as response:
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(data_str)
                                if chunk_data.get("choices") and chunk_data["choices"][
                                    0
                                ]["delta"].get("content"):
                                    yield chunk_data["choices"][0]["delta"]["content"]
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            yield f"Error: {str(e)}"

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "type": "openrouter",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
