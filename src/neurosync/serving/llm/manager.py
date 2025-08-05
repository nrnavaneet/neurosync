"""
Manager to handle different LLM providers.
"""
import asyncio
import os
from typing import Any, AsyncGenerator, Dict, Optional

import requests

from neurosync.core.exceptions.custom_exceptions import ConfigurationError
from neurosync.serving.llm.base import BaseLLM
from neurosync.serving.llm.huggingface import HuggingFaceLocalLLM


class LLMManager:
    """Selects and uses the configured LLM provider."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get("provider", "openrouter")
        self.api_key = config.get("api_key", "")
        self.model_name = config.get("model_name", "anthropic/claude-3-haiku")

        # Load API key from environment if not provided
        if not self.api_key:
            if self.provider == "openrouter":
                self.api_key = os.getenv("OPENROUTER_API_KEY", "")
            elif self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY", "")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY", "")

        if not self.api_key and self.provider not in ["huggingface_local"]:
            raise ConfigurationError(f"No API key found for provider: {self.provider}")

        self.model: Optional[BaseLLM] = self._initialize_model()

    def _initialize_model(self) -> Optional[BaseLLM]:
        """Initializes the LLM based on config."""
        if self.provider == "huggingface_local":
            model_name = self.config.get("model_name", "gpt2")
            return HuggingFaceLocalLLM(model_name=model_name)
        else:
            # For API-based models, we'll handle them directly in this manager
            return None

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """A convenience method to access the model's generate_stream function."""
        if self.model:
            async for token in self.model.generate_stream(prompt, **kwargs):
                yield token
        else:
            # Handle API-based providers - return response as single chunk
            try:
                response = self._call_api(prompt, **kwargs)
                yield response
            except Exception as e:
                yield f"Error: {str(e)}"

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a single response (synchronous version)."""
        try:
            if self.model:
                # For local models, use async method
                async def _generate():
                    response_parts = []
                    if self.model:  # Additional check for mypy
                        async for token in self.model.generate_stream(prompt, **kwargs):
                            response_parts.append(token)
                    return "".join(response_parts)

                # Run the async function
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                return loop.run_until_complete(_generate())
            else:
                # Handle API-based providers directly (synchronous)
                return self._call_api(prompt, **kwargs)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _call_api(self, prompt: str, **kwargs) -> str:
        """Call external API providers."""
        try:
            if self.provider == "openrouter":
                return self._call_openrouter(prompt, **kwargs)
            elif self.provider == "openai":
                return self._call_openai(prompt, **kwargs)
            elif self.provider == "anthropic":
                return self._call_anthropic(prompt, **kwargs)
            else:
                return f"Unsupported provider: {self.provider}"
        except Exception as e:
            return f"API call failed: {str(e)}"

    def _call_openrouter(self, prompt: str, **kwargs) -> str:
        """Call OpenRouter API."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenRouter API error: {str(e)}"

    def _call_openai(self, prompt: str, **kwargs) -> str:
        """Call OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model_name or "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenAI API error: {str(e)}"

    def _call_anthropic(self, prompt: str, **kwargs) -> str:
        """Call Anthropic API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model_name or "claude-3-sonnet-20240229",
            "max_tokens": kwargs.get("max_tokens", 1000),
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result["content"][0]["text"]
        except Exception as e:
            return f"Anthropic API error: {str(e)}"

    async def generate(self, prompt: str, **kwargs) -> str:
        """Async version for server compatibility."""
        return self.generate_response(prompt, **kwargs)

    @property
    def current_model(self) -> Optional[str]:
        """Get the current model name."""
        return self.provider if self.provider else None

    def get_available_models(self) -> Dict[str, Any]:
        """Get available models for the current provider."""
        if self.provider == "huggingface":
            return {"models": ["sentence-transformers/all-MiniLM-L6-v2"]}
        elif self.provider == "openai":
            return {"models": ["gpt-3.5-turbo", "gpt-4"]}
        elif self.provider == "anthropic":
            return {"models": ["claude-3-sonnet-20240229"]}
        else:
            return {"models": []}
