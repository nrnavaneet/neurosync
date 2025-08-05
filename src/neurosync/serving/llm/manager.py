"""
Unified LLM provider manager with fallback support.
"""
from typing import AsyncGenerator, List, Optional

from neurosync.core.logging.logger import get_logger
from neurosync.serving.llm.base import BaseLLM

logger = get_logger(__name__)


class LLMManager:
    """Manages multiple LLM providers with intelligent fallback support."""

    def __init__(self, settings):
        self.settings = settings
        self.current_model: Optional[str] = None
        self.available_providers: List[str] = []
        self._detect_available_providers()

    def _detect_available_providers(self) -> None:
        """Detect available LLM providers based on API keys."""
        self.available_providers = []

        if self.settings.OPENAI_API_KEY:
            self.available_providers.append("openai")
        if self.settings.ANTHROPIC_API_KEY:
            self.available_providers.append("anthropic")
        if self.settings.COHERE_API_KEY:
            self.available_providers.append("cohere")
        if getattr(self.settings, "GOOGLE_API_KEY", None):
            self.available_providers.append("google")
        if getattr(self.settings, "OPENROUTER_API_KEY", None):
            self.available_providers.append("openrouter")

        logger.info(f"Available LLM providers: {self.available_providers}")

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        models = []

        if "openai" in self.available_providers:
            models.extend(["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
        if "anthropic" in self.available_providers:
            models.extend(["claude-3-sonnet", "claude-3-haiku", "claude-2"])
        if "cohere" in self.available_providers:
            models.extend(["command", "command-nightly"])
        if "google" in self.available_providers:
            models.extend(["gemini-pro"])
        if "openrouter" in self.available_providers:
            models.extend(["openai/gpt-3.5-turbo", "anthropic/claude-3-sonnet"])

        return models

    def _create_model(self, model: Optional[str] = None) -> BaseLLM:
        """Create an LLM instance based on configuration."""
        if model is None:
            model = getattr(self.settings, "DEFAULT_LLM_MODEL", "gpt-3.5-turbo")

        # Try OpenRouter first if it's available and properly configured
        if getattr(self.settings, "OPENROUTER_API_KEY", None) and "sk-or-v1-" in str(
            self.settings.OPENROUTER_API_KEY
        ):
            from neurosync.serving.llm.openrouter import OpenRouterLLM

            return OpenRouterLLM(api_key=self.settings.OPENROUTER_API_KEY)

        # Try OpenAI if API key is available
        if self.settings.OPENAI_API_KEY and any(
            m in model.lower() for m in ["gpt", "openai"]
        ):
            from neurosync.serving.llm.openai import OpenAILLM

            return OpenAILLM(api_key=self.settings.OPENAI_API_KEY)

        # Try Anthropic
        if self.settings.ANTHROPIC_API_KEY and any(
            m in model.lower() for m in ["claude", "anthropic"]
        ):
            from neurosync.serving.llm.anthropic import AnthropicLLM

            return AnthropicLLM(api_key=self.settings.ANTHROPIC_API_KEY)

        # Try Cohere
        if self.settings.COHERE_API_KEY and any(
            m in model.lower() for m in ["command", "cohere"]
        ):
            from neurosync.serving.llm.cohere import CohereLLM

            return CohereLLM(api_key=self.settings.COHERE_API_KEY)

        # Try Google Gemini
        if getattr(self.settings, "GOOGLE_API_KEY", None) and any(
            m in model.lower() for m in ["gemini", "google"]
        ):
            from neurosync.serving.llm.gemini import GeminiLLM

            return GeminiLLM(api_key=self.settings.GOOGLE_API_KEY)

        # Fallback to any available provider
        if self.settings.OPENAI_API_KEY:
            from neurosync.serving.llm.openai import OpenAILLM

            return OpenAILLM(api_key=self.settings.OPENAI_API_KEY)

        if self.settings.ANTHROPIC_API_KEY:
            from neurosync.serving.llm.anthropic import AnthropicLLM

            return AnthropicLLM(api_key=self.settings.ANTHROPIC_API_KEY)

        if self.settings.COHERE_API_KEY:
            from neurosync.serving.llm.cohere import CohereLLM

            return CohereLLM(api_key=self.settings.COHERE_API_KEY)

        raise ValueError(
            "No LLM provider configured. Please set at least one API key: "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY, or GOOGLE_API_KEY"
        )

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Optional[str]:
        """Generate text using the configured LLM with fallback support."""
        fallback_enabled = getattr(self.settings, "LLM_ENABLE_FALLBACK", True)

        # Try primary model first
        try:
            llm = self._create_model(model)
            self.current_model = model or getattr(
                self.settings, "DEFAULT_LLM_MODEL", "gpt-3.5-turbo"
            )
            response = await llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            if response:
                return response
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")
            if not fallback_enabled:
                raise

        # Try fallback providers if enabled
        if fallback_enabled:
            for provider in self.available_providers:
                try:
                    fallback_model = f"{provider}-default"
                    if fallback_model != (
                        model or getattr(self.settings, "DEFAULT_LLM_MODEL", "")
                    ):
                        llm = self._create_model(fallback_model)
                        self.current_model = fallback_model
                        response = await llm.generate(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        )
                        if response:
                            logger.info(
                                f"Fallback successful with provider: {provider}"
                            )
                            return response
                except Exception as e:
                    logger.warning(f"Fallback provider {provider} failed: {e}")
                    continue

        raise ValueError("All LLM providers failed to generate streaming response")

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using the configured LLM with fallback support."""
        fallback_enabled = getattr(self.settings, "LLM_ENABLE_FALLBACK", True)

        # Try primary model first
        try:
            llm = self._create_model(model)
            self.current_model = model or getattr(
                self.settings, "DEFAULT_LLM_MODEL", "gpt-3.5-turbo"
            )
            async for chunk in llm.generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            ):
                yield chunk
            return
        except Exception as e:
            logger.warning(f"Primary model streaming failed: {e}")
            if not fallback_enabled:
                raise

        # Try fallback providers if enabled
        if fallback_enabled:
            for provider in self.available_providers:
                try:
                    fallback_model = f"{provider}-default"
                    if fallback_model != (
                        model or getattr(self.settings, "DEFAULT_LLM_MODEL", "")
                    ):
                        llm = self._create_model(fallback_model)
                        self.current_model = fallback_model
                        logger.info(f"Streaming fallback with provider: {provider}")
                        async for chunk in llm.generate_stream(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        ):
                            yield chunk
                        return
                except Exception as e:
                    logger.warning(
                        f"Fallback streaming provider {provider} failed: {e}"
                    )
                    continue

        raise ValueError("All LLM providers failed to generate streaming response")
