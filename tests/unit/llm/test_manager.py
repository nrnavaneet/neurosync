"""
Tests for LLM Manager functionality.
"""
from unittest.mock import AsyncMock, Mock, patch

import pytest

from neurosync.core.config.settings import Settings
from neurosync.serving.llm.manager import LLMManager


class TestLLMManager:
    """Test cases for LLM Manager."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.OPENAI_API_KEY = "test-openai-key"
        settings.ANTHROPIC_API_KEY = "test-anthropic-key"
        settings.COHERE_API_KEY = "test-cohere-key"
        settings.GOOGLE_API_KEY = "test-google-key"
        settings.OPENROUTER_API_KEY = "sk-or-v1-test-openrouter-key"
        settings.DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
        settings.LLM_ENABLE_FALLBACK = True
        return settings

    @pytest.fixture
    def llm_manager(self, mock_settings):
        """Create LLM manager instance for testing."""
        return LLMManager(mock_settings)

    def test_initialization(self, mock_settings):
        """Test LLM manager initialization."""
        manager = LLMManager(mock_settings)
        assert manager.settings == mock_settings
        assert "openai" in manager.available_providers
        assert "anthropic" in manager.available_providers
        assert "cohere" in manager.available_providers
        assert "google" in manager.available_providers
        assert "openrouter" in manager.available_providers

    def test_detect_available_providers(self, llm_manager):
        """Test provider detection based on API keys."""
        assert len(llm_manager.available_providers) == 5
        assert "openai" in llm_manager.available_providers
        assert "openrouter" in llm_manager.available_providers

    def test_detect_available_providers_no_keys(self):
        """Test provider detection with no API keys."""
        settings = Mock(spec=Settings)
        settings.OPENAI_API_KEY = None
        settings.ANTHROPIC_API_KEY = None
        settings.COHERE_API_KEY = None
        settings.GOOGLE_API_KEY = None
        settings.OPENROUTER_API_KEY = None

        manager = LLMManager(settings)
        assert len(manager.available_providers) == 0

    @patch("neurosync.serving.llm.openrouter.OpenRouterLLM")
    def test_create_model_openrouter_priority(self, mock_openrouter, mock_settings):
        """Test that OpenRouter is prioritized when available."""
        manager = LLMManager(mock_settings)
        _ = manager._create_model()
        mock_openrouter.assert_called_once_with(api_key="sk-or-v1-test-openrouter-key")

    @patch("neurosync.serving.llm.openai.OpenAILLM")
    def test_create_model_fallback_to_openai(self, mock_openai):
        """Test fallback to OpenAI when OpenRouter not available."""
        settings = Mock(spec=Settings)
        settings.OPENAI_API_KEY = "test-openai-key"
        settings.ANTHROPIC_API_KEY = None
        settings.COHERE_API_KEY = None
        settings.GOOGLE_API_KEY = None
        settings.OPENROUTER_API_KEY = None
        settings.DEFAULT_LLM_MODEL = "gpt-3.5-turbo"

        manager = LLMManager(settings)
        _ = manager._create_model()
        mock_openai.assert_called_once_with(api_key="test-openai-key")

    def test_create_model_no_providers(self):
        """Test error when no providers are available."""
        settings = Mock(spec=Settings)
        settings.OPENAI_API_KEY = None
        settings.ANTHROPIC_API_KEY = None
        settings.COHERE_API_KEY = None
        settings.GOOGLE_API_KEY = None
        settings.OPENROUTER_API_KEY = None
        settings.DEFAULT_LLM_MODEL = "gpt-3.5-turbo"

        manager = LLMManager(settings)
        with pytest.raises(ValueError, match="No LLM provider configured"):
            manager._create_model()

    @pytest.mark.asyncio
    @patch("neurosync.serving.llm.openrouter.OpenRouterLLM")
    async def test_generate_success(self, mock_openrouter_class, llm_manager):
        """Test successful text generation."""
        mock_model = AsyncMock()
        mock_model.generate.return_value = "Test response"
        mock_openrouter_class.return_value = mock_model

        response = await llm_manager.generate("Test prompt")
        assert response == "Test response"
        mock_model.generate.assert_called_once()

    @pytest.mark.asyncio
    @patch("neurosync.serving.llm.openrouter.OpenRouterLLM")
    async def test_generate_stream_success(self, mock_openrouter_class, llm_manager):
        """Test successful streaming generation."""
        mock_model = AsyncMock()

        async def mock_stream(prompt, **kwargs):
            yield "Hello"
            yield " "
            yield "World"

        mock_model.generate_stream = mock_stream
        mock_openrouter_class.return_value = mock_model

        response_parts = []
        async for part in llm_manager.generate_stream("Test prompt"):
            response_parts.append(part)

        assert response_parts == ["Hello", " ", "World"]

    def test_get_available_models(self, llm_manager):
        """Test getting available models."""
        models = llm_manager.get_available_models()
        assert isinstance(models, list)
        # Should include at least the models from available providers
        assert len(models) > 0

    def test_get_available_models_no_providers(self):
        """Test getting models when no providers available."""
        settings = Mock(spec=Settings)
        settings.OPENAI_API_KEY = None
        settings.ANTHROPIC_API_KEY = None
        settings.COHERE_API_KEY = None
        settings.GOOGLE_API_KEY = None
        settings.OPENROUTER_API_KEY = None

        manager = LLMManager(settings)
        models = manager.get_available_models()
        assert models == []
