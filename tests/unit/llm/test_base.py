"""
Tests for Base LLM class functionality.
"""
import pytest

from neurosync.serving.llm.base import BaseLLM


class TestBaseLLM:
    """Test cases for Base LLM class."""

    class MockLLM(BaseLLM):
        """Mock implementation of BaseLLM for testing."""

        def __init__(self):
            self.info = {"type": "mock", "model": "test-model"}

        async def generate_stream(self, prompt: str, **kwargs):
            """Mock streaming generation."""
            yield "Hello"
            yield " "
            yield "World"

        def get_info(self):
            """Return mock info."""
            return self.info

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM instance."""
        return self.MockLLM()

    @pytest.mark.asyncio
    async def test_generate_method(self, mock_llm):
        """Test that base generate method collects streaming tokens."""
        result = await mock_llm.generate("Test prompt")
        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_generate_with_empty_stream(self, mock_llm):
        """Test generate method with empty stream."""

        # Override generate_stream to return empty
        async def empty_stream(prompt, **kwargs):
            return
            yield  # This line is never reached

        mock_llm.generate_stream = empty_stream
        result = await mock_llm.generate("Test prompt")
        assert result == ""

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self, mock_llm):
        """Test that kwargs are passed to generate_stream."""
        called_kwargs = {}

        async def capture_kwargs_stream(prompt, **kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            yield "response"

        mock_llm.generate_stream = capture_kwargs_stream

        await mock_llm.generate("Test prompt", temperature=0.8, max_tokens=100)

        assert called_kwargs["temperature"] == 0.8
        assert called_kwargs["max_tokens"] == 100

    def test_get_info_abstract_method(self, mock_llm):
        """Test that get_info is implemented."""
        info = mock_llm.get_info()
        assert info["type"] == "mock"
        assert info["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_generate_stream_abstract_method(self, mock_llm):
        """Test that generate_stream is implemented."""
        tokens = []
        async for token in mock_llm.generate_stream("Test"):
            tokens.append(token)

        assert tokens == ["Hello", " ", "World"]

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that BaseLLM cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLM()

    class IncompleteImplementation(BaseLLM):
        """Incomplete implementation missing required methods."""

        pass

    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations cannot be instantiated."""
        with pytest.raises(TypeError):
            self.IncompleteImplementation()

    class MinimalImplementation(BaseLLM):
        """Minimal valid implementation."""

        async def generate_stream(self, prompt: str, **kwargs):
            yield "test"

        def get_info(self):
            return {"type": "minimal"}

    @pytest.mark.asyncio
    async def test_minimal_implementation_works(self):
        """Test that minimal valid implementation works."""
        llm = self.MinimalImplementation()

        # Test generate_stream
        tokens = []
        async for token in llm.generate_stream("test"):
            tokens.append(token)
        assert tokens == ["test"]

        # Test get_info
        info = llm.get_info()
        assert info["type"] == "minimal"

        # Test base generate method
        result = await llm.generate("test")
        assert result == "test"
