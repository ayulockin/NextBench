import pytest
from unittest.mock import MagicMock, AsyncMock
import asyncio

from nextbench.clients.base_client import BaseLLMClient
from nextbench.utils import RequestResult


class MockLLMClient(BaseLLMClient):
    """
    A concrete subclass of BaseLLMClient used for testing.
    """
    async def _run_llm_call(self, prompt: str) -> str:
        """
        A mock implementation to simulate LLM call.
        By default, returns a fixed string unless overridden.
        """
        return "Mock LLM response"


@pytest.mark.asyncio
async def test_generate_cache_key():
    """
    Test that the cache key is generated based on the expected fields.
    """
    client = MockLLMClient(client_name="test_client")
    prompt = "Test prompt"

    # Points of uniqueness in key:
    #   model, temperature, max_completion_tokens, system_prompt, and prompt
    key = client._generate_cache_key(prompt)
    # Should be a string of hex characters (SHA-256)
    assert len(key) == 64, "Cache key should be a valid SHA-256 hex digest"


@pytest.mark.asyncio
async def test_predict_when_cached():
    """
    Test that if a prompt is present in cache, we return the cached result.
    """
    client = MockLLMClient(client_name="test_client")
    prompt = "Test prompt for caching"

    # Mock the cache
    mock_cache = MagicMock()
    mock_cache.has_key.return_value = True
    mock_cache.get.return_value = "Cached response"
    client._cache = mock_cache

    result: RequestResult = await client.predict(prompt)
    assert result.success is True
    assert result.cached is True
    assert result.completions == ["Cached response"]
    mock_cache.has_key.assert_called_once_with(client._generate_cache_key(prompt))
    mock_cache.get.assert_called_once()


@pytest.mark.asyncio
async def test_predict_when_not_cached_success():
    """
    Test that if a prompt is not cached, client calls _run_llm_call and then caches the result.
    """
    client = MockLLMClient(client_name="test_client")
    prompt = "Test prompt for no-cache success"

    # Mock the cache
    mock_cache = MagicMock()
    mock_cache.has_key.return_value = False
    client._cache = mock_cache

    # The default _run_llm_call in our mock returns "Mock LLM response"
    result: RequestResult = await client.predict(prompt)
    assert result.success is True
    assert result.cached is False
    assert result.completions == ["Mock LLM response"]
    mock_cache.set.assert_called_once_with(client._generate_cache_key(prompt), "Mock LLM response")
