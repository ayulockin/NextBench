# base_llm.py

import os
import weave
from abc import ABC, abstractmethod
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

from pydantic import PrivateAttr
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_random_exponential,
    RetryError,
)
from nextbench.utils import RequestResult
from nextbench.utils import DiskCacheBackend


class BaseLLMClient(weave.Model, ABC):
    client_name: str
    model: str = "gpt-4"
    temperature: float = 0.0
    max_completion_tokens: int = 4096
    system_prompt: str = ""

    _cache: DiskCacheBackend = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._cache = DiskCacheBackend()

    @abstractmethod
    async def _run_llm_call(self, prompt: str) -> str:
        """
        Subclasses override this with an actual LLM call (async).
        """
        raise NotImplementedError

    def _generate_cache_key(self, prompt: str) -> str:
        import hashlib
        import json

        key_data = {
            "model": self.model,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "system_prompt": self.system_prompt,
            "prompt": prompt,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode("utf-8")).hexdigest()

    @weave.op()
    async def predict(self, prompt: str) -> RequestResult:
        cache_key = self._generate_cache_key(prompt)
        # Check cache
        if self._cache.has_key(cache_key):
            cached_content = self._cache.get(cache_key)
            return RequestResult(
                success=True,
                cached=True,
                completions=[cached_content],
                error=None,
            )

        # Not cached, attempt LLM call with Tenacity
        content: Optional[str] = None
        error: Optional[str] = None

        retryer = AsyncRetrying(
            wait=wait_random_exponential(multiplier=1, max=60),
            stop=stop_after_attempt(3),
            reraise=True,
        )

        try:
            async for attempt in retryer:
                with attempt:
                    content = await self._run_llm_call(prompt)
        except RetryError:
            error = "LLM call failed after 3 attempts."

        if content is not None and error is None:
            self._cache.set(cache_key, content)
            return RequestResult(
                success=True,
                cached=False,
                completions=[content],
                error=None
            )
        else:
            return RequestResult(
                success=False,
                cached=False,
                completions=[],
                error=error
            )

# openai_client.py

import openai
from typing import Any
from pydantic import PrivateAttr


class OpenAIClient(BaseLLMClient):
    client_name: str = "openai"
    _client: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @weave.op()
    async def _run_llm_call(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "developer", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
        )
        content = response.choices[0].message.content
        return content

if __name__ == "__main__":
    import asyncio

    weave_client = weave.init("nextbench-dev")

    client = OpenAIClient()
    async def main():
        # First call
        result = await client.predict("What is the capital of India?")
        print(result)

        # Second call (same prompt => cached)
        result = await client.predict("What is the capital of India?")
        print(result)

    asyncio.run(main())
