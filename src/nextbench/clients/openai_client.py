import os
import weave
import openai
from typing import Any
from pydantic import PrivateAttr
from dotenv import load_dotenv
load_dotenv()

from nextbench.utils import RequestResult
from nextbench.utils import cacher, DiskCacheBackend


class OpenAIClient(weave.Model):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_completion_tokens: int = 4096
    system_prompt: str = ""

    _client: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @weave.op()
    async def predict(self, prompt: str) -> RequestResult:
        content, cached = await self._call_llm(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            system_prompt=self.system_prompt,
        )

        return RequestResult(
            success=True,
            cached=cached,
            completions=[content],
        )

    @cacher(cache_backend=None)
    async def _call_llm(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_completion_tokens: int,
        system_prompt: str,
    ) -> tuple[str, bool]:
        raise NotImplementedError("We'll override this below.")


class OpenAIClientWithCache(OpenAIClient):
    @cacher(cache_backend=DiskCacheBackend())
    async def _call_llm(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_completion_tokens: int,
        system_prompt: str,
    ) -> tuple[str, bool]:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        content = response.choices[0].message.content

        return content


if __name__ == "__main__":
    import asyncio

    client = OpenAIClientWithCache()
    # First call (should hit the OpenAI API)
    result = asyncio.run(client.predict("What is the capital of France?"))
    print(result)  # cached=False, presumably.

    # Second call with same arguments (should come from cache this time)
    result = asyncio.run(client.predict("What is the capital of France?"))
    print(result)  # cached=True or the same content as before
