import os
from typing import Any

import weave
from pydantic import PrivateAttr

from nextbench.clients import BaseLLMClient

try:
    import openai
except ImportError:
    raise ImportError(
        "OpenAI client requires the openai package to be installed. Do `uv add openai` to install it."
    )


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
