import os
import weave
import openai
from typing import Any
from pydantic import PrivateAttr
from dotenv import load_dotenv
load_dotenv()

from nextbench.utils import RequestResult


class OpenAIClient(weave.Model):
    _client: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @weave.op()
    async def predict(self, prompt: str) -> RequestResult:
        response = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return RequestResult(success=True, cached=False, completions=[response.choices[0].message.content])

if __name__ == "__main__":
    import asyncio

    client = OpenAIClient()
    result = asyncio.run(client.predict("What is the capital of France?"))
    print(result)
