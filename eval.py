import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import weave
import asyncio

from nextbench.clients import OpenAIClient
from nextbench.scenarios import Math500Scenario

# Initialize the weave client
weave_client = weave.init('nextbench-dev')

# Metrics
class ExactMatch(weave.Scorer):
    @weave.op()
    def score(self, answer: str, output: str) -> bool:
        return answer == output


scenario = Math500Scenario(metric=ExactMatch())

evaluation = weave.Evaluation(
    dataset=scenario.get_dataset_rows()[:2],
    scorers=[scenario],
    preprocess_model_input=scenario.preprocess_input,
)

client = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.0,
    max_completion_tokens=4096,
    system_prompt=scenario.system_prompt, # TODO: don't like this implementation detail
)

asyncio.run(evaluation.evaluate(client))
