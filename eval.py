import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import weave
import asyncio

from nextbench.clients import OpenAIClient
from nextbench.scenarios import Math500Scenario, MMLUProScenario

# Initialize the weave client
weave_client = weave.init('nextbench-dev')

# Metrics
class ExactMatch(weave.Scorer):
    @weave.op()
    def score(self, answer: str, output: str) -> bool:
        return answer == output


scenario1 = Math500Scenario(metric=ExactMatch())
dataset1 = scenario1.get_dataset_rows()
evaluation1 = weave.Evaluation(
    dataset=dataset1[:2],
    scorers=[scenario1],
    preprocess_model_input=scenario1.preprocess_input,
)
client1 = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.0,
    max_completion_tokens=4096,
    system_prompt=scenario1.system_prompt,
    enable_cache=False,
)

scenario2 = MMLUProScenario(metric=ExactMatch())
dataset2 = scenario2.get_dataset_rows()
evaluation2 = weave.Evaluation(
    dataset=dataset2[:2],
    scorers=[scenario2],
    preprocess_model_input=scenario2.preprocess_input,
)
client2 = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.0,
    max_completion_tokens=4096,
    system_prompt=scenario2.system_prompt,
    enable_cache=True,
)

asyncio.run(evaluation1.evaluate(client1))
asyncio.run(evaluation2.evaluate(client2))
