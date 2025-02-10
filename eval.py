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

dataset = scenario.get_dataset_rows()
# Check for column names for deciding the requirements for scenario
keys = list(dataset.rows[0].keys())
assert "question" in keys, "Question column is required"
assert "answer" in keys, "Answer column is required"

evaluation = weave.Evaluation(
    dataset=dataset[:10],
    scorers=[scenario],
    preprocess_model_input=scenario.preprocess_input,
)

# Setup the client
client = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.0,
    max_completion_tokens=4096,
    system_prompt=scenario.system_prompt,
)

# Run the evaluation
asyncio.run(evaluation.evaluate(client))
