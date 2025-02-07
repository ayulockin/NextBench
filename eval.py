import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import weave
import asyncio

from nextbench.benchmarks import get_dataset, DATASETS
from nextbench.utils import RequestResult
from nextbench.clients import OpenAIClient
from nextbench.utils import parse_math_answer, preprocess_example

# Initialize the weave client
weave_client = weave.init('nextbench-dev')


# Metrics

class ExactMatch(weave.Scorer):
    @weave.op()
    def score(self, answer: str, output: RequestResult) -> bool:
        parsed_prediction = parse_math_answer(output.completions[0])
        return answer == parsed_prediction

# Setup evaluation scenarios
evaluation_scenarios = []

for scenario, ref in DATASETS.items():
    datset = get_dataset(ref)
    evaluation = weave.Evaluation(
        dataset=datset.rows[:10],
        scorers=[ExactMatch()],
        preprocess_model_input=preprocess_example,
    )
    evaluation_scenarios.append(evaluation)


# Setup the client
client = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.0,
    max_completion_tokens=4096,
    system_prompt="You are given maths problems and you need to solve them. Return the answer in the \\boxed{}"
)

# Run the evaluation
for scenario in evaluation_scenarios:
    asyncio.run(scenario.evaluate(client))
