import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import weave
import asyncio

from nextbench.benchmarks import get_dataset, DATASETS
from nextbench.utils import RequestResult
from nextbench.clients import OpenAIClient

weave_client = weave.init('nextbench-dev')


@weave.op()
def preprocess_example(example):
    return {
        "prompt": example["question"]
    }

@weave.op()
def parse_answer(completion: str) -> str:
    return re.search(r'\\boxed{(.*)}', completion).group(1)

@weave.op()
def exact_match(answer: str, output: RequestResult) -> bool:
    parsed_prediction = parse_answer(output.completions[0])
    return answer == parsed_prediction

evaluation_scenarios = []

for scenario, ref in DATASETS.items():
    datset = get_dataset(ref)
    evaluation = weave.Evaluation(
        dataset=datset.rows[:10],
        scorers=[exact_match],
        preprocess_model_input=preprocess_example,
    )
    evaluation_scenarios.append(evaluation)


client = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.0,
    max_completion_tokens=4096,
    system_prompt="You are given maths problems and you need to solve them. Return the answer in the \\boxed{}"
)

for scenario in evaluation_scenarios:
    asyncio.run(scenario.evaluate(client))
