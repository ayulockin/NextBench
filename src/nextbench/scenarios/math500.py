import weave
import re

from nextbench.scenarios import BaseScenario
from nextbench.utils import RequestResult


@weave.op()
def parse_math_answer(completion: str) -> str:
    return re.search(r'\\boxed{(.*)}', completion).group(1)


class Math500Scenario(BaseScenario):
    dataset_ref: str = "weave:///ayut/NextBench/object/MATH500:YMovIwbHIlH2hxe70wriWldlxHwEfXnkfuDX4nvdbzw"
    system_prompt: str = "Answer the question and return the answer in the \\boxed{} format."

    def preprocess_input(self, row: dict) -> dict:
        assert "question" in row, "Question column is required"
        assert "answer" in row, "Answer column is required"

        question = row["question"]
        prompt = (
            f"Question: {question}\n"
        )
        return {"prompt": prompt}

    def postprocess_output(self, output: RequestResult) -> str:
        raw_text = output.completions[0] if output.completions else ""
        return parse_math_answer(raw_text)


if __name__ == '__main__':
    # Define a dummy metric for testing purposes.
    class DummyMetric(weave.Scorer):
        @weave.op()
        def score(self, answer: str, output: str) -> bool:
            return answer == output

    # Instantiate the scenario.
    scenario = Math500Scenario(metric=DummyMetric())

    # Create a sample input row.
    test_row = {"question": "What is 20 + 22?"}
    input_data = scenario.preprocess_input(test_row)

    # Simulate a RequestResult with a correctly formatted boxed answer.
    test_result = RequestResult(
        success=True,
        cached=False,
        completions=["\\boxed{42}"],
        error=None
    )

    # Process the output.
    result = scenario.postprocess_output(test_result)

    # Print test details.
    print("Preprocessed prompt:", input_data["prompt"])
    print("Processed output:", result)
    print("Test passed:", result == "42")
