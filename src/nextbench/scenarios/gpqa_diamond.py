import re

import weave

from nextbench.scenarios import BaseScenario
from nextbench.utils import RequestResult


@weave.op()
def parse_option_number(completion: str) -> str:
    return re.search(r"\\boxed{(.*)}", completion).group(1)


class GPQADiamondScenario(BaseScenario):
    dataset_ref: str = "weave:///ayut/NextBench/object/GPQA-Diamond:latest"
    system_prompt: str = (
        "Select the most appropriate answer from the given options and return the correct option number in the \\boxed{} format"
    )
    scenario_name: str = "GPQA-Diamond"

    def preprocess_input(self, row: dict) -> dict:
        assert "question" in row, "Question column is required"
        assert "options" in row, "Options column is required"
        assert "answer" in row, "Answer column is required"

        question = row["question"]

        options = list(row["options"])
        assert isinstance(options, list)
        # TODO: (ayulockin) shuffle options to avoid bias?
        options_str = "\n".join(
            [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
        )

        prompt = f"Question: {question}\n" f"{options_str}\n"
        return {"prompt": prompt}

    def postprocess_output(self, output: RequestResult) -> str:
        raw_text = output.completions[0] if output.completions else ""
        return parse_option_number(raw_text)


if __name__ == "__main__":
    # Define a dummy metric for testing purposes.
    class DummyMetric(weave.Scorer):
        @weave.op()
        def score(self, answer: str, output: str) -> bool:
            return answer == output

    # Instantiate the scenario.
    scenario = GPQADiamondScenario(metric=DummyMetric())

    # Download the dataset.
    dataset = scenario.get_dataset_rows()

    # Create a sample input row.
    test_row = dataset[0]
    input_data = scenario.preprocess_input(test_row)

    # Simulate a RequestResult with a correctly formatted boxed answer.
    test_result = RequestResult(
        success=True, cached=False, completions=["\\boxed{C}"], error=None
    )

    # Process the output.
    result = scenario.postprocess_output(test_result)

    # Print test details.
    print("Preprocessed prompt: \n", input_data["prompt"])
    print("Processed output: \n", result)
    print("Test passed: \n", result == "C")
