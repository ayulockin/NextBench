import re

import weave

from nextbench.scenarios import BaseScenario
from nextbench.utils import RequestResult


@weave.op()
def parse_python_code(completion: str) -> str:
    # Search for code block with more flexible pattern
    match = re.search(r"```(?:python)?(.*?)```", completion, re.DOTALL)
    if match is None:
        raise ValueError("No Python code block found in the completion")
    return match.group(1).strip()


class LiveCodeBenchScenario(BaseScenario):
    dataset_ref: str = "weave:///ayut/NextBench/object/LiveCodeBench:latest"
    system_prompt: str = "Come up with a self-contained python code that solves the given problem statement. Return the code in a python code block. Make sure that the code is runnable end to end."
    scenario_name: str = "LiveCodeBench"
    # Since `BaseScenario` is a `weave.Scorer`, we can column map to change the argument name from "answer" to "test_cases" expected by the `score` method.
    column_map: dict[str, str] = {"answer": "test_cases"}

    def preprocess_input(self, row: dict) -> dict:
        assert "question" in row, "question column is required"
        assert "starter_code" in row, "starter_code column is required"
        assert "test_cases" in row, "test_cases column is required"

        question = row["question"]
        # TODO: (ayulockin) affect of starter code in the final eval result)
        # Skipping for now and asking the model to generate the entire runnable code.
        
        prompt = f"Question: {question}\n"
        return {"prompt": prompt}

    def postprocess_output(self, output: RequestResult) -> str:
        raw_python_code = output.completions[0] if output.completions else ""
        return parse_python_code(raw_python_code)
    
    @weave.op()
    def score(self, answer: str, output: str) -> bool:
        import subprocess
        import json

        def run_code(code: str, input_data: str) -> str:
                # Run the code in a subprocess and capture the output
            result = subprocess.run(
                ["python3", "-c", code],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result
            
        # Convert the answer string to a list of dictionaries
        try:
            answer = json.loads(answer)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return False
        
        print("Inside score: ", answer, type(answer))

        for test_case in answer:
            input_data = test_case["input"]
            print("Input data: ", input_data)
            expected_output = test_case["output"]
            print("Expected output: ", expected_output)
            test_type = test_case["testtype"]

            # Run the code with the given input
            actual_output = run_code(output, input_data)
            print("Actual output: ", actual_output)

            # TODO: (ayulockin) add metric

        return True


if __name__ == "__main__":
    # Define a dummy metric for testing purposes.
    class DummyMetric(weave.Scorer):
        @weave.op()
        def score(self, answer: str, output: str) -> bool:
            return answer == output

    # Instantiate the scenario.
    scenario = LiveCodeBenchScenario(metric=DummyMetric())

    # Download the dataset.
    dataset = scenario.get_dataset_rows()

    # Create a sample input row.
    test_row = dataset[0]
    input_data = scenario.preprocess_input(test_row)

    # Simulate a RequestResult with a correctly formatted boxed answer.
    test_result = RequestResult(
        success=True,
        cached=False,
        completions=[
            """
```python
def can_be_abc(s: str) -> bool:
    # If already "abc", no swap is needed.
    if s == "abc":
        return True
    # Try swapping every pair of indices.
    s_list = list(s)
    n = len(s_list)
    for i in range(n):
        for j in range(i + 1, n):
            s_list[i], s_list[j] = s_list[j], s_list[i]
            if "".join(s_list) == "abc":
                return True
            # Swap back to restore original order.
            s_list[i], s_list[j] = s_list[j], s_list[i]
    return False

def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    t = int(input_data[0])
    results = []
    for i in range(1, t + 1):
        s = input_data[i].strip()
        results.append("YES" if can_be_abc(s) else "NO")
    return("\n".join(results))

if __name__ == "__main__":
    main()
```
"""
        ],
        error=None,
    )

    # Process the output.
    result = scenario.postprocess_output(test_result)

    # print("Preprocessed prompt: \n", input_data["prompt"])
    print("Processed output: \n", result)

    print("Test cases: \n", test_row["test_cases"])

    # Score the output.
    score = scenario.score(answer=test_row["test_cases"], output=result)

    # Print test details.
    print("Score: \n", score)
