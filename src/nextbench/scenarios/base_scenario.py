import abc
import weave
from nextbench.utils import RequestResult


class BaseScenario(weave.Scorer, metaclass=abc.ABCMeta):
    dataset_ref: str
    system_prompt: str
    metric: weave.Scorer

    def get_dataset_rows(self) -> list[dict]:
        """
        Returns the list of rows (dicts) from the chosen dataset. The dataset is provided as a weave Dataset ref.
        More info: https://weave-docs.wandb.ai/guides/core-types/datasets
        """
        return weave.ref(self.dataset_ref).get().rows

    @abc.abstractmethod
    def preprocess_input(self, row: dict) -> dict:
        """Implement this method to preprocess the input row provided to the client.
        Ideally you should not add the system prompt to the input here. Use this abstract method to add any pre-processing logic.

        Make sure to return a dictionary with a single key "prompt" that maps to the preprocessed input string.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def postprocess_output(self, output: RequestResult) -> str:
        """
        Optional override: parse any special formatting from the model's output.
        Example: if the model returns \(\boxed{42}\), you can parse out '42'.
        Default is to return the raw completion text.
        """
        raise NotImplementedError

    @weave.op()
    def score(self, answer: str, output: RequestResult) -> bool:
        """
        The required method from weave.Scorer.
        Compares the ground truth 'answer' with the parsed output 
        (by default `postprocess_output`) to yield a boolean or numeric metric.
        """
        prediction = self.postprocess_output(output)
        return self.metric.score(answer, prediction)


if __name__ == '__main__':
    # These tests assume that the BaseScenario class (defined in this file)
    # is the abstract base class for scenarios. We create a dummy concrete
    # subclass to allow instantiation and testing.
    from nextbench.utils import RequestResult

    # Define a dummy metric for testing purposes.
    class DummyMetric(weave.Scorer):
        @weave.op()
        def score(self, answer: str, output: str) -> bool:
            return answer == output

    # Define a concrete implementation of the BaseScenario for testing.
    # We assume that the abstract BaseScenario is defined earlier in this module.
    class DummyScenario(BaseScenario):
        metric: weave.Scorer = DummyMetric()
        dataset_ref: str = "test_dataset"
        system_prompt: str = "You are a helpful assistant."

        def preprocess_input(self, row: dict) -> str:
            # For testing, simply return the "question" field if it exists.
            return row.get("question", "")

        def postprocess_output(self, output: RequestResult) -> str:
            # For testing, return the first (and assumed only) completion.
            return output.completions[0] if output.completions else ""

    # Create dummy RequestResult instances to simulate LLM output.
    correct_result = RequestResult(
        success=True,
        cached=False,
        completions=["42"],
        error=None
    )
    incorrect_result = RequestResult(
        success=True,
        cached=False,
        completions=["24"],
        error=None
    )

    # Instantiate our dummy scenario.
    scenario = DummyScenario()

    weave_client = weave.init("nextbench-dev")

    # Test Case 1: When the model output matches the answer.
    answer = "42"
    result_score = scenario.score(answer, correct_result)
    print("Test Case 1 - Correct Answer: Expected True, Got:", result_score)

    # Test Case 2: When the model output does not match the answer.
    result_score = scenario.score(answer, incorrect_result)
    print("Test Case 2 - Incorrect Answer: Expected False, Got:", result_score)
