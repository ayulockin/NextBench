import asyncio
import warnings
import weave
import typer
from enum import Enum
from typing import Annotated

# Ignore user warnings
warnings.filterwarnings("ignore", category=UserWarning)

from nextbench.clients import OpenAIClient
from nextbench.scenarios import Math500Scenario, MMLUProScenario

# Metrics
class ExactMatch(weave.Scorer):
    @weave.op()
    def score(self, answer: str, output: str) -> bool:
        return answer == output
    

class ScenarioChoice(str, Enum):
    math500 = "math500"
    mmlupro = "mmlupro"
    all = "all"


def get_scenario_and_dataset(scenario_name: str, num_samples: int):
    """
    Returns an instance of the scenario and a subset of its dataset.
    
    :param scenario_name: Name of the scenario ('math500' or 'mmlupro').
    :param num_samples: Number of dataset samples to evaluate.
    :return: A tuple (scenario, dataset).
    """
    if scenario_name == "math500":
        scenario = Math500Scenario(metric=ExactMatch())
    elif scenario_name == "mmlupro":
        scenario = MMLUProScenario(metric=ExactMatch())
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    dataset = scenario.get_dataset_rows()[:num_samples]
    return scenario, dataset


async def run_evaluation(scenario, dataset, enable_cache: bool):
    """
    Runs the evaluation for a given scenario and dataset.
    
    :param scenario: The scenario instance.
    :param dataset: A subset of dataset rows.
    :return: The obtained results from the evaluation.
    """
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[scenario],
        preprocess_model_input=scenario.preprocess_input,
    )
    # TODO: make this configurable
    client = OpenAIClient(
        model="gpt-4o-mini",
        temperature=0.0,
        max_completion_tokens=4096,
        system_prompt=scenario.system_prompt.content,
        enable_cache=enable_cache,
    )
    results = await evaluation.evaluate(client)
    return results


app = typer.Typer()


@app.command()
def evaluate(
    scenario: Annotated[
        ScenarioChoice, typer.Option(case_sensitive=False)
    ] = ScenarioChoice.all,
    num_samples: int = 2,
    enable_cache: bool = True,
):
    """
    Run evaluation scenarios for NextBench.

    SCENARIO: Choose the evaluation scenario ("math500", "mmlupro" or "all").
    NUM_SAMPLES: Number of samples to evaluate from the dataset.
    """
    # Initialize the weave client
    weave.init("nextbench-dev")

    # Determine which scenarios to run
    if scenario == ScenarioChoice.all:
        # TODO: make this better 
        scenario_names = [ScenarioChoice.math500.value, ScenarioChoice.mmlupro.value]
    else:
        scenario_names = [scenario.value]

    async def run_all():
        for scenario_name in scenario_names:
            typer.echo(
                typer.style(
                    f"Running evaluation for '{scenario_name}' scenario...", fg=typer.colors.GREEN, bold=True
                )
            )
            scenario_instance, dataset = get_scenario_and_dataset(scenario_name, num_samples)
            result = await run_evaluation(scenario_instance, dataset, enable_cache)
            # TODO: save the results to a file?

    asyncio.run(run_all())


if __name__ == "__main__":
    app()