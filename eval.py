import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import asyncio
import typer
import importlib
from enum import Enum
from typing import Annotated

import weave
from weave.flow import leaderboard
from weave.trace.weave_client import get_ref

from nextbench.clients import OpenAIClient, BaseLLMClient
from nextbench.scenarios import Math500Scenario, MMLUProScenario, BaseScenario
from nextbench.configs.config_registry import (
    register_model_configs, register_scenario_configs, SCENARIO_CONFIGS, MODEL_CONFIGS
)

register_model_configs()
register_scenario_configs()


# Metrics
class ExactMatch(weave.Scorer):
    @weave.op()
    def score(self, answer: str, output: str) -> bool:
        return answer == output
    

class ScenarioChoice(str, Enum):
    math500 = "math500"
    mmlupro = "mmlupro"
    all = "all"


def dynamic_import(class_path: str):
    """
    Dynamically import a class from a string path.
    
    :param class_path: The full path to the class (e.g. "nextbench.clients.openai_client.OpenAIClient").
    :return: The class object.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_scenario(scenario_name: str) -> BaseScenario:
    """Returns an instance of the scenario."""
    scenario_config = SCENARIO_CONFIGS[scenario_name]
    scenario_spec = scenario_config.scenario_spec
    scenario_class = dynamic_import(scenario_spec.class_name)
    # NOTE: We only support one metric per scenario for now. In future we will support multiple metrics.
    metric = ExactMatch() if scenario_spec.metric_name == "exact_match" else None
    scenario = scenario_class(metric=metric)
    return scenario


def load_client(model_name: str, enable_cache: bool) -> BaseLLMClient:
    model_config = MODEL_CONFIGS[model_name]
    client_spec = model_config.client_spec

    client_class = dynamic_import(client_spec.class_name)

    client = client_class(
        **client_spec.args,
        enable_cache=enable_cache,
    )
    client.model = model_name

    return client


async def run_single_evaluation(
    scenario: BaseScenario,
    client: BaseLLMClient,
    num_samples: int,
):
    """
    Run the evaluation for a given scenario and client.

    :param scenario: The scenario instance. This handles the dataset and the metric.
    :param client: The client instance. This handles making the LLM call.
    :param enable_cache: Whether to enable caching.
    :param num_samples: Number of dataset samples to evaluate.
    :return: The obtained results from the evaluation.
    """
    # Initialize the evaluation
    evaluation = weave.Evaluation(
        dataset=scenario.get_dataset_rows(num_samples),
        scorers=[scenario],
        preprocess_model_input=scenario.preprocess_input,
    )
    
    # Configure the system prompt for the client
    client.system_prompt = scenario.system_prompt.content
    
    # Run the evaluation
    eval_name = f"{client.client_name}:{client.model}-{scenario.scenario_name}"
    results = await evaluation.evaluate(
        client, __weave={"display_name": eval_name}
    )
    return results, evaluation


app = typer.Typer()


@app.command()
def evaluate(
    scenario: Annotated[
        ScenarioChoice, typer.Option(case_sensitive=False)
    ] = ScenarioChoice.all,
    model_name: str = typer.Option(
        "gpt-4o",
        help="The name of the model to use for evaluation.",
        case_sensitive=True,
        show_default=True,
        prompt="Please choose a model",
        autocompletion=lambda incomplete: [name for name in MODEL_CONFIGS.keys() if name.startswith(incomplete)]
    ),
    num_samples: int = None,
    enable_cache: bool = True,
):
    """
    Run evaluation scenarios for NextBench.

    SCENARIO: Choose the evaluation scenario ("math500", "mmlupro" or "all").
    NUM_SAMPLES: Number of samples to evaluate from the dataset.
    """
    # Initialize the weave client
    weave.init("nextbench-dev")

    async def run():
        typer.echo(
            typer.style(
                f"Evaluating model: {model_name} on {scenario}", fg=typer.colors.GREEN, bold=True
            )
        )

        scenario_instance = load_scenario(scenario)
        client = load_client(model_name, enable_cache)

        result, evaluation = await run_single_evaluation(
            scenario_instance,
            client,
            num_samples,
        )

    return asyncio.run(run())

if __name__ == "__main__":
    app()
