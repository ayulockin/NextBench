import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import asyncio
import importlib
from enum import Enum
from typing import Optional

import typer
import weave
from weave.flow import leaderboard
from weave.trace.weave_client import get_ref

from nextbench.clients import BaseLLMClient
from nextbench.configs.config_registry import (MODEL_CONFIGS, SCENARIO_CONFIGS,
                                               register_model_configs,
                                               register_scenario_configs)
from nextbench.scenarios import BaseScenario

register_scenario_configs()
register_model_configs()

app = typer.Typer()


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
    metric = ExactMatch() if scenario_spec.metric_name == "exact_match" else None  # type: ignore
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
    num_samples: Optional[int],
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
    evaluation = weave.Evaluation(  # type: ignore
        dataset=scenario.get_dataset_rows(num_samples),
        scorers=[scenario],
        preprocess_model_input=scenario.preprocess_input,
    )

    # Configure the system prompt for the client
    client.system_prompt = scenario.system_prompt.content

    # Run the evaluation
    eval_name = f"{client.client_name}:{client.model}-{scenario.scenario_name}"
    results = await evaluation.evaluate(client, __weave={"display_name": eval_name})
    return results, evaluation


async def run(
    scenario_name: str,
    model_name: str,
    num_samples: Optional[int],
    enable_cache: bool,
):
    typer.echo(
        typer.style(
            f"Evaluating model: {model_name} on {scenario_name}",
            fg=typer.colors.GREEN,
            bold=True,
        )
    )

    scenario_instance = load_scenario(scenario_name)
    client = load_client(model_name, enable_cache)

    result, evaluation = await run_single_evaluation(
        scenario_instance,
        client,
        num_samples,
    )

    return result, evaluation


@app.command()
def evaluate(
    scenario_name: str = typer.Option(
        help="The name of the scenario to evaluate.",
        case_sensitive=True,
    ),
    model_name: str = typer.Option(
        help="The name of the model to use for evaluation.",
        case_sensitive=True,
        prompt="Please choose a model",
        autocompletion=lambda incomplete: [
            name for name in MODEL_CONFIGS.keys() if name.startswith(incomplete)
        ],
    ),
    num_samples: Optional[int] = typer.Option(
        help="The number of samples to evaluate. If not provided, all samples will be evaluated.",
        default=None,
    ),
    enable_cache: bool = typer.Option(
        help="Whether to enable caching.",
        default=True,
    ),
    weave_project: str = typer.Option(
        help="The W&B project where you want to log the evaluations.",
        default="nextbench-dev",
    ),
    weave_entity: Optional[str] = typer.Option(
        help="The W&B entity where you want to log the evaluations.",
        default=None,
    ),
):
    """
    Run evaluation scenarios for NextBench.

    :param scenario: The scenario to evaluate.
    :param model_name: The name of the model to use for evaluation.
    :param num_samples: The number of samples to evaluate.
    :param enable_cache: Whether to enable caching.
    """
    # Initialize the weave client
    if weave_entity is None:
        weave.init(weave_project)
    else:
        weave.init(f"{weave_entity}/{weave_project}")

    return asyncio.run(run(scenario_name, model_name, num_samples, enable_cache))


if __name__ == "__main__":
    app()
