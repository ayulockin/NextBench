import os
from typing import Any

import yaml
from pydantic import BaseModel

MODEL_DEPLOYMENTS_FILE: str = "models.yaml"
SCENARIO_DEPLOYMENTS_FILE: str = "scenarios.yaml"


class ClientSpec(BaseModel):
    class_name: str
    args: dict[str, Any]


class ScenarioSpec(BaseModel):
    class_name: str
    # NOTE: metric_name will be a list of strs in the future and they will represent the class names of the metrics.
    metric_name: str


class ModelConfig(BaseModel):
    name: str
    display_name: str
    creator_organization_name: str
    access: str
    release_date: str
    client_spec: ClientSpec


class ScenarioConfig(BaseModel):
    name: str
    display_name: str
    creator_organization_name: str
    scenario_spec: ScenarioSpec


MODEL_CONFIGS: dict[str, ModelConfig] = {}
SCENARIO_CONFIGS: dict[str, ScenarioConfig] = {}


def register_model_configs() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    model_deployments_path = os.path.join(repo_root, MODEL_DEPLOYMENTS_FILE)
    if not os.path.exists(model_deployments_path):
        raise FileNotFoundError(
            f"Model deployments file not found at {model_deployments_path}"
        )

    with open(model_deployments_path, "r") as f:
        model_deployments = yaml.safe_load(f)["models"]

    for model_deployment in model_deployments:
        model_config = ModelConfig(**model_deployment)
        MODEL_CONFIGS[model_config.name] = model_config


def register_scenario_configs() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    scenario_deployments_path = os.path.join(repo_root, SCENARIO_DEPLOYMENTS_FILE)
    if not os.path.exists(scenario_deployments_path):
        raise FileNotFoundError(
            f"Scenario deployments file not found at {scenario_deployments_path}"
        )

    with open(scenario_deployments_path, "r") as f:
        scenario_deployments = yaml.safe_load(f)["scenarios"]

    for scenario_deployment in scenario_deployments:
        scenario_config = ScenarioConfig(**scenario_deployment)
        SCENARIO_CONFIGS[scenario_config.name] = scenario_config


if __name__ == "__main__":
    register_model_configs()
    register_scenario_configs()
    print(MODEL_CONFIGS)
    print(SCENARIO_CONFIGS)
