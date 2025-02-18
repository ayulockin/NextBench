import os
import yaml
from typing import Any
from pydantic import BaseModel

MODEL_DEPLOYMENTS_FILE: str = "models.yaml"


class ClientSpec(BaseModel):
    class_name: str
    args: dict[str, Any]


class ModelConfig(BaseModel):
    name: str
    display_name: str
    creator_organization_name: str
    access: str
    release_date: str
    client_spec: ClientSpec


MODEL_CONFIGS: dict[str, ModelConfig] = {}


def register_model_configs() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    model_deployments_path = os.path.join(repo_root, MODEL_DEPLOYMENTS_FILE)
    if not os.path.exists(model_deployments_path):
        raise FileNotFoundError(f"Model deployments file not found at {model_deployments_path}")

    with open(model_deployments_path, "r") as f:
        model_deployments = yaml.safe_load(f)["models"]

    for model_deployment in model_deployments:
        model_config = ModelConfig(**model_deployment)
        MODEL_CONFIGS[model_config.name] = model_config


if __name__ == "__main__":
    register_model_configs()
    print(MODEL_CONFIGS)
