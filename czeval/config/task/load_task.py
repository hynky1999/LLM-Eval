from pathlib import Path
from pydantic import BaseModel, Field, RootModel, validator
import yaml
from typing import Any, List


class InfoConfig(BaseModel):
    name: str
    description: str


class PromptsConfig(BaseModel):
    system_prompt_path: str
    user_prompt_path: str
    variables: List[str]


class TransformConfig(BaseModel):
    name: str
    kwargs: dict[str, Any] = Field({}, alias="kwargs")
    args: list[Any] = Field([], alias="args")


class OutputConfig(BaseModel):
    key: str
    transform: TransformConfig


class MetricConfig(BaseModel):
    name: str
    inputs: List[str]


class TaskConfig(BaseModel):
    info: InfoConfig
    prompts: PromptsConfig
    outputs: list[OutputConfig] = Field(..., alias="outputs")
    metrics: list[MetricConfig] = Field(..., alias="metrics")


def load_task_config(file_path: Path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        config = TaskConfig(**data)
        return config
