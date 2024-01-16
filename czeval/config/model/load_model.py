import logging
from pathlib import Path
from pydantic import BaseModel, Field, RootModel
import yaml
from typing import Annotated, Literal, Union

logger = logging.getLogger(__name__)


class OpenRouterModelConfig(BaseModel):
    type: Literal["openrouter"]
    name: str
    max_requests_per_second: int = Field(1)


class HuggingfaceModelConfig(BaseModel):
    type: Literal["huggingface"]
    name: str


class ModelConfig(RootModel):
    root: Annotated[
        Union[OpenRouterModelConfig, HuggingfaceModelConfig],
        Field(..., discriminator="type"),
    ]


def load_model_config(file_path: Path) -> ModelConfig:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        return ModelConfig.model_validate(data)
