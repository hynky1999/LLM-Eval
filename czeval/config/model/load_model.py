import logging
from pathlib import Path
from pydantic import BaseModel, Field, RootModel
import yaml
from pydantic import validator
from typing import Annotated, Literal, Union

logger = logging.getLogger(__name__)


class CommonModelConfig(BaseModel):
    temperate: float = 0.001
    max_tokens: int = 2048


class OpenRouterModelConfig(CommonModelConfig):
    type: Literal["openrouter"]
    name: str
    max_requests_per_second: int = Field(1, ge=1)


class HuggingfaceModelConfig(CommonModelConfig):
    type: Literal["huggingface"]
    name: str
    chat_template_path: str | None = None
    chat_template: str | None = None
    batch_size: int | None = None

    @validator("chat_template", pre=True, always=True)
    def set_chat_template(cls, v, values):
        chat_template_path = values.get("chat_template_path")
        if chat_template_path is not None and Path(chat_template_path).is_file():
            text = Path(chat_template_path).read_text()
            return text
        return v


class ModelConfig(RootModel):
    root: Annotated[
        Union[OpenRouterModelConfig, HuggingfaceModelConfig],
        Field(..., discriminator="type"),
    ]


def load_model_config(file_path: Path) -> ModelConfig:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        return ModelConfig.model_validate(data)
