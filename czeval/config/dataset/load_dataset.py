from pathlib import Path
from pydantic import BaseModel
import yaml
from typing import Optional


class DatasetConfig(BaseModel):
    name: str
    split: str
    size: Optional[int] = None


def load_dataset_config(file_path: Path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        config = DatasetConfig.model_validate(data)
        return config
