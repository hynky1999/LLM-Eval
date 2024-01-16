from pathlib import Path
from czeval.run import run
from czeval.config.model import load_model_config
from czeval.config.dataset import load_dataset_config
from czeval.config.task import load_task_config


def test_load_configs():
    root = Path(__file__).parent.parent
    model_cfg = load_model_config(root / "config" / "model" / "openrouter.yaml")
    dataset_cfg = load_dataset_config(root / "config" / "dataset" / "test.yaml")
    task_cfg = load_task_config(root / "config" / "task" / "test.yaml")


def test_run():
    root = Path(__file__).parent.parent
    run(
        model_config_path=root / "config" / "model" / "openrouter.yaml",
        dataset_config_path=root / "config" / "dataset" / "test.yaml",
        task_config_path=root / "config" / "task" / "test.yaml",
    )
