from functools import lru_cache
import importlib
from typing import Any, Callable
from datasets import Dataset

from czeval.config.task.load_task import TaskConfig
import czeval.metrics.functions

Metric = int | float


@lru_cache(maxsize=None)
def load_metric(trans_name: str) -> Callable[..., Metric]:
    """
    Loads transform,  using dynamic loading from based on function name from config.
    It should load the function from src/transforms.py
    """

    # Dyamically load function from src/answer_extractors.py
    # use importlib
    function: Callable[..., Metric] = importlib.import_module(
        czeval.metrics.functions.__name__
    ).__dict__[trans_name]
    # Check if function is callable
    if not callable(function):
        raise ValueError(f"Function {trans_name} is not callable")

    return function


def load_metrics(cfg: TaskConfig) -> Callable[[Dataset], dict[str, Metric]]:
    """
    Loads transforms from config
    """
    return lambda dataset: {
        metric.name: load_metric(metric.name)(
            *[dataset[metric_input] for metric_input in metric.inputs]
        )
        for metric in cfg.metrics
    }
