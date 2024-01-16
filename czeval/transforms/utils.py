from functools import lru_cache
import importlib
from typing import Any, Callable
from czeval.config.task import OutputConfig
from czeval.config.task.load_task import TaskConfig
import czeval.transforms.functions


def identity(x):
    return x


@lru_cache(maxsize=None)
def load_transform(trans_name: str) -> Callable[[Any], Any]:
    """
    Loads transform,  using dynamic loading from based on function name from config.
    It should load the function from src/transforms.py
    """

    # Dyamically load function from src/answer_extractors.py
    # use importlib
    function: Callable[..., Any] = importlib.import_module(
        czeval.transforms.functions.__name__
    ).__dict__[trans_name]
    # Check if function is callable
    if not callable(function):
        raise ValueError(f"Function {trans_name} is not callable")

    return function


def load_transforms(
    cfgs: TaskConfig,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Loads transforms from config
    """
    return lambda dct: {
        cfg.key: (
            load_transform(cfg.transform.name)(
                dct[cfg.key], *cfg.transform.args, **cfg.transform.kwargs
            )
            if cfg.transform
            else dct[cfg.key]
        )
        for cfg in cfgs.outputs
    }
