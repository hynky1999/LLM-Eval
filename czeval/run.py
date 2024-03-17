from typing import Any, Awaitable, Callable
from czeval.config.dataset.load_dataset import load_dataset_config
from czeval.config.model import load_model_config, ModelConfig, OpenRouterModelConfig
from czeval.config.model.load_model import HuggingfaceModelConfig
from czeval.config.task import load_task_config, TaskConfig
from czeval.config.dataset import load_dataset_config, DatasetConfig
from czeval.model.open_router import predict_samples as open_router_predict_samples
from czeval.model.modal import predict_samples as modal_predict_samples
from datasets import load_dataset as hf_load_dataset, Dataset, concatenate_datasets
from functools import partial
import mlflow
from czeval.prompt_utils import load_prompt_file, prepare_conversations
from datetime import datetime
from czeval.metrics.utils import Metric, load_metrics
from czeval.transforms import load_transforms
from czeval.utils import maybe


def load_dataset(dataset_config: DatasetConfig, task_config: TaskConfig) -> Dataset:
    """
    Load dataset from huggingface,
    restricts it's length based on config
    """

    # Load dataset
    dataset_dict = hf_load_dataset(dataset_config.name)
    dataset: Dataset = dataset_dict[dataset_config.split]  # type: ignore
    sys_cfg = load_prompt_file(task_config.prompts.system_prompt_path)
    user_cfg = load_prompt_file(task_config.prompts.user_prompt_path)
    dataset = dataset.map(
        lambda x: {
            "messages": prepare_conversations(
                x,
                sys_cfg,
                user_cfg,
                task_config.prompts.variables,
            )
        }
    )

    # Restrict dataset length
    if dataset_config.size:
        dataset = dataset.select(range(dataset_config.size))

    return dataset


def load_model_prediction_method(
    model_config: ModelConfig,
) -> Callable[[list[dict]], list[dict]]:
    """
    Loads model prediction method
    """

    # Load model
    if isinstance(model_config.root, OpenRouterModelConfig):
        open_router_cfg: OpenRouterModelConfig = model_config.root
        model_fc = partial(
            open_router_predict_samples,
            model=open_router_cfg.name,
            temp=open_router_cfg.temperate,
            max_tokens=open_router_cfg.max_tokens,
            max_requests_per_second=open_router_cfg.max_requests_per_second,
        )

    elif isinstance(model_config.root, HuggingfaceModelConfig):
        hf_cfg: HuggingfaceModelConfig = model_config.root
        model_fc = partial(
            modal_predict_samples,
            model=hf_cfg.name,
            chat_template=hf_cfg.chat_template,
            batch_size=hf_cfg.batch_size,
            max_tokens=hf_cfg.max_tokens,
            temp=hf_cfg.temperate,
        )

    else:
        raise ValueError(f"Model type {model_config.root.type} not supported")

    return model_fc


def push_results_to_mlflow(dataset: Dataset, metrics: dict[str, Metric]):
    """
    Pushes results and metrics to mlflow
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    # Create a table for dataset

    mlflow.log_table(artifact_file="dataset.json", data=dataset.to_pandas())


def get_params(
    dst_config: DatasetConfig, model_config: ModelConfig, task_config: TaskConfig
):
    params = {
        **{f"dataset_{k}": v for k, v in dst_config.model_dump().items()},
        **{f"model_{k}": v for k, v in model_config.model_dump().items()},
        **task_config.model_dump(),
    }
    return params


def _run(dst_config: DatasetConfig, model_config: ModelConfig, task_config: TaskConfig):
    # Construct the name
    experiment_name = f"{dst_config.name}/{dst_config.split}".replace("/", "-")
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{experiment_name}-{model_config.root.name}-{task_config.info.name}-{current_datetime}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # Load dataset configuration
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        mlflow.log_params(get_params(dst_config, model_config, task_config))

        # Load entities first, so that we fail quickly if they are not valid
        dataset = load_dataset(dst_config, task_config)
        model = load_model_prediction_method(model_config)
        metrics_calculator = load_metrics(task_config)
        transforms = load_transforms(task_config)

        # Predict results
        model_results = model(dataset["messages"])

        # Append results to dataset
        dataset = concatenate_datasets(
            [
                dataset,
                Dataset.from_dict(
                    {"answer": model_results[0], "model_output": model_results[0], "completion_tokens": model_results[1], "prompt_tokens": model_results[2]}
                ),
            ],
            axis=1,
        )

        # Duplicate the model result

        # Extract answers
        dataset = dataset.map(
            lambda x: transforms(x),
        )

        # Compute metrics
        metrics = metrics_calculator(dataset)

        # Save results and metrics to database
        push_results_to_mlflow(dataset, metrics)


def run(dataset_config_path, model_config_path, task_config_path):
    dataset_config = load_dataset_config(dataset_config_path)
    model_config = load_model_config(model_config_path)
    task_config = load_task_config(task_config_path)

    _run(dataset_config, model_config, task_config)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config-path", type=str, required=True)
    parser.add_argument("--model-config-path", type=str, required=True)
    parser.add_argument("--task-config-path", type=str, required=True)
    args = parser.parse_args()

    run(args.dataset_config_path, args.model_config_path, args.task_config_path)


if __name__ == "__main__":
    main()
