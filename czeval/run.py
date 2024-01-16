from typing import Any, Awaitable, Callable
from czeval.config.dataset.load_dataset import load_dataset_config
from czeval.config.model import load_model_config, ModelConfig, OpenRouterModelConfig
from czeval.config.task import load_task_config, TaskConfig
from czeval.config.dataset import load_dataset_config, DatasetConfig
from czeval.model.open_router import predict_samples
from datasets import load_dataset as hf_load_dataset, Dataset, concatenate_datasets
from functools import partial
import mlflow
from czeval.prompt_utils import load_prompt_file, prepare_conversations
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
    dataset = dataset.map(
        lambda x: {
            "messages": prepare_conversations(
                x,
                load_prompt_file(task_config.prompts.system_prompt_path),
                load_prompt_file(task_config.prompts.user_prompt_path),
                task_config.prompts.variables,
            )
        },
    )

    # Restrict dataset length
    if dataset_config.size:
        dataset = dataset.select(range(dataset_config.size))

    return dataset


def load_model_prediction_method(
    model_config: OpenRouterModelConfig,
) -> Callable[[list[dict]], list[dict]]:
    """
    Loads model prediction method
    """

    # Load model
    model_fc = partial(
        predict_samples,
        model=model_config.name,
        max_requests_per_second=model_config.max_requests_per_second,
    )

    return model_fc


def push_results_to_mlflow(dataset: Dataset, metrics: dict[str, Metric]):
    """
    Pushes results and metrics to mlflow
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    # Create a table for dataset

    mlflow.log_table(artifact_file="dataset.csv", data=dataset.to_pandas())


def _run(dst_config: DatasetConfig, model_config: ModelConfig, task_config: TaskConfig):
    # Load dataset configuration
    with mlflow.start_run():
        # Load entities first, so that we fail quickly if they are not valid
        dataset = load_dataset(dst_config, task_config)
        model = load_model_prediction_method(model_config.root)
        metrics_calculator = load_metrics(task_config)
        transforms = load_transforms(task_config)

        # Predict results
        model_results = model(dataset["messages"])

        # Append results to dataset
        dataset = concatenate_datasets(
            [dataset, Dataset.from_dict({"answer": model_results})], axis=1
        )

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
