## Overview
CZ-EVAL is an evaluation project designed to provide a compressive toolkit for evaluation of LLM models.

Supported tasks include:
- Q&A

Supported datasets include:
- Anything from huggingface
- TODO: Local datasets

Supported models include:
- Anything from openrouter
- Modal with local models
- TODO: Local models

## Installation

To install the required dependencies, use Poetry which is specified in the pyproject.toml file.


## Usage
```bash
python -m czeval.run --dataset-config-path=dataset --model-config-path=model --task-config-path=task