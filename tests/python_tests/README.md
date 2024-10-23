# OpenVINO™ GenAI Tests

This tests aim to validate support for vanilla and continuous batching GenAI APIs.

## Setup environment

In order to run tests first of all build or install OpenVINO GenAI library, follow instructions [GenAI Library README](../../src/README.md).

Then install requirements for tests:
```sh
pip install -r tests/python_tests/requirements.txt
```

## Run Tests

```sh
python -m pytest tests/python_tests/ -m precommit
```

During the test downloaded HuggingFace (HF) models will be saved into the current directory. If you wish to place them somewhere else you can specify `GENAI_MODELS_PATH_PREFIX` environenment variable, e.g.
```sh
GENAI_MODELS_PATH_PREFIX=$HOME/test_models python -m pytest tests/python_tests/ -m precommit
```

If you have built GenAI library by yourself instead of using wheel please set `PYTHONPATH` so that test could find library, e.g.
```sh
PYTHONPATH=$PYTHONPATH:.../openvino.genai/build-Release/ python -m pytest tests/python_tests/ -m precommit
```

## Customise tests run

Tests have `precommit` and `nightly` set of models. `precommit` contains lightweight models which can be quickly inferred, `nightly` models are heavier and required more time for interence. If you wish to run specific tests only for nightly models, you can use `-k` option, for example to run only multibatch and chat tests:
```sh
python -m pytest tests/python_tests/ -m nightly -k "test_multibatch and test_chat"
```

If you wish to run all tests except beam search do the following:
```sh
python -m pytest tests/python_tests/ -m precommit -k "not test_beam_search"
```

Argument `--model_ids` can be used to run tests selectively only for specific models. HF model ids should be separated by space, e.g:
```sh
python -m pytest tests/python_tests/ -m nightly -k "test_multibatch" --model_ids "TinyLlama/TinyLlama-1.1B-Chat-v1.0 Qwen/Qwen2-0.5B-Instruct"
```

List of currently supported `nightly` and `precommit` models can be found in tests/python_tests/ov_genai_test_utils.py:get_models_list
