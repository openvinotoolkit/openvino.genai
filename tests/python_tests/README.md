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
python -m pytest tests/python_tests/
```

If you have built GenAI library by yourself instead of using wheel please set `PYTHONPATH` so that test could find library, e.g.
```sh
PYTHONPATH=$PYTHONPATH:.../openvino.genai/build-Release/ python -m pytest tests/python_tests/
```

## Customize tests run

Tests have different sets of models for different purposes. If you wish to run specific tests, you can use `-k` option, for example to run only multibatch and chat tests:
```sh
python -m pytest tests/python_tests/ -k "test_multibatch and test_chat"
```

If you wish to run all tests except beam search do the following:
```sh
python -m pytest tests/python_tests/ -k "not test_beam_search"
```

Argument `--model_ids` can be used to run tests selectively only for specific models. HF model ids should be separated by space, e.g:
```sh
python -m pytest tests/python_tests/ -k "test_multibatch" --model_ids "TinyLlama/TinyLlama-1.1B-Chat-v1.0 Qwen/Qwen2-0.5B-Instruct"
```

List of currently supported models can be found in tests/python_tests/data/models.py:get_models_list

## Test Samples
To test samples, set the `SAMPLES_PY_DIR` and `SAMPLES_CPP_DIR` environment variables to the directories containing your Python samples and built C++ samples respectively. The `SAMPLES_CPP_DIR` should point to the folder with built C++ samples, which can be installed using `cmake --component samples_bin`. For example:
```sh
SAMPLES_PY_DIR=openvino.genai/samples/python SAMPLES_CPP_DIR=openvino.genai/samples_bin python -m pytest tests/python_tests -m samples
```

You can also use markers such as `llm` and `whisper` to run specific sets of tests. For example, to run only the `llm` tests:
```sh
python -m pytest tests/python_tests/samples -m llm
```

Or to run only the `whisper` tests:
```sh
python -m pytest tests/python_tests/samples -m whisper
```

Downloaded and converted models are automatically cached using pytest's cache mechanism in `~/.pytest_cache/ov_models/` with 24-hour expiration. The cache is organized by date and package versions to ensure compatibility. Old cache entries are automatically cleaned up. The cache expiration can be customized using the `OV_CACHE_EXPIRY_HOURS` environment variable:
```sh
OV_CACHE_EXPIRY_HOURS=48 python -m pytest tests/python_tests -m samples
```

If the `CLEANUP_CACHE` environment variable is set, all downloaded and converted models will be removed right after the tests have stopped using them. Note that this does not affect the HuggingFace (HF) cache. For example:
```sh
CLEANUP_CACHE=1 python -m pytest tests/python_tests -m samples
```

## Cache Configuration

Models and test data are cached using pytest's built-in cache mechanism. The cache location can be customized:

```sh
# Use custom cache directory
python -m pytest tests/python_tests/ -m precommit -o cache_dir=/path/to/custom/cache

# Use default cache location (~/.pytest_cache/)
python -m pytest tests/python_tests/ -m precommit
```

The model cache automatically expires after 24 hours and is organized by date and package versions. You can clear the cache manually:

```sh
# Clear all pytest cache
python -m pytest tests/python_tests/ --cache-clear
```
