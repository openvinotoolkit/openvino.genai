# ModulePipeline Python Unit Tests

This directory contains unit tests for the ModulePipeline Python bindings.

## Test Coverage

- **ValidationResult** - Test validation result structure and attributes
- **validate_config / validate_config_string** - Test YAML configuration validation
- **comfyui_json_to_yaml / comfyui_json_string_to_yaml** - Test ComfyUI JSON to YAML conversion
- **ConfigModelsMap** - Test ModulePipeline constructor with models map

## Setup

### 1. Install Test Dependencies

GENAI_ROOT_DIR=${SCRIPT_DIR_GENAI_MODULE_PY}/../../../../openvino.genai/install/python/
export PYTHONPATH=${GENAI_ROOT_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}/../runtime/lib/intel64/:$LD_LIBRARY_PATH

```sh
pip install -r tests/python_tests/requirements.txt
```

### 2. Ensure OpenVINO GenAI is Installed

Make sure you have built or installed the OpenVINO GenAI library. Follow the instructions in [GenAI Library README](../../../src/README.md).

If you built GenAI library locally, set the `PYTHONPATH`:

```sh
# Linux/macOS
export PYTHONPATH=$PYTHONPATH:/path/to/openvino.genai/build-Release/

# Windows PowerShell
$env:PYTHONPATH = "$env:PYTHONPATH;C:\path\to\openvino.genai\cmake-build-debug-visual-studio"
```

## Run Tests

### Run All ModulePipeline Tests

```sh
python -m pytest tests/module_genai/python/test_module_pipeline_api.py -v
```

### Run Specific Test Class

```sh
# Run only ValidationResult tests
python -m pytest tests/module_genai/python/test_module_pipeline_api.py::TestValidationResult -v

# Run only ComfyUI conversion tests
python -m pytest tests/module_genai/python/test_module_pipeline_api.py::TestComfyUIJsonConversion -v

# Run only ConfigModelsMap tests
python -m pytest tests/module_genai/python/test_module_pipeline_api.py::TestConfigModelsMap -v
```

### Run Specific Test Case

```sh
python -m pytest tests/module_genai/python/test_module_pipeline_api.py::TestValidationResult::test_validation_result_default_constructor -v
```

### Run with Pattern Matching

```sh
# Run tests matching "valid"
python -m pytest tests/module_genai/python/test_module_pipeline_api.py -k "valid" -v

# Run tests matching "comfyui"
python -m pytest tests/module_genai/python/test_module_pipeline_api.py -k "comfyui" -v
```

## Common Options

| Option | Description |
|--------|-------------|
| `-v` | Verbose output |
| `-s` | Show print statements |
| `-k "pattern"` | Run tests matching pattern |
| `--tb=short` | Shorter traceback on failure |
| `--tb=no` | No traceback on failure |
| `-x` | Stop on first failure |

## Example Output

```
$ python -m pytest tests/module_genai/python/test_module_pipeline_api.py -v

collected 23 items

test_module_pipeline_api.py::TestValidationResult::test_validation_result_default_constructor PASSED
test_module_pipeline_api.py::TestValidationResult::test_validation_result_attributes PASSED
test_module_pipeline_api.py::TestValidationResult::test_validation_result_repr PASSED
...
================================================ 23 passed in 0.23s ================================================
```
