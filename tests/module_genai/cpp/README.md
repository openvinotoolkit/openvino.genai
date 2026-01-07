# Unit Test

Each module's unit test.

# How to run

Refer script: run_test.sh

```
cd [PATH]/openvino.genai/tests/module_genai/cpp

export DATA_DIR=./test_data
export MODEL_DIR=./test_models
export DUMP_YAML=1  # Dump config yaml to file.

../../../build/tests/module_genai/cpp/genai_modules_test

<!-- Filter test example -->
../../../build/tests/module_genai/cpp/genai_modules_test --gtest_filter="*cat_120_100_dog_120_120*"

```