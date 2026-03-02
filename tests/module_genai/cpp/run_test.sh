#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR_UNIT_TEST_CPP="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd ${SCRIPT_DIR_UNIT_TEST_CPP}

source ../../../../source_ov.sh
cd ${SCRIPT_DIR_UNIT_TEST_CPP}

# if not exist libopenvino_tokenizers.so, cpy from build dir
OV_TOKENIZERS_LIB_PATH=${SCRIPT_DIR_UNIT_TEST_CPP}/../../../build/tests/module_genai/cpp/libopenvino_tokenizers.so
if [ ! -f ${OV_TOKENIZERS_LIB_PATH} ]; then
    echo "libopenvino_tokenizers.so not found in build dir, copying..."
    cp ${SCRIPT_DIR_UNIT_TEST_CPP}/../../../build/openvino_genai/libopenvino_tokenizers.so ${OV_TOKENIZERS_LIB_PATH}
fi

export DATA_DIR=${SCRIPT_DIR_UNIT_TEST_CPP}/test_data
export MODEL_DIR=${SCRIPT_DIR_UNIT_TEST_CPP}/test_models

# export DEVICE=GPU             # Specific device for testing, default is CPU
# export ENABLE_PROFILE=1       # Dump profiling data. default 0.
# export DUMP_YAML=1            # Dump pipeline to YAML file. default 0.
# export OPENVINO_LOG_LEVEL=2   # Set OpenVINO log level.

app=../../../build/tests/module_genai/cpp/genai_modules_test

# All tests
# $app

# All ModuleTest examples
$app --gtest_filter="ModuleTest*"

# All PipelineTest examples
# $app --gtest_filter="PipelineTest*"