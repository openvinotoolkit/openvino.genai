SCRIPT_DIR_UNIT_TEST_CPP="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_UNIT_TEST_CPP}

export DATA_DIR=${SCRIPT_DIR_UNIT_TEST_CPP}/test_data
export MODEL_DIR=${SCRIPT_DIR_UNIT_TEST_CPP}/test_models

app=../../../build/tests/module_genai/cpp/genai_modules_test

$app
# $app --gtest_filter="*cat_120_100_dog_120_120*"