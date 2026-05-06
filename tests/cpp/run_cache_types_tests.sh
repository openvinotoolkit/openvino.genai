#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Converts all models listed in data/cache_types_models.csv (if needed),
# rebuilds the test binary, and runs the real-model cache type and backend routing tests.
#
# Usage:
#   ./run_cache_types_tests.sh [BUILD_DIR] [MODELS_BASE_DIR]
#
# Defaults:
#   BUILD_DIR        ../../../build
#   MODELS_BASE_DIR  /tmp/ov_cache_types_models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="${SCRIPT_DIR}/data/cache_types_models.csv"

BUILD_DIR="${1:-${SCRIPT_DIR}/../../build}"
MODELS_BASE_DIR="${2:-/tmp/ov_cache_types_models}"

resolve_test_binary() {
    local candidates=(
        "${BUILD_DIR}/bin/tests_continuous_batching"
        "${BUILD_DIR}/vscode-__unspec__/bin/tests_continuous_batching"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -x "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    echo "${BUILD_DIR}/bin/tests_continuous_batching"
}

resolve_tokenizers_library() {
    local candidates=(
        "${BUILD_DIR}/openvino_genai/libopenvino_tokenizers.so"
        "${BUILD_DIR}/bin/libopenvino_tokenizers.so"
        "${BUILD_DIR}/vscode-__unspec__/openvino_genai/libopenvino_tokenizers.so"
        "${BUILD_DIR}/vscode-__unspec__/bin/libopenvino_tokenizers.so"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

TEST_BINARY="$(resolve_test_binary)"

# ── Validate optimum-cli ────────────────────────────────────────────────────
if ! command -v optimum-cli &>/dev/null; then
    echo "ERROR: optimum-cli not found. Install it with:" >&2
    echo "  pip install optimum[openvino]" >&2
    exit 1
fi

# ── Build test binary ───────────────────────────────────────────────────────
echo "Building tests_continuous_batching ..."
cmake --build "${BUILD_DIR}" --target tests_continuous_batching -j"$(nproc)"

# ── Convert models listed in the CSV ────────────────────────────────────────
mkdir -p "${MODELS_BASE_DIR}"

while IFS=',' read -r model_id _rest; do
    # Skip blank lines and comment lines
    [[ -z "${model_id}" || "${model_id}" =~ ^[[:space:]]*# ]] && continue

    model_name="${model_id##*/}"   # keep the part after the last '/'
    model_dir="${MODELS_BASE_DIR}/${model_name}"

    if [[ -f "${model_dir}/openvino_model.xml" ]]; then
        echo "[skip] ${model_id} — already converted at ${model_dir}"
    else
        echo "[convert] ${model_id} → ${model_dir} ..."
        optimum-cli export openvino \
            -m "${model_id}" \
            --task text-generation-with-past \
            "${model_dir}"
        echo "[done] ${model_id}"
    fi
done < "${CSV_FILE}"

# ── Run tests ───────────────────────────────────────────────────────────────
echo ""
echo "Running real-model cache type and backend routing tests ..."
echo "  TEST_MODELS_BASE_DIR = ${MODELS_BASE_DIR}"
echo "  CACHE_TYPES_CSV      = ${CSV_FILE}"
echo ""
export TEST_MODELS_BASE_DIR="${MODELS_BASE_DIR}"
export CACHE_TYPES_CSV="${CSV_FILE}"
if [[ ! -x "${TEST_BINARY}" ]]; then
    echo "ERROR: tests_continuous_batching not found: ${TEST_BINARY}" >&2
    exit 1
fi

TOKENIZERS_LIB="$(resolve_tokenizers_library || true)"
if [[ -n "${TOKENIZERS_LIB}" ]]; then
    if [[ -n "${OPENVINO_TOKENIZERS_PATH_GENAI:-}" && ! -f "${OPENVINO_TOKENIZERS_PATH_GENAI}" ]]; then
        echo "Overriding stale OPENVINO_TOKENIZERS_PATH_GENAI=${OPENVINO_TOKENIZERS_PATH_GENAI}" >&2
    fi
    export OPENVINO_TOKENIZERS_PATH_GENAI="${TOKENIZERS_LIB}"
    echo "  OPENVINO_TOKENIZERS_PATH_GENAI = ${OPENVINO_TOKENIZERS_PATH_GENAI}"
else
    echo "WARNING: libopenvino_tokenizers.so was not found under ${BUILD_DIR}; relying on existing environment" >&2
fi
"${TEST_BINARY}" --gtest_filter="*GetCacheTypesRealModel*:*LLMPipelineBackendRealModel*"
