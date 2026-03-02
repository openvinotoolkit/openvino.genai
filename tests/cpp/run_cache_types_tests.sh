#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Converts all models listed in data/cache_types_models.csv (if needed),
# rebuilds the test binary, and runs the GetCacheTypesRealModel tests.
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

BUILD_DIR="${1:-${SCRIPT_DIR}/../../../build}"
MODELS_BASE_DIR="${2:-/tmp/ov_cache_types_models}"
TEST_BINARY="${BUILD_DIR}/bin/tests_continuous_batching"

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
echo "Running GetCacheTypesRealModel tests ..."
echo "  TEST_MODELS_BASE_DIR = ${MODELS_BASE_DIR}"
echo "  CACHE_TYPES_CSV      = ${CSV_FILE}"
echo ""
export TEST_MODELS_BASE_DIR="${MODELS_BASE_DIR}"
export CACHE_TYPES_CSV="${CSV_FILE}"
"${TEST_BINARY}" --gtest_filter="*GetCacheTypesRealModel*"
