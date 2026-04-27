#!/bin/bash
# PR 1 verification driver: rebuild and run the MODEL_PROPERTIES
# Python smoke test against the Qwen3-VL model from the previous session.

set -e

BUILD_DIR="/home/dkal/openvino.genai/build"
VENV_DIR="/home/dkal/openvino.genai/.venv"
MODEL_DIR="$HOME/model_server/demos/common/export_models/models/vlm_models_with_export_models/Qwen/Qwen3-VL-4B-Instruct_fp32"
SCRIPT="/home/dkal/openvino.genai/pr1_check.py"
JOBS=32

echo "=== Rebuilding ==="
cmake "$BUILD_DIR" -B "$BUILD_DIR" > /dev/null
cmake --build "$BUILD_DIR" --parallel "$JOBS" > /dev/null

echo ""
echo "=== Running PR 1 smoke test ==="
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$BUILD_DIR:$PYTHONPATH"
python "$SCRIPT" "$MODEL_DIR"
echo ""
echo "=== Done ==="
