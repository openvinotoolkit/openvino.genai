#!/usr/bin/env bash
# Runs the streaming ASR sample on a Voxtral Realtime split export with the PagedAttention backend.
#
# Usage: ./run_voxtral_pa_streaming.sh [MODEL_DIR] [extra asr_streaming.py arguments]
#   ./run_voxtral_pa_streaming.sh
#   ./run_voxtral_pa_streaming.sh /path/to/model_dir --streams 2 --wav /path/to/audio.wav
#
# MODEL_DIR must hold audio_encoder/, text_embeddings/ and text_decoder/ next to config.json,
# generation_config.json, preprocessor_config.json and tekken.json.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-voxtral-test}"
PYTHON="$REPO_DIR/.venv/bin/python"
DEFAULT_MODEL_DIR="$HOME/models/OpenVINO/voxtral-stateful"
WAV_FILE="$REPO_DIR/SayLargestAnimal.wav"
CHUNK_SIZE_SEC=2
NUM_STREAMS=4

MODEL_DIR="$DEFAULT_MODEL_DIR"
if [[ $# -gt 0 && "$1" != -* ]]; then
    MODEL_DIR="$1"
    shift
fi

for required in "$PYTHON" "$BUILD_DIR/openvino_genai" "$MODEL_DIR/text_decoder/openvino_model.xml" "$MODEL_DIR/config.json"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing: $required" >&2
        exit 1
    fi
done

OPENVINO_LIBS="$("$PYTHON" -c 'import os, openvino; print(os.path.join(os.path.dirname(openvino.__file__), "libs"))')"
export PYTHONPATH="$BUILD_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BUILD_DIR/openvino_genai:$OPENVINO_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Arguments given on the command line come last, so they override the defaults above.
exec "$PYTHON" "$REPO_DIR/samples/python/automatic_speech_recognition/asr_streaming.py" "$MODEL_DIR" \
    --wav "$WAV_FILE" --chunk-size-sec "$CHUNK_SIZE_SEC" --attention-backend PA --streams "$NUM_STREAMS" "$@"
