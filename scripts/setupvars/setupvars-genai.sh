#!/bin/bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

abs_path () {
    script_path=$(eval echo "$1")
    directory=$(dirname "$script_path")
    builtin cd "$directory" >/dev/null 2>&1 || exit
    pwd -P
}

SCRIPT_DIR="$(abs_path "${BASH_SOURCE:-$0}")" >/dev/null 2>&1
INSTALLDIR="${SCRIPT_DIR}"

if [ ! -d "$INSTALLDIR/runtime" ]; then
    echo "[setupvars-genai.sh] ERROR: runtime directory not found at $INSTALLDIR/runtime"
    echo "[setupvars-genai.sh] Please source this script from the OpenVINO GenAI install directory"
    return 1 2>/dev/null || exit 1
fi

# Export OpenVINOGenAI_DIR for find_package(OpenVINOGenAI)
if [ -d "$INSTALLDIR/runtime/cmake" ]; then
    export OpenVINOGenAI_DIR=$INSTALLDIR/runtime/cmake
fi

# Detect architecture dynamically (intel64, aarch64, arm64, etc.)
system_type=$(/bin/ls "$INSTALLDIR/runtime/lib/")

GENAI_LIB_PATH=$INSTALLDIR/runtime/lib/$system_type

# Set library paths
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH=${GENAI_LIB_PATH}/Release:${GENAI_LIB_PATH}/Debug${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${GENAI_LIB_PATH}/Release:${GENAI_LIB_PATH}/Debug${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
else
    export LD_LIBRARY_PATH=${GENAI_LIB_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

# Set PYTHONPATH
if [ -d "$INSTALLDIR/python" ]; then
    export PYTHONPATH="$INSTALLDIR/python${PYTHONPATH:+:$PYTHONPATH}"
fi

unset system_type
unset GENAI_LIB_PATH

echo "[setupvars-genai.sh] OpenVINO GenAI environment initialized"
