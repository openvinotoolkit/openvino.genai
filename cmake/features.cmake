# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

option(ENABLE_PYTHON "Enable Python API build" ON)
option(ENABLE_GIL_PYTHON_API "Build Python API with Global Interpreter Lock" ON)
option(ENABLE_JS "Enable JS API build" OFF)
option(ENABLE_SAMPLES "Enable samples build" ON)
option(ENABLE_TESTS "Enable tests build" ON)
option(ENABLE_TOOLS "Enable tools build" ON)
option(ENABLE_GGUF "Enable support for GGUF format" ON)
option(ENABLE_XGRAMMAR "Enable support for structured output generation with xgrammar backend" ON)

# Disable building samples for NPM package
if(CPACK_GENERATOR STREQUAL "NPM")
    set(ENABLE_SAMPLES OFF)
    set(ENABLE_PYTHON OFF)
    set(ENABLE_JS ON)
else()
    set(ENABLE_JS OFF)
endif()
