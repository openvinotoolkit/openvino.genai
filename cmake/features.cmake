# Copyright (C) 2018-2026 Intel Corporation
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
option(ENABLE_LTO "Enable Link Time Optimization" OFF)

# When building without OpenVINODeveloperPackage, verify IPO/LTO support
if(ENABLE_LTO AND NOT OpenVINODeveloperPackage_FOUND)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED
                        OUTPUT OUTPUT_MESSAGE
                        LANGUAGES C CXX)
    if(NOT IPO_SUPPORTED)
        set(ENABLE_LTO OFF CACHE BOOL "Enable link-time optimization" FORCE)
        message(WARNING "ENABLE_LTO=ON was requested, but IPO / LTO is not supported: ${OUTPUT_MESSAGE}")
    endif()
endif()

# Disable building samples for NPM package
if(CPACK_GENERATOR STREQUAL "NPM")
    set(ENABLE_SAMPLES OFF)
    set(ENABLE_PYTHON OFF)
    set(ENABLE_JS ON)
else()
    set(ENABLE_JS OFF)
endif()
