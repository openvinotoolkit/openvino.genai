# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

option(BUILD_TOKENIZERS "Build OpenVINO Tokenizers together with OpenVINO GenAI" ON)

option(BUILD_LLAMA_CPP "Build llama.cpp with OpenVINO backend for dynamic GGUF loading" ON)


if(BUILD_TOKENIZERS)
    add_subdirectory(./openvino_tokenizers/ "${CMAKE_BINARY_DIR}/openvino_tokenizers/")
    # Put binaries to a single dir to mimic package structure.
    set_target_properties(openvino_tokenizers PROPERTIES
        # Generator expressions to disable appending a per-configuration subdirectory (Release, Debug).
        # ARCHIVE_OUTPUT is irrelevant. It's here just to keep all the artifacts in one place.
        ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
        LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
        RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
    )
endif()

option(BUILD_LLAMA_CPP "Build llama.cpp with OpenVINO backend for dynamic GGUF loading" OFF)

if(BUILD_LLAMA_CPP)
    set(GGML_OPENVINO ON CACHE BOOL "Enable OpenVINO backend in llama.cpp" FORCE)

    # Save and restore to avoid polluting global scope
    set(_orig_build_shared_libs ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS ON CACHE BOOL "Build llama.cpp as shared library" FORCE)

    add_subdirectory(llama.cpp "${CMAKE_BINARY_DIR}/llama_cpp/")

    set(BUILD_SHARED_LIBS ${_orig_build_shared_libs} CACHE BOOL "" FORCE)
endif()
