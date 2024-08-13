# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

option(BUILD_TOKENIZERS "Build OpenVINO Tokenizers together with OpenVINO GenAI" ON)

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
    if(TARGET core_tokenizers)
        set_target_properties(core_tokenizers PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
            LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
            RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
        )
    else()
        # Prebuilt dependencies
        if(WIN32)
            set(extra_libs "${CMAKE_BINARY_DIR}/_deps/fast_tokenizer-src/lib/core_tokenizers.dll"
                        "${CMAKE_BINARY_DIR}/_deps/fast_tokenizer-src/third_party/lib/icudt70.dll"
                        "${CMAKE_BINARY_DIR}/_deps/fast_tokenizer-src/third_party/lib/icuuc70.dll")
        elseif(LINUX)
            set(extra_libs "${CMAKE_BINARY_DIR}/_deps/fast_tokenizer-src/lib/libcore_tokenizers.so")
        elseif(APPLE)
            set(extra_libs "${CMAKE_BINARY_DIR}/_deps/fast_tokenizer-src/lib/libcore_tokenizers.dylib")
        endif()
        add_custom_command(OUTPUT "${extra_libs}"
            COMMAND "${CMAKE_COMMAND}" -E copy "${extra_libs}" "${CMAKE_BINARY_DIR}/openvino_genai/"
            DEPENDS openvino_tokenizers)
    endif()
endif()
