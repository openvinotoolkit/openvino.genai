# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

function(add_sample_executable target_name)
    genai_add_c_sample_target(
            NAME ${target_name}
            ROOT ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES ${target_name}.c
            LINK_LIBRARIES cxxopts::cxxopts indicators::indicators
            INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR})
endfunction()

set (SAMPLE_LIST
    greedy_causal_lm_c
    chat_sample_c
    benchmark_genai_c)

foreach(sample IN LISTS SAMPLE_LIST)
    add_sample_executable(${sample})
endforeach()

