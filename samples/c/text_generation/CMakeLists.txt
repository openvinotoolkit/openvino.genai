# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

function(add_sample_executable target_name)
    ov_add_sample(
            NAME ${target_name}
            SOURCES ${target_name}.c
            DEPENDENCIES openvino::genai::c)
    set_source_files_properties(${target_name}.c PROPERTIES LANGUAGE C)
endfunction()

set (SAMPLE_LIST
    greedy_causal_lm_c
    chat_sample_c
    benchmark_genai_c)

foreach(sample IN LISTS SAMPLE_LIST)
    add_sample_executable(${sample})
endforeach()

