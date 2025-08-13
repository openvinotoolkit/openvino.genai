# Copyright (C) 2025 Intel Corporation
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
            DEPENDENCIES openvino::genai
            SOURCES ${target_name}.cpp)

    set_target_properties(${target_name} PROPERTIES
        COMPILE_PDB_NAME ${target_name})
endfunction()

set(SAMPLE_LIST text_embeddings text_rerank)

foreach(sample ${SAMPLE_LIST})
    add_sample_executable(${sample})
endforeach()


# benchmark_genai
include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
