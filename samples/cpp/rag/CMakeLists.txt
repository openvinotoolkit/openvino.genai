# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

function(add_sample_executable target_name)
    add_executable(${target_name} ${target_name}.cpp)
    target_link_libraries(${target_name} PRIVATE openvino::genai)
    set_target_properties(${target_name} PROPERTIES
        COMPILE_PDB_NAME ${target_name}
        # Ensure out-of-box LC_RPATH on macOS with SIP
        INSTALL_RPATH_USE_LINK_PATH ON)
    install(TARGETS ${target_name}
            RUNTIME DESTINATION samples_bin/
            COMPONENT samples_bin
            EXCLUDE_FROM_ALL)
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
