# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

function(add_sample_executable target_name)
    add_executable(${target_name} ${target_name}.c)
    # Specifies that the source file should be compiled as a C source file
    set_source_files_properties(${target_name}.c PROPERTIES LANGUAGE C)
    target_link_libraries(${target_name} PRIVATE openvino::genai::c)
    set_target_properties(${target_name} PROPERTIES
        # Ensure out-of-box LC_RPATH on macOS with SIP
        INSTALL_RPATH_USE_LINK_PATH ON)
    install(TARGETS ${target_name}
            RUNTIME DESTINATION samples_bin/
            COMPONENT samples_bin
            EXCLUDE_FROM_ALL)
endfunction()

set (SAMPLE_LIST
    greedy_causal_lm_c
    chat_sample_c
    benchmark_genai_c)

foreach(sample IN LISTS SAMPLE_LIST)
    add_sample_executable(${sample})
endforeach()

