# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"
        ${OpenVINO_DIR}
    NO_CMAKE_FIND_ROOT_PATH
)

function(add_sample_executable target_name)
    add_executable(${target_name} ${target_name}.c text2speech_sample_utils.c)
    set_source_files_properties(${target_name}.c text2speech_sample_utils.c PROPERTIES LANGUAGE C)
    target_link_libraries(${target_name} PRIVATE openvino::genai::c)
    set_target_properties(${target_name} PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH ON)
    install(TARGETS ${target_name}
            RUNTIME DESTINATION samples_bin/
            COMPONENT samples_bin
            EXCLUDE_FROM_ALL)
endfunction()

add_sample_executable(text2speech_sample_c)
