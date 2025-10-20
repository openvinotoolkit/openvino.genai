# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_MakeAvailable(stb)

add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${stb_SOURCE_DIR})

# VLM Pipeline Sample
add_executable(vlm_pipeline_c vlm_pipeline.c load_image.c)

# Specifies that the source file should be compiled as a C source file
set_source_files_properties(vlm_pipeline.c load_image.c PROPERTIES LANGUAGE C)

target_include_directories(vlm_pipeline_c PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/c/include
    ${stb_SOURCE_DIR}
)

target_link_libraries(vlm_pipeline_c PRIVATE openvino::genai::c stb_image)
if(UNIX AND NOT APPLE)
    target_link_libraries(vlm_pipeline_c PRIVATE m)
endif()

set_target_properties(vlm_pipeline_c PROPERTIES
    # Ensure out-of-box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

# Install
install(TARGETS vlm_pipeline_c
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

