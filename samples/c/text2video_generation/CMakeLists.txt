# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

# Text-to-Video Pipeline Sample
add_executable(text2video_sample_c text2video_sample_c.c)

# Specifies that the source file should be compiled as a C source file
set_source_files_properties(text2video_sample_c.c PROPERTIES LANGUAGE C)
target_link_libraries(text2video_sample_c PRIVATE openvino::genai::c)

set_target_properties(text2video_sample_c PROPERTIES
    # Ensure out-of-box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

# Install
install(TARGETS text2video_sample_c
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)