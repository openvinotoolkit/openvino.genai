# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

# stb_image.h / stb_image_resize2.h are fetched by the visual_language_chat samples into CMAKE_BINARY_DIR
add_executable(classification classification.cpp ../visual_language_chat/load_image.cpp)
target_include_directories(classification PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(classification PRIVATE openvino::genai)

set_target_properties(classification PROPERTIES
    COMPILE_PDB_NAME classification
    # Ensure out-of-box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
install(TARGETS classification
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
