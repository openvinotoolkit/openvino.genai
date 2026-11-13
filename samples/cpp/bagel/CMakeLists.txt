# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h
    ${CMAKE_BINARY_DIR}/stb_image.h
    EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image_resize2.h
    ${CMAKE_BINARY_DIR}/stb_image_resize2.h
    EXPECTED_HASH MD5=9293a158ddf9e44dd186c446372784d8)

add_executable(bagel bagel.cpp ../visual_language_chat/load_image.cpp)

target_include_directories(bagel PRIVATE
    ${CMAKE_BINARY_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/../visual_language_chat")
target_link_libraries(bagel PRIVATE openvino::genai)

set_target_properties(bagel PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS bagel
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
