# Copyright (C) 2023-2025 Intel Corporation
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

# create main sample executable

add_executable(visual_language_chat visual_language_chat.cpp load_image.cpp)
target_include_directories(visual_language_chat PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(visual_language_chat PRIVATE openvino::genai)

set_target_properties(visual_language_chat PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS visual_language_chat
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create encrypted model sample executable

add_executable(encrypted_model_vlm encrypted_model_vlm.cpp load_image.cpp)
target_include_directories(encrypted_model_vlm PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(encrypted_model_vlm PRIVATE openvino::genai)

set_target_properties(encrypted_model_vlm PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS encrypted_model_vlm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create benchmark executable
add_executable(benchmark_vlm benchmark_vlm.cpp load_image.cpp ../text_generation/read_prompt_from_file.cpp)
target_include_directories(benchmark_vlm PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(benchmark_vlm PRIVATE openvino::genai cxxopts::cxxopts)
set_target_properties(benchmark_vlm PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS benchmark_vlm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
