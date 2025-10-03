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

ov_add_sample(
        NAME visual_language_chat
        SOURCES visual_language_chat.cpp load_image.cpp
        DEPENDENCIES openvino::genai
        INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR})

# create encrypted model sample executable

ov_add_sample(
        NAME encrypted_model_vlm
        SOURCES encrypted_model_vlm.cpp load_image.cpp
        DEPENDENCIES openvino::genai
        INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR})

# create benchmark executable
ov_add_sample(
        NAME benchmark_vlm
        SOURCES benchmark_vlm.cpp load_image.cpp ../text_generation/read_prompt_from_file.cpp
        DEPENDENCIES cxxopts::cxxopts openvino::genai
        INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR})
