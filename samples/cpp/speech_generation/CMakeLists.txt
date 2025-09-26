# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

include(FetchContent)

if(NOT TARGET dr_libs)
    FetchContent_Declare(dr_libs
        URL https://github.com/mackron/dr_libs/archive/da35f9d6c7374a95353fd1df1d394d44ab66cf01.tar.gz
        URL_HASH SHA256=2704d347f480ca1bc92233fb01747e4550cc8031735b6ea62ca9990ebb8851ae)
    FetchContent_MakeAvailable(dr_libs)
endif()

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# create main sample executable

ov_add_sample(
        NAME text2speech
        SOURCES text2speech.cpp audio_utils.cpp
        DEPENDENCIES openvino::genai
        INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${dr_libs_SOURCE_DIR}>")

target_compile_features(text2speech PRIVATE cxx_std_11)
