# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

file(DOWNLOAD https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h ${CMAKE_BINARY_DIR}/stb_image.h
     EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(indicators
    URL https://github.com/p-ranav/indicators/archive/refs/tags/v2.3.tar.gz
    URL_HASH SHA256=70da7a693ff7a6a283850ab6d62acf628eea17d386488af8918576d0760aef7b)
FetchContent_MakeAvailable(indicators)

# create main sample executable

add_executable(text2video text2video.cpp)

target_include_directories(text2video PRIVATE ${CMAKE_BINARY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpp/src")
target_link_libraries(text2video PRIVATE openvino::genai indicators::indicators)

set_target_properties(text2video PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS text2video
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
