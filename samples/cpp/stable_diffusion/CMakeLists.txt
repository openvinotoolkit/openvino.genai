# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# dependencies

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.1.1.tar.gz
    URL_HASH SHA256=523175f792eb0ff04f9e653c90746c12655f10cb70f1d5e6d6d9491420298a08)

FetchContent_MakeAvailable(cxxopts)

# create executable

add_executable(stable_diffusion
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imwrite.cpp)

target_include_directories(stable_diffusion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(stable_diffusion PRIVATE openvino::genai cxxopts::cxxopts)
