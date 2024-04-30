# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)
project(causal_lm)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

add_subdirectory(../../../thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")

# start of dependencies

include(FetchContent)

FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.1.1.tar.gz
    URL_HASH SHA256=523175f792eb0ff04f9e653c90746c12655f10cb70f1d5e6d6d9491420298a08)

FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
    URL_HASH SHA256=a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d)

FetchContent_MakeAvailable(cxxopts)
FetchContent_MakeAvailable(nlohmann_json)

# end of dependencies

set(TARGET_NAME greedy_sample)
add_executable(${TARGET_NAME} greedy_causal_lm.cpp)
target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)

set(TARGET_NAME beam_search_sample)
add_executable(${TARGET_NAME} beam_search_causal_lm.cpp)
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
