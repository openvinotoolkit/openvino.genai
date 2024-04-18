# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# start of dependencies

add_subdirectory(../../../../../thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")

include(FetchContent)

FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
    URL_HASH SHA256=a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d)

FetchContent_MakeAvailable(nlohmann_json)

# end of dependencies

set(TARGET_NAME openvino_continuous_batching)

add_library(${TARGET_NAME} STATIC
    src/tokenizer.cpp
    src/paged_attention.cpp
    src/continuous_batching_pipeline.cpp
    src/paged_attention_transformations.cpp)

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
                                          PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
