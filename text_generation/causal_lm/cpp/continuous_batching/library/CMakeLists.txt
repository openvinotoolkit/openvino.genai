# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# start of dependencies

add_subdirectory(../../../../../thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")

include(FetchContent)

FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)

FetchContent_MakeAvailable(nlohmann_json)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# end of dependencies

set(TARGET_NAME openvino_continuous_batching)

add_library(${TARGET_NAME} STATIC
    src/tokenizer.cpp
    src/paged_attention.cpp
    src/generation_config.cpp
    src/continuous_batching_pipeline.cpp
    src/paged_attention_transformations.cpp)

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
                                          PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime nlohmann_json::nlohmann_json)

# gtest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)


set(TEST_TARGET_NAME "tests_continuous_batching")
add_executable(${TEST_TARGET_NAME} "src/tests/scheduler.cpp")
target_link_libraries(${TEST_TARGET_NAME} PUBLIC ${TARGET_NAME} openvino::runtime gtest_main)
target_include_directories(${TEST_TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/"
                                          PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
