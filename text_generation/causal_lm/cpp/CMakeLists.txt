# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)
project(causal_lm)

set(JINJA2CPP_DEPS_MODE internal)


add_subdirectory(../../../thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")
add_subdirectory(../../../thirdparty/nlohmann_json/ "${CMAKE_CURRENT_BINARY_DIR}/nlohmann_json/")

# todo: remove hardcodes and make submodule work
include_directories($ENV{HOME}/opt/jinja2cpp/include)
# add_subdirectory(../../../thirdparty/Jinja2Cpp/ "${CMAKE_CURRENT_BINARY_DIR}/Jinja2Cpp/")
# include_directories(../../../thirdparty/inja/include/Jinja2Cpp)

set(TARGET_NAME greedy_causal_lm)
add_executable(${TARGET_NAME} greedy_causal_lm.cpp)
# todo: remove hardcode
target_link_libraries(${TARGET_NAME} PRIVATE $ENV{HOME}/opt/jinja2cpp/lib/static/libjinja2cpp.a)
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)

set(TARGET_NAME beam_search_causal_lm)
add_executable(${TARGET_NAME} beam_search_causal_lm.cpp)
# todo: remove hardcode
target_link_libraries(${TARGET_NAME} PRIVATE $ENV{HOME}/opt/jinja2cpp/lib/static/libjinja2cpp.a)
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)

set(TARGET_NAME speculative_decoding_lm)
add_executable(${TARGET_NAME} speculative_decoding_lm.cpp)
# todo: remove hardcode
target_link_libraries(${TARGET_NAME} PRIVATE $ENV{HOME}/opt/jinja2cpp/lib/static/libjinja2cpp.a)
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)

set(TARGET_NAME generate_sample)
add_executable(${TARGET_NAME} generate_pipeline/generate_sample.cpp generate_pipeline/llm_tokenizer.cpp generate_pipeline/llm_pipeline.cpp)
# todo: remove hardcode
target_link_libraries(${TARGET_NAME} PRIVATE $ENV{HOME}/opt/jinja2cpp/lib/static/libjinja2cpp.a)
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
target_link_libraries(${TARGET_NAME} PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)

set(TARGET_NAME chat_sample)
add_executable(${TARGET_NAME} generate_pipeline/chat_sample.cpp generate_pipeline/llm_tokenizer.cpp generate_pipeline/llm_pipeline.cpp)
# todo: remove hardcode
target_link_libraries(${TARGET_NAME} PRIVATE $ENV{HOME}/opt/jinja2cpp/lib/static/libjinja2cpp.a)
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
target_link_libraries(${TARGET_NAME} PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)


set(TARGET_NAME prompt_lookup_decoding_lm)
add_executable(${TARGET_NAME} prompt_lookup_decoding_lm.cpp)
target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_include_directories(${TARGET_NAME} PRIVATE ./)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
