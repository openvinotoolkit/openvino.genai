# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)
project(causal_lm)

add_subdirectory(../../../thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")

add_executable(greedy_causal_lm greedy_causal_lm.cpp)
target_compile_definitions(greedy_causal_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(greedy_causal_lm PRIVATE openvino::runtime)
set_target_properties(greedy_causal_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(greedy_causal_lm PROPERTIES CXX_STANDARD_REQUIRED ON)

add_executable(beam_search_causal_lm beam_search_causal_lm.cpp)
target_compile_definitions(beam_search_causal_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_include_directories(beam_search_causal_lm PRIVATE ./)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(beam_search_causal_lm PRIVATE openvino::runtime)
set_target_properties(beam_search_causal_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(beam_search_causal_lm PROPERTIES CXX_STANDARD_REQUIRED ON)

add_executable(speculative_decoding_lm speculative_decoding_lm.cpp)
target_compile_definitions(speculative_decoding_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_include_directories(speculative_decoding_lm PRIVATE ./)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(speculative_decoding_lm PRIVATE openvino::runtime)
set_target_properties(speculative_decoding_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(speculative_decoding_lm PROPERTIES CXX_STANDARD_REQUIRED ON)

add_executable(prompt_lookup_decoding_lm prompt_lookup_decoding_lm.cpp)
target_compile_definitions(prompt_lookup_decoding_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_include_directories(prompt_lookup_decoding_lm PRIVATE ./)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(prompt_lookup_decoding_lm PRIVATE openvino::runtime)
set_target_properties(prompt_lookup_decoding_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(prompt_lookup_decoding_lm PROPERTIES CXX_STANDARD_REQUIRED ON)
find_package(TBB COMPONENTS tbb)
target_link_libraries(prompt_lookup_decoding_lm PRIVATE TBB::tbb)
