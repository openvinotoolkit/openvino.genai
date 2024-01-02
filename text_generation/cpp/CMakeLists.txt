# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)
project(text_generation)

# Build user_ov_extensions
list(APPEND CUSTOM_OPERATIONS tokenizer)
add_subdirectory(../../thirdparty/openvino_contrib/modules/custom_operations/ "${CMAKE_CURRENT_BINARY_DIR}/custom_operations/")

add_executable(decoder_greedy decoder_greedy.cpp)
target_compile_definitions(decoder_greedy PRIVATE USER_OV_EXTENSIONS_PATH=\"$<TARGET_FILE:user_ov_extensions>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(decoder_greedy PRIVATE openvino::runtime)
set_target_properties(decoder_greedy PROPERTIES CXX_STANDARD 17)
set_target_properties(decoder_greedy PROPERTIES CXX_STANDARD_REQUIRED ON)

add_executable(decoder_beam decoder_beam.cpp)
target_compile_definitions(decoder_beam PRIVATE USER_OV_EXTENSIONS_PATH=\"$<TARGET_FILE:user_ov_extensions>\")
target_include_directories(decoder_beam PRIVATE ./)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(decoder_beam PRIVATE openvino::runtime)
set_target_properties(decoder_beam PROPERTIES CXX_STANDARD 17)
set_target_properties(decoder_beam PROPERTIES CXX_STANDARD_REQUIRED ON)
