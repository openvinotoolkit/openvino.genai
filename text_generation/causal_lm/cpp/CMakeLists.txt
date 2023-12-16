# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)
project(causal_lm)

# Build user_ov_extensions
list(APPEND CUSTOM_OPERATIONS tokenizer)
add_subdirectory(../../../thirdparty/openvino_contrib/modules/custom_operations/ "${CMAKE_CURRENT_BINARY_DIR}/custom_operations/")

add_executable(causal_lm causal_lm.cpp)
target_compile_definitions(causal_lm PRIVATE USER_OV_EXTENSIONS_PATH=\"$<TARGET_FILE:user_ov_extensions>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(causal_lm PRIVATE openvino::runtime user_ov_extensions)
set_target_properties(causal_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(causal_lm PROPERTIES CXX_STANDARD_REQUIRED ON)
if(MSVC)
    target_compile_options(
        causal_lm PRIVATE
        /Wall  # Display all warnings
        /wd4710 /wd4711  # Disable the inline warnings
        /EHsc  # Enable standard C++ stack unwinding, assume functions with extern "C" never throw
    )
else()
    target_compile_options(causal_lm PRIVATE -Wall)  # Display all warnings
endif()

add_executable(beam_search_causal_lm beam_search_causal_lm.cpp)
target_compile_definitions(beam_search_causal_lm PRIVATE USER_OV_EXTENSIONS_PATH=\"$<TARGET_FILE:user_ov_extensions>\")
target_include_directories(beam_search_causal_lm PRIVATE ./)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(beam_search_causal_lm PRIVATE openvino::runtime user_ov_extensions)
set_target_properties(beam_search_causal_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(beam_search_causal_lm PROPERTIES CXX_STANDARD_REQUIRED ON)
if(MSVC)
    target_compile_options(
        beam_search_causal_lm PRIVATE
        /Wall  # Display all warnings
        /wd4626 /wd5027  # Disable the implicit definition of assignment operator as deleted warings
        /wd4710 /wd4711  # Disable the inline warnings
        /wd4820  # Disable the padding addition warning after data members
        /EHsc  # Enable standard C++ stack unwinding, assume functions with extern "C" never throw
    )
else()
    target_compile_options(beam_search_causal_lm PRIVATE -Wall)  # Display all warnings
endif()
