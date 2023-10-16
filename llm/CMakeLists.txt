# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.10)
project(llm)
add_subdirectory(thirdparty/llama.cpp)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
add_executable(llm llm.cpp)
target_link_libraries(llm PRIVATE openvino::runtime llama common)
set_target_properties(llm PROPERTIES CXX_STANDARD 11)
set_target_properties(llm PROPERTIES CXX_STANDARD_REQUIRED ON)
if(MSVC)
    target_compile_options(llama PRIVATE /wd4146 /wd4305)  # Disable the warnings from llama.cpp
    target_compile_options(common PRIVATE /wd4267 /wd4715 /wd4996)  # Disable the warnings from llama.cpp
    target_compile_options(
        llm
        PRIVATE
        /Wall  # Display all warnings
        /wd4245 /wd4267 /wd4820 /wd4514 /wd4996  # Disable the warnings from llama.cpp
        /wd4710 /wd4711  # Disable the inline warnings
        /EHsc  # Enable standard C++ stack unwinding, assume functions with extern "C" never throw
    )
else()
    target_compile_options(llm PRIVATE -Wall)  # Display all warnings
endif()
