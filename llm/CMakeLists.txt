# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)  # The requirement comes from thirdparty/openvino_contrib/modules/custom_operations/CMakeLists.txt
project(llm)

list(APPEND CUSTOM_OPERATIONS tokenizer)
add_subdirectory(thirdparty/openvino_contrib/modules/custom_operations/)  # Build user_ov_extensions

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
add_executable(llm llm.cpp)
target_compile_definitions(llm PRIVATE USER_OV_EXTENSIONS_PATH=\"$<TARGET_FILE:user_ov_extensions>\")  # Define path to user_ov_extensions
target_link_libraries(llm PRIVATE openvino::runtime)
set_target_properties(llm PROPERTIES CXX_STANDARD 11)
set_target_properties(llm PROPERTIES CXX_STANDARD_REQUIRED ON)
if(MSVC)
    target_compile_options(
        llm
        PRIVATE
        /Wall  # Display all warnings
        /wd4710 /wd4711  # Disable the inline warnings
    )
else()
    target_compile_options(llm PRIVATE -Wall)  # Display all warnings
endif()
