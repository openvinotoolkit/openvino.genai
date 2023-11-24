# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)  # The requirement is from ../../../thirdparty/openvino_contrib/modules/custom_operations/CMakeLists.txt
project(LLaMA)

list(APPEND CUSTOM_OPERATIONS tokenizer)
add_subdirectory(../../../thirdparty/openvino_contrib/modules/custom_operations/ "${CMAKE_CURRENT_BINARY_DIR}/custom_operations/")  # Build user_ov_extensions

add_executable(llama llama.cpp)
target_compile_definitions(llama PRIVATE USER_OV_EXTENSIONS_PATH=\"$<TARGET_FILE:user_ov_extensions>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(llama PRIVATE openvino::runtime)
set_target_properties(llama PROPERTIES CXX_STANDARD 17)
set_target_properties(llama PROPERTIES CXX_STANDARD_REQUIRED ON)
if(MSVC)
    target_compile_options(
        llama PRIVATE
        /Wall  # Display all warnings
        /wd4710 /wd4711  # Disable the inline warnings
        /EHsc  # Enable standard C++ stack unwinding, assume functions with extern "C" never throw
    )
else()
    target_compile_options(llama PRIVATE -Wall)  # Display all warnings
    target_compile_options(sentencepiece-static PRIVATE -Wno-stringop-overflow)  # Disable the warning from user_ov_extensions
endif()
