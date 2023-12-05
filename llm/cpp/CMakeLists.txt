# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

add_executable(llm llm.cpp)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(llm PRIVATE openvino::runtime)
set_target_properties(llm PROPERTIES CXX_STANDARD 17)
set_target_properties(llm PROPERTIES CXX_STANDARD_REQUIRED ON)
if(MSVC)
    target_compile_options(
        llm PRIVATE
        /Wall  # Display all warnings
        /wd4820  # Disable the padding addeddition warning after data members
        /wd4710 /wd4711  # Disable the inline warnings
        /EHsc  # Enable standard C++ stack unwinding, assume functions with extern "C" never throw
    )
else()
    target_compile_options(llm PRIVATE -Wall)  # Display all warnings
endif()
