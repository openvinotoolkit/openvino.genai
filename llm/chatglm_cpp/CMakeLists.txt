# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

add_executable(chatglm chatglm.cpp)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(chatglm PRIVATE openvino::runtime)
set_target_properties(chatglm PROPERTIES CXX_STANDARD 17)
set_target_properties(chatglm PROPERTIES CXX_STANDARD_REQUIRED ON)
if(MSVC)
    target_compile_options(
        chatglm PRIVATE
        /Wall  # Display all warnings
        /wd4710 /wd4711  # Disable the inline warnings
        /EHsc  # Enable standard C++ stack unwinding, assume functions with extern "C" never throw
    )
else()
    target_compile_options(chatglm PRIVATE -Wall)  # Display all warnings
endif()

