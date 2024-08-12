# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

file(GLOB SRC *.cpp)

add_executable(minicpmv2  ${SRC})
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(minicpmv2 PRIVATE openvino::runtime)
set_target_properties(minicpmv2 PROPERTIES CXX_STANDARD 17)
set_target_properties(minicpmv2 PROPERTIES CXX_STANDARD_REQUIRED ON)
if(MSVC)
    target_compile_options(
        minicpmv2 PRIVATE
        /Wall  # Display all warnings
        /wd4710 /wd4711  # Disable the inline warnings
        /EHsc  # Enable standard C++ stack unwinding, assume functions with extern "C" never throw
    )
else()
    target_compile_options(minicpmv2 PRIVATE -Wall)  # Display all warnings
endif()

