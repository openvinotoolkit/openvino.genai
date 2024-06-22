# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.12.0
)

FetchContent_GetProperties(pybind11)
# search for FindPython3.cmake instead of legacy modules
set(PYBIND11_FINDPYTHON ON)
# the following two calls are required for cross-compilation
if(OpenVINODeveloperPackage_DIR)
    ov_find_python3(REQUIRED)
    ov_detect_python_module_extension()
else()
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    else()
        find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    endif()
endif()
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

pybind11_add_module(py_continuous_batching python.cpp)

target_link_libraries(py_continuous_batching PRIVATE openvino::continuous_batching)
