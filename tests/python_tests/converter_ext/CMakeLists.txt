# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(converter_test_ext CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

set(PROD_PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/python")

pybind11_add_module(converter_test_ext
    "${CMAKE_CURRENT_SOURCE_DIR}/converter_test_ext.cpp"
    "${PROD_PYTHON_DIR}/py_any_converter.cpp"
)
target_include_directories(converter_test_ext PRIVATE "${PROD_PYTHON_DIR}")
target_link_libraries(converter_test_ext PRIVATE openvino::runtime)
