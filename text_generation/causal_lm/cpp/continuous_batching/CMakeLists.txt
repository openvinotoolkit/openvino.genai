# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

project(continuous_batching)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CMakeDependentOption)

option(ENABLE_APPS "Enable C++ apps" ON)
option(ENABLE_PYTHON "Enable Python API" ON)

add_subdirectory(library)

if(ENABLE_APPS)
    add_subdirectory(apps)
endif()

if(ENABLE_PYTHON)
    add_subdirectory(python)
endif()
