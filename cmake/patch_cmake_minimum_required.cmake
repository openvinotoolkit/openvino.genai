# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Patch script to update cmake_minimum_required for CMake 3.27+ compatibility
# Usage: cmake -DFILE=<path_to_CMakeLists.txt> -P patch_cmake_minimum_required.cmake

if(NOT DEFINED FILE)
    message(FATAL_ERROR "FILE variable must be defined")
endif()

if(NOT EXISTS "${FILE}")
    message(FATAL_ERROR "File does not exist: ${FILE}")
endif()

file(READ "${FILE}" content)

# Replace old cmake_minimum_required with version range syntax
# yaml-cpp 0.8.0 uses: cmake_minimum_required(VERSION 3.4)
# We change it to: cmake_minimum_required(VERSION 3.5...3.28)
string(REGEX REPLACE
    "cmake_minimum_required\\(VERSION[ \t]+([0-9]+\\.[0-9]+)\\)"
    "cmake_minimum_required(VERSION 3.5...3.28)"
    content "${content}")

file(WRITE "${FILE}" "${content}")
message(STATUS "Patched ${FILE} for CMake 3.27+ compatibility")
