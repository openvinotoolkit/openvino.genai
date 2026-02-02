# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# yaml-cpp dependency management
# This module provides yaml-cpp either from system installation or via FetchContent
#

include(FetchContent)

# Check if yaml-cpp target already exists (from parent project)
if(TARGET yaml-cpp OR TARGET yaml-cpp::yaml-cpp)
    message(STATUS "yaml-cpp target already available")
    if(TARGET yaml-cpp::yaml-cpp)
        set(YAML_CPP_TARGET yaml-cpp::yaml-cpp CACHE STRING "yaml-cpp target name" FORCE)
    else()
        set(YAML_CPP_TARGET yaml-cpp CACHE STRING "yaml-cpp target name" FORCE)
    endif()
    return()
endif()

# Option to force FetchContent even if system yaml-cpp is available
# This is useful for wheel builds to ensure yaml-cpp.dll is included
option(YAML_CPP_FORCE_FETCH "Force FetchContent for yaml-cpp even if system version is available" OFF)

# For wheel builds, always use FetchContent to ensure DLL is included
if(DEFINED PY_BUILD_CMAKE_PACKAGE_NAME)
    set(YAML_CPP_FORCE_FETCH ON)
    message(STATUS "Wheel build detected, forcing FetchContent for yaml-cpp")
endif()

# Try to find yaml-cpp from system (unless forced to fetch)
if(NOT YAML_CPP_FORCE_FETCH)
    find_package(yaml-cpp QUIET)
endif()

if(yaml-cpp_FOUND AND NOT YAML_CPP_FORCE_FETCH)
    message(STATUS "Found yaml-cpp from system: ${yaml-cpp_VERSION}")
    if(TARGET yaml-cpp::yaml-cpp)
        set(YAML_CPP_TARGET yaml-cpp::yaml-cpp CACHE STRING "yaml-cpp target name" FORCE)
    else()
        set(YAML_CPP_TARGET ${YAML_CPP_LIBRARIES} CACHE STRING "yaml-cpp target name" FORCE)
    endif()
    set(YAML_CPP_INCLUDE_DIRS ${YAML_CPP_INCLUDE_DIR} CACHE PATH "yaml-cpp include directories" FORCE)
else()
    message(STATUS "yaml-cpp not found or FetchContent forced, fetching from GitHub...")

    # Disable yaml-cpp tests and tools
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

    # yaml-cpp 0.8.0 is the latest release (as of Jan 2026)
    # Use PATCH_COMMAND to fix cmake_minimum_required for CMake 3.27+ compatibility
    FetchContent_Declare(yaml-cpp
        URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
        URL_HASH SHA256=fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16
        PATCH_COMMAND ${CMAKE_COMMAND} -E echo "Patching yaml-cpp CMakeLists.txt for CMake 3.27+ compatibility"
        COMMAND ${CMAKE_COMMAND} -DFILE=<SOURCE_DIR>/CMakeLists.txt
                -P ${CMAKE_CURRENT_LIST_DIR}/patch_cmake_minimum_required.cmake
    )
    FetchContent_MakeAvailable(yaml-cpp)

    set(YAML_CPP_TARGET yaml-cpp CACHE STRING "yaml-cpp target name" FORCE)
    set(YAML_CPP_INCLUDE_DIRS ${yaml-cpp_SOURCE_DIR}/include CACHE PATH "yaml-cpp include directories" FORCE)
    set(YAML_CPP_FETCHED TRUE CACHE BOOL "yaml-cpp was fetched from GitHub" FORCE)

    message(STATUS "yaml-cpp fetched and built from source (version 0.8.0)")
endif()
