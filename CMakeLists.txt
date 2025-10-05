# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.23.0)  # The requirement comes from Jinja2Cpp

# Multi config generators such as Visual Studio ignore CMAKE_BUILD_TYPE. Multi config generators are configured with
# CMAKE_CONFIGURATION_TYPES, but limiting options in it completely removes such build options
get_property(GENERATOR_IS_MULTI_CONFIG_VAR GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(CMAKE_GENERATOR STREQUAL "Ninja Multi-Config")
    # 'Ninja Multi-Config' specific, see:
    # https://cmake.org/cmake/help/latest/variable/CMAKE_DEFAULT_BUILD_TYPE.html
    set(CMAKE_DEFAULT_BUILD_TYPE "Release" CACHE STRING "CMake default build type")
elseif(NOT GENERATOR_IS_MULTI_CONFIG_VAR AND NOT DEFINED CMAKE_BUILD_TYPE)
    message(STATUS "CMAKE_BUILD_TYPE is not defined, 'Release' will be used")
    # Setting CMAKE_BUILD_TYPE as CACHE must go before project(). Otherwise project() sets its value and set() doesn't take an effect
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ...")
endif()

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

if(UNIX AND NOT (APPLE OR ANDROID OR CYGWIN))
    set(LINUX ON)
endif()

project(OpenVINOGenAI
        VERSION 2025.4.0.0
        DESCRIPTION "OpenVINO GenAI"
        HOMEPAGE_URL "https://github.com/openvinotoolkit/openvino.genai"
        LANGUAGES CXX C)

if(NOT DEFINED Python3_FIND_VIRTUALENV)
    set(Python3_FIND_VIRTUALENV FIRST)
endif()

# Looking for OpenVINO in the python distribution. It doesn't work for cross-compiling build
if(NOT CMAKE_CROSSCOMPILING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
            OUTPUT_VARIABLE OpenVINO_DIR_PY
            ERROR_QUIET
        )
    endif()
endif()

# Find OpenVINODeveloperPackage first to compile with SDL flags
set(OV_COMPATIBILITY_VERSION ${OpenVINOGenAI_VERSION_MAJOR}.${OpenVINOGenAI_VERSION_MINOR}.${OpenVINOGenAI_VERSION_PATCH})
find_package(OpenVINODeveloperPackage ${OV_COMPATIBILITY_VERSION} QUIET
             COMPONENTS Runtime Threading
             PATHS "${OpenVINO_DIR}")
if(NOT OpenVINODeveloperPackage_FOUND)
    find_package(OpenVINO ${OV_COMPATIBILITY_VERSION} REQUIRED
                 COMPONENTS Runtime Threading
                 PATHS "${OpenVINO_DIR_PY}")
endif()

include(cmake/features.cmake)
include(cmake/version.cmake)
include(cmake/vs_version.cmake)

if(ENABLE_PYTHON)
    # the following two calls are required for cross-compilation
    if(OpenVINODeveloperPackage_FOUND)
        ov_find_python3(REQUIRED)
        ov_detect_python_module_extension()
    else()
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
            find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
        else()
            find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
        endif()
    endif()
endif()

if(WIN32 OR APPLE)
  set(CMAKE_DEBUG_POSTFIX "d")
endif()

# Workaround for an MSVC compiler issue in some versions of Visual Studio 2022.
# The issue involves a null dereference to a mutex. For details, refer to link https://github.com/microsoft/STL/wiki/Changelog#vs-2022-1710
if(MSVC AND MSVC_VERSION GREATER_EQUAL 1930 AND MSVC_VERSION LESS 1941)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

add_subdirectory(thirdparty)
add_subdirectory(src)
if(EXISTS "${OpenVINOGenAI_SOURCE_DIR}/samples")
    add_subdirectory(samples)
endif()
if(EXISTS "${OpenVINOGenAI_SOURCE_DIR}/tools/continuous_batching" AND ENABLE_TOOLS)
    add_subdirectory(tools/continuous_batching)
endif()
if(EXISTS "${OpenVINOGenAI_SOURCE_DIR}/tests/cpp" AND ENABLE_TESTS)
    add_subdirectory(tests/cpp)
endif()

install(FILES LICENSE DESTINATION docs/licensing COMPONENT licensing_genai RENAME LICENSE-GENAI)
install(FILES third-party-programs.txt DESTINATION docs/licensing COMPONENT licensing_genai RENAME third-party-programs-genai.txt)
if(NOT DEFINED CPACK_ARCHIVE_COMPONENT_INSTALL)
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
endif()
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
# Workaround https://gitlab.kitware.com/cmake/cmake/-/issues/2614
set(CPACK_COMPONENTS_ALL core_genai core_genai_dev core_c_genai core_c_genai_dev cpp_samples_genai licensing_genai openvino_tokenizers openvino_tokenizers_docs)
if(ENABLE_PYTHON)
    list(APPEND CPACK_COMPONENTS_ALL pygenai_${Python3_VERSION_MAJOR}_${Python3_VERSION_MINOR})
    if(NOT ENABLE_GIL_PYTHON_API)
        if(Python3_VERSION VERSION_LESS "3.13")
            message(FATAL_ERROR "Disabling GIL requires Python >= 3.13, but found Python ${Python3_VERSION}")
        else()
            set(Python3_FIND_ABI "ANY" "ANY" "ANY" "ON")
        endif()
    endif()
endif()
if(ENABLE_JS)
    list(APPEND CPACK_COMPONENTS_ALL genai_node_addon)
endif()
if(WIN32 AND NOT DEFINED CPACK_GENERATOR)
    set(CPACK_GENERATOR "ZIP")
endif()
if(CPACK_GENERATOR STREQUAL "NPM")
    set(CPACK_GENERATOR "TGZ")
endif()
include(CPack)
